"""Run a multi-step block task with the camera in the loop.

A task is a list of steps. Each step moves one block to a goal:
    {"object": "red", "goal": {"type": "cell", "cell": "C1"}}
    {"object": "purple", "goal": {"type": "beside", "reference": "red", "side": "right", "gap_mm": 10}}
    {"object": "yellow", "goal": {"type": "on_top", "reference": "red"}}
Goal types and sides follow docs/proposals/video-pipeline (sides are as the printed page reads:
left -x, right +x, top -y toward the arm, bottom +y).

For every step: look at the table, resolve the goal against where things are *now*, pick, place,
park, look again, and check the block landed within tolerance. Any failure stops the task with a
clear reason; nothing after it runs.
"""

import hashlib
import json
import re
import time

import numpy as np

from .correction import CORRECTION_FILE
from .perception.config import BANK_ZONE_MM, CAMERA_XYZ_MM, CUBE_SIZE_MM, ROOT, SHEET_SPEC
from .perception.naming import object_name
from .perception.scene import build_scene
from .perception.segment import in_bank
from .arm_calibration import CALIB_FILE
from .skills import LARGE_ROLL_DEG, SkillError

TOLERANCE_MM = 15.0
MAX_RETRIES = 2          # AGENTS.md: after two failed retries, stop and report
MAX_AIM_SHIFT_MM = 25.0  # a retry aims off by the measured miss, but never further than this
SLIP_MM = MAX_AIM_SHIFT_MM  # a bigger miss is the block slipping in the jaws, not the arm being off:
                            # don't aim (or pick) off by it (run 00:12: a 41 mm slip shifted the
                            # re-pick 33 mm off the block and the jaws closed on nothing)
LEVEL_OFFSETS_NAME = "level_offsets.json"  # in robot/outputs: learned per-height placement errors
LOOK_AROUND_MM = (-80.0, -90.0, 400.0, 330.0)  # where to look for a block the step can't find
MISSING_LOOKS = 4        # a block the step needs (or the sheet) isn't in view: look again this many times...
MISSING_WAIT_S = 2.0     # ...this far apart (the bank is being refilled, or a hand is over it)
FRAME_TRIES, FRAME_WAIT_S = 3, 1.0  # the camera gave no frame: try again before giving up
MAX_FREES = 2            # per step: blocks moved out of the gripper's way (a bank block can be boxed in twice)
MIN_GAP_MM = 8.0         # blocks closer than this (edge to edge) to a target get moved out of the way
                         # (run 21:09: purple 3 mm from B2's footprint caught red's corner on the way
                         # down). Blocks in neighbouring cells stand 9 mm apart, which is fine.
SPOT_STEP_MM = 6.0       # lattice for finding a free spot to move a block to
MAX_LEVELS = 3           # tallest tower: travel height (120 mm) clears a 2-high stack with a cube in hand
SHEET_W_MM, SHEET_H_MM = 215.9, 279.4
SIDES = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


class TaskError(RuntimeError):
    pass


def cell_center(cell):
    g = json.loads(open(SHEET_SPEC).read())["grid"]
    cell = cell.strip().upper()
    if len(cell) != 2 or cell[0] not in "ABC" or cell[1] not in "123":
        raise TaskError(f"no cell {cell!r} (cells are A1-C3)")
    r, c = "ABC".index(cell[0]), int(cell[1]) - 1
    return np.array([g["origin_mm"][0] + (c + 0.5) * g["cell_mm"], g["origin_mm"][1] + (r + 0.5) * g["cell_mm"]])


def _calibration_id():
    """Everything a learned placement error depends on: the arm calibration, the placement
    correction and the pinned camera. Any change retires the lessons (run 03:21: an offset learned
    before the camera was moved and re-pinned pushed a block off its tower)."""
    h = hashlib.sha1()
    for f in (CALIB_FILE, CORRECTION_FILE):
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"-")
    h.update(repr(CAMERA_XYZ_MM).encode())
    return h.hexdigest()[:12]


def _clip(v, limit=MAX_AIM_SHIFT_MM):
    v = np.asarray(v, float)
    return v * min(1.0, limit / max(float(np.hypot(*v)), 1e-9))


def load_level_offsets():
    """{stack height mm: (dx, dy)} the arm was last measured to be off by when working that high,
    for the current arm calibration only. The touch calibration is all at table height, and the
    error grows with height (run 20:52: 8 mm at the table, 5 at 64 mm up, 20 at 112 mm up)."""
    path = ROOT / "outputs" / LEVEL_OFFSETS_NAME
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if data.get("calibration") != _calibration_id():
        return {}  # recalibrated since: start over
    return {float(k): np.asarray(v, float) for k, v in data.get("levels", {}).items()}


def save_level_offsets(levels):
    path = ROOT / "outputs" / LEVEL_OFFSETS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"calibration": _calibration_id(), "updated": time.time(),
                                "levels": {str(k): np.round(v, 1).tolist() for k, v in levels.items() if k > 0}},
                               indent=1))


def grid_box():
    """The printed 3x3 grid as (x0, y0, x1, y1) sheet mm: what parking must leave in view."""
    g = json.loads(open(SHEET_SPEC).read())["grid"]
    (x0, y0), side = g["origin_mm"], 3 * g["cell_mm"]
    return (x0, y0, x0 + side, y0 + side)


def view_box():
    """What parking must leave in the camera's view: the grid, and the cube bank beside it
    (parking over the bank hid its blocks from the next look)."""
    x0, y0, x1, y1 = grid_box()
    if BANK_ZONE_MM is None:
        return (x0, y0, x1, y1)
    bx0, by0, bx1, by1 = BANK_ZONE_MM
    return (min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1))


def by_colour(objects):
    """{name: object}, numbering repeats of a colour within this group only ('red_cube',
    'red_cube_2'), so a spare red in the bank never renames the red on the sheet."""
    out, seen = {}, {}
    for o in objects:
        base = re.sub(r"_\d+$", "", o["name"])
        seen[base] = seen.get(base, 0) + 1
        name = base if seen[base] == 1 else f"{base}_{seen[base]}"
        out[name] = {**o, "name": name}
    return out


def at_height(center_mm, camera_xyz, base_mm):
    """The scene places every blob as if its cube stood on the table. For a cube standing on
    something base_mm tall, move that point along the camera ray to the right height (run 19:14:
    a stacked cube read 31 mm off when it was ~10 mm off)."""
    xy = np.asarray(center_mm, float)
    if not base_mm or camera_xyz is None:
        return xy
    c = np.asarray(camera_xyz, float)
    mid = CUBE_SIZE_MM / 2
    return c[:2] + (xy - c[:2]) * (c[2] - base_mm - mid) / (c[2] - mid)


def describe(step):
    g = step["goal"]
    where = {"cell": lambda: f"cell {g['cell'].upper()}",
             "beside": lambda: f"{g.get('side', 'right')} of {g['reference']}",
             "on_top": lambda: f"on top of {g['reference']}",
             "spot": lambda: f"out of the way, to ({g['xy'][0]:.0f}, {g['xy'][1]:.0f})"}[g["type"]]()
    return f"{step['object']} -> {where}"


def colour_of(name):
    """'light_blue_cube' -> 'light blue' (what a step's "object" holds)."""
    return name[:-len("_cube")].replace("_", " ") if name.endswith("_cube") else name


def edge_gap(a, b):
    """Edge-to-edge gap (mm) between two upright, square-on cubes centred at a and b."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) - CUBE_SIZE_MM
    return float(max(d[0], d[1]))


def too_close(a, b, gap=MIN_GAP_MM):
    return edge_gap(a, b) < gap


class Executor:
    def __init__(self, skills, cam, loc, rec=None, log=print):
        self.sk, self.cam, self.loc, self.rec, self.log = skills, cam, loc, rec, log
        self.camera_xyz = None
        # Stacks built during this run (every block is assumed to start on the table):
        self.level_err = load_level_offsets()  # stack height -> how far off the arm works there (clean evidence, kept)
        self.step_err = {}  # the same, from this step's own landings, for its retries: a miss repeats to ~1 mm
                            # for the same move whatever caused it (run 04:21: three 22 mm misses at roll 84)
        for lvl, v in sorted(self.level_err.items()):
            self.log(f"  remembered from earlier runs: the arm lands ({v[0]:+.0f}, {v[1]:+.0f}) mm off at {lvl:.0f} mm up")
        self.under = {}    # block -> the block it sits on
        self.last_xy = {}  # block -> where it was last measured landing (covered blocks can't be seen)
        self.bank = {}     # spare blocks in the cube bank beside the sheet, from the last look
        self.placed = set()  # blocks this task has put where a step wanted them: never shuffled aside

    def _learn(self, level, miss, applied, err):
        """Update the remembered error at this stack height from a clean landing. `miss` is the
        arm's actual error this time (landed minus aimed); `applied` the offset this attempt
        already aimed off by; `err` how far from the target it landed. An offset that made the
        landing worse than no offset (err > |miss|) was wrong: replace it with this measurement.
        One that helped is averaged in, so a single odd landing can't swing it."""
        old = self.level_err.get(level)
        if old is None or not np.any(applied):
            new = _clip(miss)
        elif err > np.hypot(*miss) + 1.0:
            new = _clip(miss)
            self.log(f"  the {np.hypot(*applied):.0f} mm offset remembered for {level:.0f} mm up made the landing "
                     f"worse: replaced by this landing's {np.hypot(*miss):.0f} mm")
        else:
            new = _clip(0.5 * np.asarray(old, float) + 0.5 * np.asarray(miss, float))
        self.level_err[level] = new
        if level > 0:
            save_level_offsets(self.level_err)

    def offset(self, level):
        """How far off the arm was last measured working at this stack height (zero if unknown):
        this step's own evidence first, then what earlier clean landings taught."""
        if level in self.step_err:
            return self.step_err[level]
        return self.level_err.get(level, np.zeros(2))

    def base(self, name):
        """Height (mm) of what `name` stands on: 0 on the table, 48 on one block, ..."""
        return 0.0 if name not in self.under else self.base(self.under[name]) + CUBE_SIZE_MM

    def on_top_of(self, name):
        return next((n for n, u in self.under.items() if u == name), None)

    def pos(self, name, objs):
        """Where `name` really is: a covered block by its last landing (the camera sees only a
        sliver of it), a stacked block measured at its height, otherwise as the scene says."""
        if self.on_top_of(name) is not None and name in self.last_xy:
            return np.asarray(self.last_xy[name], float)
        return at_height(objs[name]["center_mm"], self.camera_xyz, self.base(name))

    def track_tallest(self, objs):
        """Tell the arm how tall the tallest stack is, so it travels high enough over it."""
        self.sk.tallest_mm = max([self.base(n) + CUBE_SIZE_MM for n in objs] + [CUBE_SIZE_MM])

    def neighbours(self, name, objs):
        """Other blocks, the bank's included, as (x, y, height of their top) for the grip planner."""
        return ([(*self.pos(n, objs), self.base(n) + CUBE_SIZE_MM) for n in objs if n != name]
                + [(*o["center_mm"], CUBE_SIZE_MM) for o in self.bank.values()])

    def _frame(self):
        """A camera frame, trying again for a few seconds if the camera stalls."""
        err = None
        for _ in range(FRAME_TRIES):
            try:
                self.last_frame = self.cam.read()
                return self.last_frame
            except RuntimeError as e:
                err = e
                self.log(f"  camera: {e}; trying again")
                time.sleep(FRAME_WAIT_S)
        raise TaskError(f"the camera gave no frame for {FRAME_TRIES * FRAME_WAIT_S:.0f} s ({err})")

    def observe(self, source=None):
        """Visible blocks on the sheet by object name; touching/merged blobs are left out (can't
        be grasped). Blocks in the cube bank go to self.bank. With source (the block a step
        moves): if the sheet has none of that colour, a bank block of it is returned under that
        name, so the step builds from the bank. A sheet the camera can't find (hand, arm or a
        block over the corner tags) gets the arm parked and a few more looks."""
        scene, _, _ = build_scene(self._frame(), self.loc)
        for _ in range(MISSING_LOOKS):
            if scene is not None:
                break
            self.log("  the camera can't see the sheet (need 2+ tags): parking and looking again")
            self.sk.park(camera_xyz=self.camera_xyz, keep_clear=view_box())
            time.sleep(MISSING_WAIT_S)
            scene, _, _ = build_scene(self._frame(), self.loc)
        if scene is None:
            raise TaskError(f"the camera can't see the sheet (need 2+ tags) after {MISSING_LOOKS} looks: "
                            "clear the corner tags")
        self.camera_xyz = scene["sheet"]["camera_xyz_mm"]
        merged = [o["name"] for o in scene["objects"] if o["score"] < 0]
        if merged:
            self.log(f"  note: {merged} look like touching blocks merged together; they can't be picked")
        good = [o for o in scene["objects"] if o["score"] >= 0]
        objs = by_colour([o for o in good if not in_bank(o["center_mm"])])
        self.bank = by_colour([o for o in good if in_bank(o["center_mm"])])
        if source and source not in objs and source in self.bank:
            objs[source] = self.bank.pop(source)
        return objs

    def resolve(self, step, objs):
        """(target xy, yaw, height of what it lands on) for this step, from the live scene."""
        g = step["goal"]
        kind = g["type"]
        if kind == "cell":
            return cell_center(g["cell"]), float(step.get("yaw_deg") or 0.0), 0.0
        if kind == "spot":  # internal: a block being moved out of the way
            return np.asarray(g["xy"], float), 0.0, 0.0
        ref = object_name(g["reference"])
        if ref not in objs:
            raise TaskError(f"can't see the reference block {g['reference']!r} (sees {sorted(objs)})")
        ref_xy, ref_yaw = self.pos(ref, objs), objs[ref]["yaw_deg"]
        if kind == "on_top":
            top = self.on_top_of(ref)
            if top is not None:
                raise TaskError(f"{g['reference']} already has {top} on top of it")
            on = self.base(ref) + CUBE_SIZE_MM
            if on >= MAX_LEVELS * CUBE_SIZE_MM:
                raise TaskError(f"the tower would be more than {MAX_LEVELS} blocks tall")
            return ref_xy, ref_yaw, on
        if kind == "beside":
            side = g.get("side", "right")
            if side not in SIDES:
                raise TaskError(f"side must be one of {sorted(SIDES)}, not {side!r}")
            target = ref_xy + np.array(SIDES[side]) * (CUBE_SIZE_MM + float(g.get("gap_mm", 10.0)))
            if not (CUBE_SIZE_MM / 2 <= target[0] <= SHEET_W_MM - CUBE_SIZE_MM / 2
                    and CUBE_SIZE_MM / 2 <= target[1] <= SHEET_H_MM - CUBE_SIZE_MM / 2):
                raise TaskError(f"{side} of {g['reference']} is off the sheet ({target.round(0).tolist()} mm)")
            return target, ref_yaw, 0.0
        raise TaskError(f"unknown goal type {kind!r}")

    def _move(self, name, objs, src_xy, src_yaw, src_base, aim, yaw, on):
        """Pick `name` from src_xy (standing on src_base mm), place it at aim, park out of view."""
        heights = self.neighbours(name, objs)
        pick_grip, place_grip = self.sk.plan_move(src_xy, src_yaw, aim, yaw, on, pick_neighbours=heights,
                                                  place_neighbours=heights, src_base_mm=src_base)
        self.sk.pick(src_xy, src_yaw, grip=pick_grip, base_mm=src_base)
        self.sk.place(aim, yaw, on, grip=place_grip)
        self.sk.park(camera_xyz=self.camera_xyz, keep_clear=view_box())
        return pick_grip, place_grip

    def _landed(self, obj, target, on):
        """(where the moved block is, what it stands on). For a stacking step it is measured at
        stack height, unless it is clearly not on the stack (fell off): then on the table."""
        if on:
            up = at_height(obj["center_mm"], self.camera_xyz, on)
            if np.hypot(*(up - target)) < 0.75 * CUBE_SIZE_MM:
                return up, on
        return np.asarray(obj["center_mm"], float), 0.0

    def run(self, steps):
        report = {"started": time.time(), "steps": []}
        self.report = report
        try:
            for i, step in enumerate(steps, 1):
                self.log(f"\nStep {i}/{len(steps)}: {describe(step)}")
                entry = {"step": i, "description": describe(step)}
                report["steps"].append(entry)
                self._do_step(step, entry, later=steps[i:])
            report["result"] = "done"
        except (TaskError, SkillError) as e:
            report["result"] = f"stopped: {e}"
            if getattr(self.sk, "holding", None) is not None:  # don't stop with a block in the jaws
                try:
                    self.sk.put_back()
                    self.sk.park(camera_xyz=self.camera_xyz, keep_clear=view_box())
                    report["result"] += "; put the block back where it was picked up"
                except SkillError as e2:
                    self.log(f"  couldn't put the block back ({e2}); rest will set it down beside the sheet")
            if report["steps"]:
                report["steps"][-1].setdefault("ok", False)
            raise TaskError(str(e)) from None
        finally:
            report["finished"] = time.time()
            out = ROOT / "outputs" / "tasks" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            self.report_path = out
        return report

    # -- clearing the way ------------------------------------------------------
    def _keep(self, step, name):
        """Blocks a step must not move aside: the one it moves, its reference, and what's under it."""
        keep = {name}
        g = step["goal"]
        if g["type"] in ("on_top", "beside"):
            ref = object_name(g["reference"])
            keep.add(ref)
            while ref in self.under:  # the whole stack the reference stands on
                ref = self.under[ref]
                keep.add(ref)
        return keep

    def _future_targets(self, later):
        return [cell_center(s["goal"]["cell"]) for s in later if s["goal"]["type"] == "cell"]

    def free_spot(self, mover, objs, avoid, away_from=None):
        """Where to move `mover` out of the way of `away_from`: the nearest empty cell centre
        (cells keep blocks 9 mm apart, as designed), else any clear spot in the grid, that is
        farther from `away_from` than the block stands now (run 04:48: the "nearest free cell"
        was the block's own, and it was lifted and set straight back down). Never on the `avoid`
        points (this step's and later steps' targets)."""
        here = self.pos(mover, objs) if mover in objs else np.asarray(self.bank[mover]["center_mm"], float)
        away_from = here if away_from is None else np.asarray(away_from, float)
        now = float(np.hypot(*(here - away_from)))  # with no away_from: any spot but this one
        covered = [np.asarray(self.last_xy[n], float) for n in set(self.under.values())
                   if n not in objs and n in self.last_xy]  # under a stack: the camera can't see it
        others = [self.pos(n, objs) for n in objs if n != mover] + covered + [np.asarray(a, float) for a in avoid]

        def free(p):
            return (all(edge_gap(p, o) >= MIN_GAP_MM for o in others)
                    and float(np.hypot(*(p - away_from))) >= now + MIN_GAP_MM)

        cells = [cell_center(f"{r}{c}") for r in "ABC" for c in "123"]
        x0, y0, x1, y1 = grid_box()
        half = CUBE_SIZE_MM / 2
        lattice = [np.array([x, y]) for x in np.append(np.arange(x0 + half, x1 - half, SPOT_STEP_MM), x1 - half)
                   for y in np.append(np.arange(y0 + half, y1 - half, SPOT_STEP_MM), y1 - half)]
        for candidates in (cells, lattice):
            ok = [p for p in candidates if free(p)]
            if ok:
                return min(ok, key=lambda p: float(np.hypot(*(p - here))))
        return None

    def _move_aside(self, blocker, objs, why, avoid, away_from):
        """Move a loose block out of the way. Returns False when nothing useful can be moved:
        the block is part of a stack (run 04:48: the tower's top block was lifted off and set down
        into its own tower), or the best free spot is where it already stands."""
        if self.on_top_of(blocker) is not None or self.base(blocker) > 0:
            self.log(f"  {colour_of(blocker)} is {why}, but it is part of a stack: leaving it")
            return False
        spot = self.free_spot(blocker, objs, avoid, away_from)
        if spot is None:
            raise TaskError(f"{colour_of(blocker)} is {why}, and there is no clearer spot to move it to")
        step = {"object": colour_of(blocker), "goal": {"type": "spot", "xy": spot.round(1).tolist()}}
        self.log(f"\n  Clearing the way: {colour_of(blocker)} is {why}; moving it to ({spot[0]:.0f}, {spot[1]:.0f})")
        entry = {"step": "aside", "description": describe(step)}
        self.report["steps"].append(entry)
        self._do_step(step, entry, later=[], clearing=True)
        return True

    def _clear_target(self, step, name, target, on, objs, later):
        """Move every block that crowds a table-level target out of the way first."""
        if on > 0:
            return False  # the jaws work above the block below; the planner checks taller neighbours
        keep = self._keep(step, name)
        moved = False
        for other in sorted(objs, key=lambda n: float(np.hypot(*(self.pos(n, objs) - target)))):
            if (other in keep or other in self.placed or self.base(other) > 0
                    or not too_close(self.pos(other, objs), target)):
                continue  # placed by this task, or stacked: part of the build, not clutter
            if self._move_aside(other, objs, "too close to the target", [target] + self._future_targets(later),
                                away_from=target):
                objs = self.observe()
                moved = True
        return moved

    def _missing(self, colour, name, objs):
        """Why a step's block can't be used, for the error message: what the sheet and the bank
        hold, and where a block of that colour is if it sits outside both (another look, wider)."""
        sees = (f"sheet: {', '.join(sorted(objs)) or 'nothing'}; "
                f"bank: {', '.join(f'{n} ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f})' for n, o in self.bank.items()) or 'nothing'}")
        msg = f"can't see {colour!r} on the sheet or in the cube bank ({sees})"
        if not hasattr(self, "last_frame") or BANK_ZONE_MM is None:
            return msg
        try:
            scene, _, _ = build_scene(self.last_frame, self.loc, naming="local", bank_zone=LOOK_AROUND_MM)
        except Exception as e:  # only for the message: never hide the real problem behind this one
            return msg + f" (wider look failed: {e})"
        hits = [o for o in (scene or {}).get("objects", [])
                if re.sub(r"_\d+$", "", o["name"]) == name and not in_bank(o["center_mm"])
                and not (0 <= o["center_mm"][0] <= SHEET_W_MM and 0 <= o["center_mm"][1] <= SHEET_H_MM)]
        if not hits:
            return msg + "; none nearby either: is it on the table, and named that colour?"
        x, y = hits[0]["center_mm"]
        x0, y0, x1, y1 = BANK_ZONE_MM
        return (msg + f"; there is one at ({x:.0f}, {y:.0f}) mm, outside the bank (block centres must be "
                f"within x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f} mm, where the arm can grip): move it into the bank")

    def _wait_for(self, names, objs, source):
        """Look again, a few times, while any of `names` is out of view (the bank is being
        refilled, a hand is over the sheet, the parked arm hides a block). Returns the last look."""
        for _ in range(MISSING_LOOKS):
            missing = [colour_of(n) for n in names if n not in objs]
            if not missing:
                break
            self.log(f"  can't see {', '.join(missing)} yet (refilling the bank? a hand in view?); "
                     f"looking again in {MISSING_WAIT_S:.0f} s")
            time.sleep(MISSING_WAIT_S)
            objs = self.observe(source=source)
        return objs

    # -- one step ----------------------------------------------------------------
    def _do_step(self, step, entry, later, clearing=False):
        name = object_name(step["object"])
        g = step["goal"]
        ref = object_name(g["reference"]) if g["type"] in ("on_top", "beside") else None
        objs = self._wait_for([n for n in (name, ref) if n], self.observe(source=name), source=name)
        if name not in objs:
            raise TaskError(self._missing(step["object"], name, objs))
        self.track_tallest(objs)
        top = self.on_top_of(name)
        if top is not None:
            raise TaskError(f"{step['object']} has {colour_of(top)} on top of it; move {colour_of(top)} first")
        target, yaw, on = self.resolve(step, objs)
        if not clearing and self._clear_target(step, name, target, on, objs, later):
            objs = self.observe(source=name)
            target, yaw, on = self.resolve(step, objs)  # a beside/on target follows its reference
        keep = self._keep(step, name)
        for other in list(objs):  # don't drop a block onto another one by accident
            if other != name and on == 0.0 and self.base(other) == 0 \
                    and np.hypot(*(self.pos(other, objs) - target)) < CUBE_SIZE_MM:
                if other in self.placed:
                    raise TaskError(f"the target is occupied by {colour_of(other)}, which this task put there: "
                                    "two steps want the same cell")
                if clearing or other in keep or not self._move_aside(
                        other, objs, "sitting on the target", [target] + self._future_targets(later), away_from=target):
                    raise TaskError(f"the target is occupied by {colour_of(other)}")
                objs = self.observe(source=name)
                target, yaw, on = self.resolve(step, objs)
                break
        src = objs[name]
        src_xy = self.pos(name, objs)
        if in_bank(src["center_mm"]):
            self.log(f"  taking {step['object']} from the cube bank at "
                     f"({src['center_mm'][0]:.0f}, {src['center_mm'][1]:.0f}) mm")
        self.log(f"  {name} at ({src_xy[0]:.0f}, {src_xy[1]:.0f}) -> "
                 f"({target[0]:.0f}, {target[1]:.0f}) mm" + (f" on a {on:.0f} mm block" if on else ""))
        if self.rec:
            self.rec("plan", cube_xy=src_xy.tolist(), target_xy=target.tolist())
        t = time.time()
        src_yaw, src_base = src["yaw_deg"], self.base(name)
        self.level_err.pop(0.0, None)  # on the table the miss depends on where: learn it per step
        self.step_err = {}
        entry["attempts"] = []
        freed = 0
        attempt = 0       # place retries (landed off target)
        grasp_fails = 0   # grabs that closed on nothing: look again, pick where the block is now
        while True:
            # The arm is off by about the same amount for picks and places at the same height,
            # so both get the learned offset (run 20:52: correcting only the place left the
            # re-picked block 17 mm off-centre in the jaws, and it toppled off the tower).
            applied = self.offset(on).copy()  # what this attempt aims off by, from earlier evidence
            aim = target - applied
            pick_at = np.asarray(src_xy, float) - self.offset(src_base)
            if attempt:
                self.log(f"  retry {attempt}/{MAX_RETRIES}: picking {name} up again, aiming at "
                         f"({aim[0]:.0f}, {aim[1]:.0f}) to cancel the miss")
            elif np.any(self.offset(on)):
                self.log(f"  aiming {np.hypot(*self.offset(on)):.0f} mm off to cancel the error "
                         f"learned at {on:.0f} mm up")
            try:
                from_bank = attempt == 0 and in_bank(src["center_mm"])
                grips = self._move(name, objs, pick_at, src_yaw, src_base, aim, yaw, on)
                if from_bank and grips and grips[0] and abs(grips[0]["roll"]) > LARGE_ROLL_DEG:
                    self.log(f"  note: that bank pick needed a {grips[0]['roll']:.0f} deg wrist roll; the arm is "
                             f"15-20 mm off at big rolls. Set bank blocks down as diamonds (turned ~45 deg).")
            except SkillError as e:
                grab_failed = any(w in str(e) for w in ("closed on nothing", "slipped out", "grip aborted"))
                if grab_failed and grasp_fails < MAX_RETRIES:
                    # The block isn't where the arm reached (pushed, dropped, or the arm is off there):
                    # don't give up or guess, look. The camera says where it is now; pick it up there.
                    grasp_fails += 1
                    entry["grasp_retries"] = grasp_fails
                    if self.rec:
                        self.rec("error", reason=str(e))
                    self.sk.park(camera_xyz=self.camera_xyz, keep_clear=view_box())
                    objs = self.observe(source=name)
                    if name not in objs:
                        raise TaskError(f"the jaws closed on nothing at ({src_xy[0]:.0f}, {src_xy[1]:.0f}) and now "
                                        + self._missing(step["object"], name, objs))
                    now = self.pos(name, objs)
                    moved = float(np.hypot(*(now - np.asarray(src_xy, float))))
                    self.log(f"  the grab at ({src_xy[0]:.0f}, {src_xy[1]:.0f}) failed ({e}); looked again: "
                             f"{step['object']} is at ({now[0]:.0f}, {now[1]:.0f}), {moved:.0f} mm from there. "
                             f"Retry {grasp_fails}/{MAX_RETRIES}, picking it up where it is now")
                    src, src_xy, src_yaw, src_base = objs[name], now, objs[name]["yaw_deg"], self.base(name)
                    target, yaw, on = self.resolve(step, objs)
                    continue
                if "backed off" in str(e) and attempt < MAX_RETRIES:
                    # The descent to the target hit something (run 04:48: a block hidden under a
                    # stack). Put the block back, look at what is there now, clear it if it is loose, retry.
                    attempt += 1
                    if self.rec:
                        self.rec("error", reason=str(e))
                    self.log(f"  the descent to ({aim[0]:.0f}, {aim[1]:.0f}) hit something ({e}): putting the "
                             f"block back, looking again, retry {attempt}/{MAX_RETRIES}")
                    if getattr(self.sk, "holding", None) is not None:
                        self.sk.put_back()
                    self.sk.park(camera_xyz=self.camera_xyz, keep_clear=view_box())
                    objs = self._wait_for([name], self.observe(source=name), source=name)
                    if name not in objs:
                        raise TaskError(self._missing(step["object"], name, objs))
                    if not clearing and self._clear_target(step, name, target, on, objs, later):
                        objs = self.observe(source=name)
                    target, yaw, on = self.resolve(step, objs)
                    src, src_xy, src_yaw, src_base = objs[name], self.pos(name, objs), objs[name]["yaw_deg"], self.base(name)
                    continue
                # No grip fits (planned before any motion): move the closest neighbour aside, on the
                # sheet or in the bank (run 02:30: red at the top of the bank was boxed in by
                # purple, and only sheet blocks were considered, so the task stopped).
                if clearing or freed >= MAX_FREES or "blocked by a neighbouring block" not in str(e):
                    raise
                where = target if "target" in str(e) else src_xy
                keep = self._keep(step, name)
                near = {n: self.pos(n, objs) for n in objs
                        if n not in keep and n not in self.placed and self.base(n) == 0}
                # A step names a block by colour, so leave bank blocks it would confuse: a repeat
                # ('red_cube_2'), or a colour the sheet already has (that block would be moved).
                near.update({n: np.asarray(o["center_mm"], float) for n, o in self.bank.items()
                             if n not in keep and n not in objs and not re.search(r"_\d+$", n)})
                if not near:
                    raise
                blocker = min(near, key=lambda n: float(np.hypot(*(near[n] - where))))
                if not self._move_aside(blocker, objs, "in the way of the gripper",
                                        [target, src_xy] + self._future_targets(later), away_from=where):
                    raise
                freed += 1
                objs = self.observe(source=name)
                target, yaw, on = self.resolve(step, objs)
                src_xy = self.pos(name, objs)
                continue
            objs = self._wait_for([name], self.observe(source=name), source=name)
            if name not in objs:  # the parked arm may hide it: look from the other parking spot
                self.sk.park()
                objs = self.observe(source=name)
            if name not in objs:
                raise TaskError(f"after placing, can't see {step['object']!r} (dropped? hidden?)")
            got, got_base = self._landed(objs[name], target, on)
            err = float(np.hypot(*(got - target)))
            entry["attempts"].append({"aim_mm": aim.round(1).tolist(), "landed_mm": got.round(1).tolist(),
                                      "error_mm": round(err, 1)})
            entry.update(landed_mm=got.round(1).tolist(), error_mm=round(err, 1), retries=attempt,
                         seconds=round(time.time() - t, 1))
            if self.rec:
                self.rec("verify", target_xy=target.tolist(), attempt=attempt, error_mm=round(err, 1))
            miss = got - aim
            rolls = [abs(float(g["roll"])) for g in (grips or ()) if g and "roll" in g]
            place_roll = abs(float(grips[1]["roll"])) if grips and grips[1] and "roll" in grips[1] else 0.0
            if got_base != on:
                pass  # it fell to another level: that says nothing about the arm at this height
            elif from_bank and place_roll <= LARGE_ROLL_DEG:
                # The miss of a block carried in from the bank is mostly how far off the arm was AT
                # THE BANK (it sets the block off-centre in the jaws); the retry re-picks on the
                # sheet, where that error is different, so it must not aim off by it (run 02:45:
                # a 23 mm miss learned, re-pick 23 mm off the block, jaws closed on nothing).
                self.log(f"  missed by {np.hypot(*miss):.0f} mm carrying a block from the bank: the retry "
                         f"picks it where the camera sees it and aims straight at the target")
            elif np.hypot(*miss) > SLIP_MM:
                self.log(f"  missed by {np.hypot(*miss):.0f} mm: the block slipped, so the retry "
                         f"picks it where the camera sees it and aims straight at the target")
            else:
                # The retry aims off by this miss: for the same move it repeats to ~1 mm.
                self.step_err[on] = _clip(miss)
                if rolls and max(rolls) > LARGE_ROLL_DEG or from_bank:
                    self.log(f"  (that miss is remembered for this step's retries only: the wrist rolled "
                             f"{max(rolls):.0f} deg, so it isn't the arm's error at this height in general)")
                else:  # clean evidence: learn how far off the arm works at this height, for good
                    self._learn(on, miss, applied, err)
            if err <= TOLERANCE_MM:
                break
            if attempt == MAX_RETRIES:
                raise TaskError(f"{step['object']} still landed {err:.0f} mm from its target after "
                                f"{MAX_RETRIES} retries (tolerance {TOLERANCE_MM:.0f})")
            self.log(f"  missed: landed {err:.0f} mm from target (tolerance {TOLERANCE_MM:.0f})")
            # Misses repeat to ~1 mm for the same move (B2 runs 18:34, 18:47): the next attempt
            # aims off by the offset just learned (see the top of this loop).
            src_xy, src_yaw, src_base = got.tolist(), objs[name]["yaw_deg"], got_base
            attempt += 1
        entry["ok"] = True
        self.last_xy[name] = got.tolist()
        if not clearing:
            self.placed.add(name)
        ref = object_name(step["goal"]["reference"]) if step["goal"]["type"] == "on_top" else None
        if ref and got_base > 0:
            self.under[name] = ref
        else:
            self.under.pop(name, None)
        self.track_tallest(objs)  # before the next move, park and rest: clear the new stack
        self.log(f"  OK: landed {err:.1f} mm from target in {entry['seconds']} s"
                 + (f" ({attempt} {'retry' if attempt == 1 else 'retries'})" if attempt else ""))
