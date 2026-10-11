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
import time

import numpy as np

from .perception.config import CUBE_SIZE_MM, ROOT, SHEET_SPEC
from .perception.naming import object_name
from .perception.scene import build_scene
from .arm_calibration import CALIB_FILE
from .skills import SkillError

TOLERANCE_MM = 15.0
MAX_RETRIES = 2          # AGENTS.md: after two failed retries, stop and report
MAX_AIM_SHIFT_MM = 25.0  # a retry aims off by the measured miss, but never further than this
LEVEL_OFFSETS_NAME = "level_offsets.json"  # in robot/outputs: learned per-height placement errors
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
    try:
        return hashlib.sha1(CALIB_FILE.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


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
        self.level_err = load_level_offsets()  # stack height -> how far off the arm works there
        self.under = {}    # block -> the block it sits on
        self.last_xy = {}  # block -> where it was last measured landing (covered blocks can't be seen)

    def offset(self, level):
        """How far off the arm was last measured working at this stack height (zero if unknown)."""
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
        """Other blocks as (x, y, height of their top) for the grip planner."""
        return [(*self.pos(n, objs), self.base(n) + CUBE_SIZE_MM) for n in objs if n != name]

    def observe(self):
        """Visible blocks by object name; touching/merged blobs are left out (can't be grasped)."""
        scene, _, _ = build_scene(self.cam.read(), self.loc)
        if scene is None:
            raise TaskError("the camera can't see the sheet (need 2+ tags)")
        self.camera_xyz = scene["sheet"]["camera_xyz_mm"]
        merged = [o["name"] for o in scene["objects"] if o["score"] < 0]
        if merged:
            self.log(f"  note: {merged} look like touching blocks merged together; they can't be picked")
        return {o["name"]: o for o in scene["objects"] if o["score"] >= 0}

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
        self.sk.park(camera_xyz=self.camera_xyz, keep_clear=grid_box())

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
                    self.sk.park(camera_xyz=self.camera_xyz, keep_clear=grid_box())
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

    def free_spot(self, mover, objs, avoid):
        """Where to move `mover` out of the way: the nearest empty cell centre (cells keep blocks
        9 mm apart, as designed), else any spot in the grid that clear. Never on the `avoid`
        points (this step's and later steps' targets)."""
        here = self.pos(mover, objs)
        others = [self.pos(n, objs) for n in objs if n != mover] + [np.asarray(a, float) for a in avoid]

        def free(p):
            return all(edge_gap(p, o) >= MIN_GAP_MM for o in others)

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

    def _move_aside(self, blocker, objs, why, avoid):
        if self.on_top_of(blocker) is not None:
            raise TaskError(f"{colour_of(blocker)} is {why}, but it has a block on top of it; clear it by hand")
        spot = self.free_spot(blocker, objs, avoid)
        if spot is None:
            raise TaskError(f"{colour_of(blocker)} is {why}, and there is no free spot to move it to")
        step = {"object": colour_of(blocker), "goal": {"type": "spot", "xy": spot.round(1).tolist()}}
        self.log(f"\n  Clearing the way: {colour_of(blocker)} is {why}; moving it to ({spot[0]:.0f}, {spot[1]:.0f})")
        entry = {"step": "aside", "description": describe(step)}
        self.report["steps"].append(entry)
        self._do_step(step, entry, later=[], clearing=True)

    def _clear_target(self, step, name, target, on, objs, later):
        """Move every block that crowds a table-level target out of the way first."""
        if on > 0:
            return False  # the jaws work above the block below; the planner checks taller neighbours
        keep = self._keep(step, name)
        moved = False
        for other in sorted(objs, key=lambda n: float(np.hypot(*(self.pos(n, objs) - target)))):
            if other in keep or not too_close(self.pos(other, objs), target):
                continue
            self._move_aside(other, objs, "too close to the target", [target] + self._future_targets(later))
            objs = self.observe()
            moved = True
        return moved

    # -- one step ----------------------------------------------------------------
    def _do_step(self, step, entry, later, clearing=False):
        name = object_name(step["object"])
        objs = self.observe()
        if name not in objs:
            raise TaskError(f"can't see {step['object']!r} (sees {sorted(objs)})")
        self.track_tallest(objs)
        top = self.on_top_of(name)
        if top is not None:
            raise TaskError(f"{step['object']} has {colour_of(top)} on top of it; move {colour_of(top)} first")
        target, yaw, on = self.resolve(step, objs)
        if not clearing and self._clear_target(step, name, target, on, objs, later):
            objs = self.observe()
            target, yaw, on = self.resolve(step, objs)  # a beside/on target follows its reference
        for other in objs:  # don't drop a block onto another one by accident
            if other != name and on == 0.0 and np.hypot(*(self.pos(other, objs) - target)) < CUBE_SIZE_MM:
                raise TaskError(f"the target is occupied by {colour_of(other)}")
        src = objs[name]
        src_xy = self.pos(name, objs)
        self.log(f"  {name} at ({src_xy[0]:.0f}, {src_xy[1]:.0f}) -> "
                 f"({target[0]:.0f}, {target[1]:.0f}) mm" + (f" on a {on:.0f} mm block" if on else ""))
        if self.rec:
            self.rec("plan", cube_xy=src_xy.tolist(), target_xy=target.tolist())
        t = time.time()
        src_yaw, src_base = src["yaw_deg"], self.base(name)
        self.level_err.pop(0.0, None)  # on the table the miss depends on where: learn it per step
        entry["attempts"] = []
        freed = False
        attempt = 0
        while True:
            # The arm is off by about the same amount for picks and places at the same height,
            # so both get the learned offset (run 20:52: correcting only the place left the
            # re-picked block 17 mm off-centre in the jaws, and it toppled off the tower).
            aim = target - self.offset(on)
            pick_at = np.asarray(src_xy, float) - self.offset(src_base)
            if attempt:
                self.log(f"  retry {attempt}/{MAX_RETRIES}: picking {name} up again, aiming at "
                         f"({aim[0]:.0f}, {aim[1]:.0f}) to cancel the miss")
            elif np.any(self.offset(on)):
                self.log(f"  aiming {np.hypot(*self.offset(on)):.0f} mm off to cancel the error "
                         f"learned at {on:.0f} mm up")
            try:
                self._move(name, objs, pick_at, src_yaw, src_base, aim, yaw, on)
            except SkillError as e:
                # No grip fits (planned before any motion): move the closest neighbour aside, once.
                if clearing or freed or "blocked by a neighbouring block" not in str(e):
                    raise
                where = target if "target" in str(e) else src_xy
                keep = self._keep(step, name)
                near = [n for n in objs if n not in keep]
                if not near:
                    raise
                blocker = min(near, key=lambda n: float(np.hypot(*(self.pos(n, objs) - where))))
                self._move_aside(blocker, objs, "in the way of the gripper",
                                 [target, src_xy] + self._future_targets(later))
                freed = True
                objs = self.observe()
                target, yaw, on = self.resolve(step, objs)
                src_xy = self.pos(name, objs)
                continue
            objs = self.observe()
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
            if got_base == on:  # still where it was put: learn how far off the arm works at this height
                self.level_err[on] = _clip(got - aim)
                if on > 0:
                    save_level_offsets(self.level_err)
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
        ref = object_name(step["goal"]["reference"]) if step["goal"]["type"] == "on_top" else None
        if ref and got_base > 0:
            self.under[name] = ref
        else:
            self.under.pop(name, None)
        self.track_tallest(objs)  # before the next move, park and rest: clear the new stack
        self.log(f"  OK: landed {err:.1f} mm from target in {entry['seconds']} s"
                 + (f" ({attempt} {'retry' if attempt == 1 else 'retries'})" if attempt else ""))
