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

import json
import time

import numpy as np

from .perception.config import CUBE_SIZE_MM, ROOT, SHEET_SPEC
from .perception.naming import object_name
from .perception.scene import build_scene
from .skills import SkillError

TOLERANCE_MM = 15.0
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


def describe(step):
    g = step["goal"]
    where = {"cell": lambda: f"cell {g['cell'].upper()}",
             "beside": lambda: f"{g.get('side', 'right')} of {g['reference']}",
             "on_top": lambda: f"on top of {g['reference']}"}[g["type"]]()
    return f"{step['object']} -> {where}"


class Executor:
    def __init__(self, skills, cam, loc, rec=None, log=print):
        self.sk, self.cam, self.loc, self.rec, self.log = skills, cam, loc, rec, log

    def observe(self):
        """Visible blocks by object name; touching/merged blobs are left out (can't be grasped)."""
        scene, _, _ = build_scene(self.cam.read(), self.loc)
        if scene is None:
            raise TaskError("the camera can't see the sheet (need 2+ tags)")
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
        ref = object_name(g["reference"])
        if ref not in objs:
            raise TaskError(f"can't see the reference block {g['reference']!r} (sees {sorted(objs)})")
        ref_xy, ref_yaw = np.array(objs[ref]["center_mm"]), objs[ref]["yaw_deg"]
        if kind == "on_top":
            return ref_xy, ref_yaw, CUBE_SIZE_MM
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

    def run(self, steps):
        report = {"started": time.time(), "steps": []}
        try:
            for i, step in enumerate(steps, 1):
                step = {**step, "object": step["object"]}
                name = object_name(step["object"])
                self.log(f"\nStep {i}/{len(steps)}: {describe(step)}")
                entry = {"step": i, "description": describe(step)}
                report["steps"].append(entry)

                objs = self.observe()
                if name not in objs:
                    raise TaskError(f"can't see {step['object']!r} (sees {sorted(objs)})")
                src = objs[name]
                target, yaw, on = self.resolve(step, objs)
                for other, o in objs.items():  # don't drop a block onto another one by accident
                    if other != name and on == 0.0 and np.hypot(*(np.array(o["center_mm"]) - target)) < CUBE_SIZE_MM:
                        raise TaskError(f"the target is occupied by {other}")
                self.log(f"  {name} at ({src['center_mm'][0]:.0f}, {src['center_mm'][1]:.0f}) -> "
                         f"({target[0]:.0f}, {target[1]:.0f}) mm" + (f" on a {on:.0f} mm block" if on else ""))
                if self.rec:
                    self.rec("plan", cube_xy=src["center_mm"], target_xy=target.tolist())
                t = time.time()
                self.sk.pick(src["center_mm"], src["yaw_deg"])
                self.sk.place(target, yaw, on)
                self.sk.park()
                entry["seconds"] = round(time.time() - t, 1)

                objs = self.observe()
                if name not in objs:
                    raise TaskError(f"after placing, can't see {step['object']!r} (dropped? hidden?)")
                got = np.array(objs[name]["center_mm"])
                err = float(np.hypot(*(got - target)))
                entry.update(landed_mm=got.round(1).tolist(), error_mm=round(err, 1))
                if self.rec:
                    self.rec("verify", target_xy=target.tolist())
                if err > TOLERANCE_MM:
                    raise TaskError(f"{step['object']} landed {err:.0f} mm from its target (tolerance {TOLERANCE_MM:.0f})")
                entry["ok"] = True
                self.log(f"  OK: landed {err:.1f} mm from target in {entry['seconds']} s")
            report["result"] = "done"
        except (TaskError, SkillError) as e:
            report["result"] = f"stopped: {e}"
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
