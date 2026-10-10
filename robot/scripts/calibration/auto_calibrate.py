"""Self-calibration: the arm moves one cube through all 9 cell centres, the camera measures
where it really lands each time, and an affine correction is fitted and saved.

    python -m robot.scripts.calibration.auto_calibrate             # red cube, all 9 cells (~10 min)
    python -m robot.scripts.calibration.auto_calibrate blue_cube

Put ONE cube of that colour anywhere on the sheet. Re-run whenever the arm, sheet or camera
moves. Writes robot/calibration/placement_correction.json (used by every pick/place) and
robot/outputs/autocal_<time>.json.
"""

import json
import sys
import time

import numpy as np

from robot import correction
from robot.arm import Arm
from robot.perception.camera import Camera
from robot.perception.config import ROOT
from robot.perception.scene import build_scene
from robot.perception.sheet import SheetLocalizer, cell_center
from robot.recorder import RunRecorder
from robot.skills import SkillError, Skills
from robot.workspace import Workspace

ORDER = ["B2", "A1", "A2", "A3", "B3", "C3", "C2", "C1", "B1"]
SKIP_IF_WITHIN_MM = 15.0  # cube already at this target: go to the next one first


def find(cam, loc, name, tries=3):
    for _ in range(tries):
        scene, _, _ = build_scene(cam.read(), loc)
        for o in (scene or {}).get("objects", []):
            if o["name"] == name and o["score"] >= 0:  # negative score = merged blob
                return np.array(o["center_mm"]), o["yaw_deg"]
        time.sleep(0.5)
    return None, None


def main():
    cube = sys.argv[1] if len(sys.argv) > 1 else "red_cube"
    ws, loc, cam, arm = Workspace(), SheetLocalizer(), Camera(), Arm()
    arm.connect()
    rec = RunRecorder(cam, loc, ws, arm, label=f"autocal_{cube}")
    sk = Skills(arm, ws, log=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"), on_phase=rec)  # no correction: measure raw
    data, fails_in_a_row = [], 0
    try:
        input("\nHold the arm up above the sheet (well clear of the table), keep holding, press ENTER...")
        arm.enable()
        input("Torque on. Let go, step back, hand near the 12V plug. ENTER to start (about 10 min)...")
        sk.park()
        for cell in ORDER:
            target = np.array(cell_center(cell))
            where, yaw = find(cam, loc, cube)
            if where is None:
                raise SkillError(f"lost sight of {cube}")
            if np.hypot(*(where - target)) < SKIP_IF_WITHIN_MM:
                print(f"\n{cell}: cube already there, skipping")
                continue
            print(f"\n{cell}: cube at ({where[0]:.0f}, {where[1]:.0f}) -> target ({target[0]:.0f}, {target[1]:.0f})")
            try:
                sk.pick(where, yaw)
                sk.place(target, 0.0)
                sk.park()
                fails_in_a_row = 0
            except SkillError as e:
                print(f"  failed: {e}")
                rec("error", reason=str(e))
                fails_in_a_row += 1
                if fails_in_a_row >= 2:
                    raise SkillError("two failures in a row; stopping")
                sk.park()
                continue
            got, _ = find(cam, loc, cube)
            if got is None:
                print("  can't see the cube after placing; skipping this point")
                continue
            err = got - target
            data.append({"cell": cell, "commanded": target.tolist(), "observed": got.tolist()})
            print(f"  landed at ({got[0]:.1f}, {got[1]:.1f}), error ({err[0]:+.1f}, {err[1]:+.1f}) = {np.hypot(*err):.1f} mm")
    except SkillError as e:
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Arm is holding its current position.")
    finally:
        cam.close()
        if arm.torque_on:
            try:
                input("\nSupport the arm with your hand (it goes limp next), then press ENTER...")
            except (EOFError, KeyboardInterrupt):
                pass
        arm.disconnect()
        print("Disconnected, torque off.")

    out = ROOT / "outputs" / f"autocal_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"cube": cube, "points": data, "evidence": str(rec.dir)}, indent=2))
    print(f"\n{len(data)} placements measured -> {out}")
    if len(data) < 4:
        print("Need at least 4 measured placements to fit a correction; nothing saved.")
        return
    C = [d["commanded"] for d in data]
    O = [d["observed"] for d in data]
    A, b, before, after, keep, problem = correction.fit_robust(C, O)
    for d, k in zip(data, keep):
        if not k:
            err = np.hypot(*np.subtract(d["observed"], d["commanded"]))
            print(f"  ignoring {d['cell']}: {err:.0f} mm off (failed placement, or the cube was moved)")
    if problem:
        print(f"NOT saving a correction: {problem}. The previous correction (if any) is unchanged.")
        return
    Ck = [c for c, k in zip(C, keep) if k]
    Ok = [o for o, k in zip(O, keep) if k]
    loo = []
    for i in range(len(Ck)):  # leave-one-out: predict where an unseen placement lands
        Ai, bi, _, _ = correction.fit(Ck[:i] + Ck[i + 1:], Ok[:i] + Ok[i + 1:])
        loo.append(float(np.hypot(*(Ai @ np.array(Ck[i]) + bi - np.array(Ok[i])))))
    correction.save(A, b, before, after, len(Ck))
    print(f"Placement error RMS ({len(Ck)} placements): {before:.1f} mm uncorrected -> {after:.1f} mm after correction (fit)")
    print(f"Leave-one-out predicted error: mean {np.mean(loo):.1f} mm, max {np.max(loo):.1f} mm")
    print(f"Saved {correction.CORRECTION_FILE.name}; pick_place_test.py uses it automatically.")


if __name__ == "__main__":
    main()
