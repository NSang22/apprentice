"""Self-calibration: the arm moves one cube through all 9 cell centres, the camera measures
where it really lands each time, and an affine correction is fitted and saved.

    python -m robot.scripts.calibration.auto_calibrate             # red cube, all 9 cells (~10 min)
    python -m robot.scripts.calibration.auto_calibrate blue_cube
    python -m robot.scripts.calibration.auto_calibrate --bank   # also 5 spots in the cube bank (~5 min more)

With --bank the arm also sets the cube down at spots in the cube bank beside the sheet
(BANK_ZONE_MM) and measures the error there separately: everything else was calibrated on the
grid, 5-10 cm away, and the grid fit extrapolated 16 mm wrong out there (run 02:33). Nothing else
should be in the bank during the run.

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
from robot.perception.segment import in_bank
from robot.perception.scene import build_scene
from robot.perception.sheet import SheetLocalizer, cell_center
from robot.recorder import RunRecorder
from robot.session import ArmSession
from robot.skills import SkillError, Skills
from robot.workspace import Workspace

ORDER = ["B2", "A1", "A2", "A3", "B3", "C3", "C2", "C1", "B1"]
BANK_ORDER = [(266.0, 50.0), (300.0, 85.0), (266.0, 125.0), (300.0, 165.0), (266.0, 205.0)]  # down the bank, both columns
MOVED_MM = 6.0  # the cube is where we last set it down (nobody touched it): pick at the command, not the reading
SKIP_IF_WITHIN_MM = 15.0  # cube already at this target: go to the next one first


def bank_error(data):
    """Median (observed - commanded) over the bank placements measured so far, or None."""
    pts = [d for d in data if d["cell"].startswith("bank")]
    if not pts:
        return None
    return np.median([np.subtract(d["observed"], d["commanded"]) for d in pts], axis=0)


def find(cam, loc, name, tries=3):
    for _ in range(tries):
        scene, _, _ = build_scene(cam.read(), loc)
        for o in (scene or {}).get("objects", []):
            if o["name"] == name and o["score"] >= 0:  # negative score = merged blob
                return np.array(o["center_mm"]), o["yaw_deg"]
        time.sleep(0.5)
    return None, None


def main():
    bank = "--bank" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cube = args[0] if args else "red_cube"
    targets = [(c, np.array(cell_center(c))) for c in ORDER]
    if bank:
        from robot.perception.config import BANK_ZONE_MM
        if BANK_ZONE_MM is None:
            raise SystemExit("--bank needs BANK_ZONE_MM in robot/perception/config.py")
        targets += [(f"bank{i + 1}", np.array(xy)) for i, xy in enumerate(BANK_ORDER)]
        targets.append(("B2", np.array(cell_center("B2"))))  # end on the grid, where the next task expects a block
    ws, loc, cam, arm = Workspace(), SheetLocalizer(), Camera(), Arm()
    rec = RunRecorder(cam, loc, ws, arm, label=f"autocal_{cube}")
    sk = Skills(arm, ws, log=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"), on_phase=rec)  # no correction: measure raw
    data, fails_in_a_row = [], 0
    last = None  # (where we commanded the cube to, where the camera then saw it)
    try:
        with ArmSession(arm, sk):
            for cell, target in targets:
                for attempt in range(2):  # a failed move gets one more try, from a fresh look
                    where, yaw = find(cam, loc, cube)
                    if where is None:
                        raise SkillError(f"lost sight of {cube}")
                    if np.hypot(*(where - target)) < SKIP_IF_WITHIN_MM:
                        print(f"\n{cell}: cube already there, skipping")
                        break
                    pick_at, note = where, ""
                    if attempt == 0 and last is not None and np.hypot(*(where - last[1])) < MOVED_MM:
                        # Still where the arm set it down: picking at that same command puts the jaws
                        # exactly around it, however far off the arm is there (the bank is uncalibrated).
                        pick_at, yaw, note = last[0], 0.0, " (where the arm set it down)"
                    elif in_bank(where) and bank_error(data) is not None:
                        pick_at, note = where - bank_error(data), " (minus the bank error measured so far)"
                    print(f"\n{cell}: cube at ({where[0]:.0f}, {where[1]:.0f}) -> target ({target[0]:.0f}, {target[1]:.0f}); "
                          f"picking at ({pick_at[0]:.0f}, {pick_at[1]:.0f}){note}")
                    try:
                        sk.pick(pick_at, yaw)
                        sk.place(target, 0.0)
                        sk.park()
                        fails_in_a_row = 0
                    except SkillError as e:
                        print(f"  failed: {e}")
                        rec("error", reason=str(e))
                        sk.park()
                        last = None  # that command is not to be trusted any more
                        again, _ = find(cam, loc, cube)
                        if again is None:
                            raise SkillError(f"lost sight of {cube} after that")
                        print(f"  looked again: the cube is at ({again[0]:.0f}, {again[1]:.0f}), "
                              f"{np.hypot(*(again - where)):.0f} mm from where it was seen before the move")
                        if attempt == 0:
                            print("  retrying this target, picking it up where it is now")
                            continue
                        fails_in_a_row += 1
                        if fails_in_a_row >= 2:
                            raise SkillError("two targets in a row failed twice; stopping")
                        print("  giving up on this target")
                        break
                    got, _ = find(cam, loc, cube)
                    if got is None:
                        print("  can't see the cube after placing; skipping this point")
                        break
                    err = got - target
                    last = (target, got)
                    data.append({"cell": cell, "commanded": target.tolist(), "observed": got.tolist()})
                    print(f"  landed at ({got[0]:.1f}, {got[1]:.1f}), error ({err[0]:+.1f}, {err[1]:+.1f}) = {np.hypot(*err):.1f} mm")
                    break
    except SkillError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        cam.close()
        print(f"Run evidence: {rec.save()}")

    out = ROOT / "outputs" / f"autocal_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"cube": cube, "points": data, "evidence": str(rec.dir)}, indent=2))
    print(f"\n{len(data)} placements measured -> {out}")
    grid = [d for d in data if not d["cell"].startswith("bank")]
    bank_data = [d for d in data if d["cell"].startswith("bank")]
    bank_fit = None
    if bank_data:
        bb, bbefore, bafter, bkeep = correction.fit_bank([d["commanded"] for d in bank_data], [d["observed"] for d in bank_data])
        for d, k in zip(bank_data, bkeep):
            if not k:
                print(f"  ignoring {d['cell']}: {np.hypot(*np.subtract(d['observed'], d['commanded'])):.0f} mm off (failed placement)")
        if bb is None or bkeep.sum() < 3:
            print("Fewer than 3 usable bank placements: the bank correction is NOT saved.")
        else:
            bank_fit = {"b": np.round(bb, 2).tolist(), "rms_before_mm": round(bbefore, 2),
                        "rms_after_mm": round(bafter, 2), "n_placements": int(bkeep.sum())}
            print(f"Bank: the arm lands ({bb[0]:+.1f}, {bb[1]:+.1f}) mm from where it aims; error {bbefore:.1f} mm -> {bafter:.1f} mm after correcting")
    data = grid
    C = [d["commanded"] for d in data]
    O = [d["observed"] for d in data]
    if len(data) < 4:
        print("Need at least 4 measured grid placements to fit a correction; nothing saved.")
        return
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
    correction.save(A, b, before, after, len(Ck), bank=bank_fit)
    print(f"Placement error RMS ({len(Ck)} placements): {before:.1f} mm uncorrected -> {after:.1f} mm after correction (fit)")
    print(f"Leave-one-out predicted error: mean {np.mean(loo):.1f} mm, max {np.max(loo):.1f} mm")
    print(f"Saved {correction.CORRECTION_FILE.name}; pick_place_test.py uses it automatically.")


if __name__ == "__main__":
    main()
