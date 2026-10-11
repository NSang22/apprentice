"""Touch calibration: record joint angles with the closed jaw tips on known grid crossings,
then fit the arm <-> sheet calibration. Torque stays OFF: you move the arm by hand.

    python -m robot.scripts.calibration.touch_calibrate          # QUICK, after moving the setup: 4 corners (~2 min).
                                       #   Keeps the arm's own joint offsets; refits only where
                                       #   the sheet sits relative to the arm.
    python -m robot.scripts.calibration.touch_calibrate --full   # FULL, once: all 16 crossings, row by row, left to right.
                                       #   Also fits the arm's joint offsets and jaw-tip length.
    python -m robot.scripts.calibration.touch_calibrate --middle # MIDDLE, after a quick one: the 4 corners of B2.
                                       #   Fits everything (like full) on these plus the last quick
                                       #   run's 4 outer corners: 8 points from near to far reach.

Grid lines are numbered 0-3: vertical lines left->right (c0..c3), horizontal lines
top->bottom (r0..r3, r0 = the grid's top edge, nearest the arm). Point "r1c2" is where
horizontal line 1 crosses vertical line 2.
"""

import json
import shutil
import sys
import time

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from robot.perception.config import ROOT, SHEET_SPEC
from robot import arm_calibration as ac
from robot.kinematics import SO101Kinematics

PORT = "COM7"
ROBOT_ID = "my_follower"
FULL_POINTS = [f"r{r}c{c}" for r in range(4) for c in range(4)]  # row by row, left to right
QUICK_POINTS = ["r0c0", "r0c3", "r3c0", "r3c3"]
MIDDLE_POINTS = ["r1c1", "r1c2", "r2c2", "r2c1"]  # the corners of B2, around the cell
MIDDLE_FREE = list(range(8))  # joint offsets + placement + table height; the tip length stays
MAX_SAVE_RMS_MM = 5.0  # a worse fit means a bad touch (or the setup moved since the quick run)

MAP = """
        arm side
   c0     c1     c2     c3
r0 +------+------+------+
   |  A1  |  A2  |  A3  |
r1 +------+------+------+
   |  B1  |  B2  |  B3  |
r2 +------+------+------+
   |  C1  |  C2  |  C3  |
r3 +------+------+------+
"""


def point_mm(name, grid):
    r, c = int(name[1]), int(name[3])
    ox, oy = grid["origin_mm"]
    return [ox + c * grid["cell_mm"], oy + r * grid["cell_mm"]]


def record(points):
    grid = json.loads(open(SHEET_SPEC).read())["grid"]
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.bus.connect()
    records = []
    try:
        if not robot.bus.is_calibrated:
            raise SystemExit("Motors don't match the calibration file; run lerobot-calibrate.")
        robot.bus.disable_torque()
        print(MAP)
        print("Close the gripper fully. For each point: put the jaw tips on the line crossing,")
        print("hold still, press ENTER. 's' + ENTER skips a point; 'u' + ENTER redoes the previous one.\n")
        i = 0
        while i < len(points):
            name = points[i]
            ans = input(f"[{i + 1}/{len(points)}] Touch {name}, hold, ENTER: ").strip().lower()
            if ans == "u" and records:
                print(f"   removed {records.pop()['point']}; redo it")
                i = points.index(records[-1]["point"]) + 1 if records else 0
                continue
            if ans == "s":
                print("   skipped")
                i += 1
                continue
            q = robot.bus.sync_read("Present_Position")
            records.append({"point": name, "sheet_mm": point_mm(name, grid),
                            "joints_deg": {k: round(float(v), 2) for k, v in q.items()}})
            print("   " + "  ".join(f"{k} {v:7.1f}" for k, v in q.items()))
            i += 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped early.")
    finally:
        robot.bus.disconnect(disable_torque=True)
    return records


def leave_one_out(records, kin, params, free):
    loo = []
    print(f"Checking: refitting {len(records)} times, leaving out one point each time (takes ~1 min)...")
    for i in range(len(records)):  # leave-one-out: predict a crossing the fit never saw
        print(f"  {i + 1}/{len(records)}", end="\r", flush=True)
        rest = records[:i] + records[i + 1:]
        pi, _ = ac.fit(rest, kin, params, free=free)
        loo.append(float(np.hypot(*ac.residuals(pi, kin, [records[i]])[:2])))
    print(f"Leave-one-out xy error: mean {np.mean(loo):.1f} mm, max {np.max(loo):.1f} mm "
          f"({', '.join(f'{r['point']} {e:.1f}' for r, e in zip(records, loo))})")


def last_quick_points():
    files = sorted(f for f in (ROOT / "outputs").glob("touch_points_quick_2*.json") if not f.stem.endswith("_fixed"))
    if not files:
        raise SystemExit("No quick (4 corner) touch points saved yet: run the quick calibration first.")
    return files[-1], json.loads(files[-1].read_text())["points"]


def main():
    full, middle = "--full" in sys.argv, "--middle" in sys.argv
    if not full and not ac.CALIB_FILE.exists():
        raise SystemExit("No arm calibration yet: run the full one first (python -m robot.scripts.calibration.touch_calibrate --full).")
    kin = SO101Kinematics()
    if middle:
        quick_file, outer = last_quick_points()
        print(f"Using the outer corners from {quick_file.name} (the setup must not have moved since).")
    mode = "full" if full else "quick+middle" if middle else "quick"
    records = record(FULL_POINTS if full else MIDDLE_POINTS if middle else QUICK_POINTS)

    out = ROOT / "outputs" / f"touch_points_{mode.replace('+', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"recorded_at": time.time(), "mode": mode, "points": records}, indent=2))
    print(f"\nSaved {len(records)} points to {out.name}")

    if middle:
        if not records:
            print("No points recorded; calibration NOT changed.")
            return
        old = ac.load_params()
        print("\nHow far off the current calibration was at these points (xy):")
        for r, e in zip(records, ac.residuals(old, kin, records).reshape(-1, 3)):
            print(f"  {r['point']}  {np.hypot(*e[:2]):4.1f} mm  (x {e[0]:+.1f}, y {e[1]:+.1f} in the arm frame)")
        records = outer + records

    need = 8 if full or middle else 3
    if len(records) < need:
        print(f"Need at least {need} points to fit; calibration NOT changed.")
        return

    if full:
        params, rms = ac.fit(records, kin, ac.initial_guess(records, kin))
        free = None
    elif middle:
        # Not the jaw-tip length: with the gripper vertical at every touch it trades off exactly
        # against the table height (run 18:58 fitted tip -40 mm, table +36 mm).
        free = MIDDLE_FREE
        params, rms = ac.fit(records, kin, ac.load_params(), free=free)
    else:
        params, rms = ac.fit(records, kin, ac.load_params(), free=ac.PLACEMENT_ONLY)
        free = ac.PLACEMENT_ONLY
    per_point = ac.residuals(params, kin, records).reshape(-1, 3)
    print(f"\nFit RMS {rms:.1f} mm. Per point (xy error, height error):")
    for r, e in zip(records, per_point):
        print(f"  {r['point']}  {np.hypot(*e[:2]):4.1f} mm   {e[2]:+5.1f} mm")
    leave_one_out(records, kin, params, free)
    if middle and rms > MAX_SAVE_RMS_MM:
        print(f"Fit RMS {rms:.1f} mm is too high to trust (limit {MAX_SAVE_RMS_MM:.0f}); calibration NOT changed.")
        print("Look for the point with the big error above and redo the run, or redo the quick run first.")
        return

    if ac.CALIB_FILE.exists():
        backup = ROOT / "outputs" / f"arm_sheet_calibration_before_{time.strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy(ac.CALIB_FILE, backup)
        print(f"Previous calibration backed up to robot/outputs/{backup.name}")
    ac.save(params, rms, len(records), mode=mode)
    print(f"Saved {ac.CALIB_FILE.name}. If placements were also auto-calibrated, re-run auto_calibrate.py.")
    print("Done.")


if __name__ == "__main__":
    main()
