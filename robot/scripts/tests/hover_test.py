"""First IK test: hover above grid crossings and lower to just above them.

    python -m robot.scripts.tests.hover_test           # real run (arm moves)
    python -m robot.scripts.tests.hover_test --plan    # print the planned joint angles only, no connection

You hold the arm up above the sheet, it locks there, then for each crossing it goes
to 120 mm above it, lowers to 10 mm, and you check by eye whether the closed jaw tips
sit right above the crossing. At the end it returns to where you held it and asks you
to support the arm before torque turns off.
"""

import json
import sys

from robot.perception.config import SHEET_SPEC
from robot.arm import Arm
from robot.workspace import Workspace

CROSSINGS = ["r2c2", "r1c1"]
HOVER_MM = 120.0
LOW_MM = 10.0
DESCENT_STEP_MM = 15.0  # IK waypoints on the way down keep the descent near-vertical


def crossing_mm(name):
    g = json.loads(open(SHEET_SPEC).read())["grid"]
    r, c = int(name[1]), int(name[3])
    return (g["origin_mm"][0] + c * g["cell_mm"], g["origin_mm"][1] + r * g["cell_mm"])


def vertical_path(ws, xy, z_from, z_to, roll):
    """IK waypoints from z_from to z_to above xy; each solve seeds from the previous one."""
    n = max(1, int(abs(z_from - z_to) // DESCENT_STEP_MM))
    path, seed = [], None
    for i in range(n + 1):
        z = z_from + (z_to - z_from) * i / n
        q = ws.solve(xy, z, wrist_roll_reading=roll, seed_reading=seed)
        path.append(q)
        seed = q
    return path


def main():
    plan_only = "--plan" in sys.argv
    ws = Workspace()
    roll = 0.0
    plans = {}
    for name in CROSSINGS:
        xy = crossing_mm(name)
        plans[name] = (xy, vertical_path(ws, xy, HOVER_MM, LOW_MM, roll))
        low = plans[name][1][-1]
        print(f"{name} at sheet ({xy[0]:.1f}, {xy[1]:.1f}) mm -> low pose "
              + "  ".join(f"{j} {v:6.1f}" for j, v in low.items()))
    if plan_only:
        return

    arm = Arm()
    arm.connect()
    try:
        input("\nHold the arm up above the sheet (gripper CLOSED, well clear of the table), keep holding, press ENTER...")
        start = arm.enable()
        input("Torque on. Let go, step back, hand near the 12V plug. ENTER to start...")
        gripper = {"gripper": start["gripper"]}
        for name in CROSSINGS:
            xy, path = plans[name]
            print(f"\n-> above {name} at {HOVER_MM:.0f} mm")
            arm.move_to({**path[0], **gripper})
            input("   ENTER to lower to 10 mm...")
            for q in path[1:]:
                arm.move_to({**q, **gripper}, speed_deg_s=15, fix_sag=False)
            arm.move_to({**path[-1], **gripper}, speed_deg_s=15)  # settle + sag fix at the bottom
            tip_xy, tip_h = ws.tip_sheet(arm.read())
            print(f"   target ({xy[0]:.1f}, {xy[1]:.1f}) mm at {LOW_MM:.0f} mm | model says tips at "
                  f"({tip_xy[0]:.1f}, {tip_xy[1]:.1f}) mm, {tip_h:.1f} mm up")
            input("   LOOK: are the jaw tips right above the crossing? Note the offset, then ENTER to go back up...")
            for q in reversed(path[:-1]):
                arm.move_to({**q, **gripper}, speed_deg_s=15, fix_sag=False)
        print("\n-> back to where you held it")
        arm.move_to(start)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Arm is holding its current position.")
    finally:
        if arm.torque_on:
            try:
                input("\nSupport the arm with your hand (it goes limp next), then press ENTER...")
            except (EOFError, KeyboardInterrupt):
                pass
        arm.disconnect()
        print("Disconnected, torque off.")


if __name__ == "__main__":
    main()
