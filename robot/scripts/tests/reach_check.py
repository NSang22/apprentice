"""Diagnostic: stop above a cube (no contact) and photograph where the gripper really is.

    python -m robot.scripts.tests.reach_check red_cube

Parks, finds the cube, moves the CLOSED gripper to 15 mm above the cube's top (aligned
for grasping), saves robot/outputs/reach_closed.png with a crosshair where the model thinks the
tips are, opens the gripper, saves robot/outputs/reach_open.png, then closes and parks.
"""

import sys

import cv2

from robot.perception.camera import Camera
from robot.perception.config import ROOT
from robot.perception.scene import build_scene
from robot.perception.sheet import SheetLocalizer
from robot.arm import Arm
from robot.skills import CUBE_MM, GRIPPER_OPEN, Skills, SkillError, roll_for_yaw
from robot.workspace import Workspace

ABOVE_TOP_MM = 15.0


def snap(cam, loc, name, marks):
    """Save a frame with crosshairs at sheet points: marks = [(label, xy, height_mm, bgr)]."""
    frame = cam.read()
    pose = loc.locate(frame)
    if pose is not None:
        for label, xy, h, col in marks:
            u, v = pose.to_image([xy], h)[0].astype(int)
            cv2.drawMarker(frame, (u, v), col, cv2.MARKER_CROSS, 40, 3)
            cv2.putText(frame, label, (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    path = ROOT / "outputs" / name
    cv2.imwrite(str(path), frame)
    return path


def main():
    cube = sys.argv[1] if len(sys.argv) > 1 else "red_cube"
    ws, loc, cam, arm = Workspace(), SheetLocalizer(), Camera(), Arm()
    arm.connect()
    sk = Skills(arm, ws)
    try:
        input("\nHold the arm up above the sheet (well clear of the table), keep holding, press ENTER...")
        arm.enable()
        input("Torque on. Let go, step back, hand near the 12V plug. ENTER to park...")
        sk.park()
        scene, _, _ = build_scene(cam.read(), loc)
        objs = {o["name"]: o for o in (scene or {}).get("objects", [])}
        if cube not in objs:
            raise SkillError(f"camera doesn't see {cube}")
        xy, yaw = objs[cube]["center_mm"], objs[cube]["yaw_deg"]
        h = CUBE_MM + ABOVE_TOP_MM
        roll = roll_for_yaw(ws, xy, h, yaw)
        print(f"{cube} at ({xy[0]:.1f}, {xy[1]:.1f}) mm yaw {yaw:.1f}; going to {h:.0f} mm above it, roll {roll:.1f}")
        sk.go(xy, 120.0, roll=roll)
        sk.go(xy, h, speed=15)
        tip_xy, tip_h = ws.tip_sheet(arm.read())
        print(f"model says tips at ({tip_xy[0]:.1f}, {tip_xy[1]:.1f}) mm, {tip_h:.1f} mm up; joints: "
              + "  ".join(f"{k} {v:.1f}" for k, v in arm.read().items()))
        marks = [("cube", xy, CUBE_MM, (0, 255, 0)), ("tips (model)", tip_xy, tip_h, (0, 0, 255))]
        print("saved", snap(cam, loc, "reach_closed.png", marks))
        input("LOOK from above and the side: where are the closed tips relative to the cube centre? ENTER to open the gripper...")
        arm.move_to({"gripper": GRIPPER_OPEN})
        print("saved", snap(cam, loc, "reach_open.png", marks))
        input("LOOK: which way did the moving jaw swing, relative to the cube? ENTER to close and park...")
        arm.move_to({"gripper": 0.0})
        sk.park()
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


if __name__ == "__main__":
    main()
