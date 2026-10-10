"""Re-record only the gripper's range of motion for my_follower.

Keeps the other joints' calibration and the gripper's homing offset as-is.
Torque stays off the whole time; nothing moves under power.
"""

import os

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = os.environ.get("ROBOT_PORT", "COM7")
ROBOT_ID = "my_follower"
MIN_SPAN = 300  # ticks; anything smaller means the jaw wasn't really moved

robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
if not robot.calibration:
    raise SystemExit(f"No calibration file for {ROBOT_ID}; run lerobot-calibrate first.")

robot.bus.connect()
try:
    robot.bus.disable_torque()
    old = robot.calibration["gripper"]
    print(f"Current gripper range: {old.range_min} -> {old.range_max}")
    print("Open the gripper fully and close it fully a couple of times, then press ENTER.")
    mins, maxes = robot.bus.record_ranges_of_motion(["gripper"])

    span = maxes["gripper"] - mins["gripper"]
    if span < MIN_SPAN:
        raise SystemExit(f"Gripper only moved {span} ticks; not saving. Check the jaw and its cable.")

    old.range_min, old.range_max = mins["gripper"], maxes["gripper"]
    robot.bus.write_calibration(robot.calibration)
    robot._save_calibration()
    print(f"New gripper range: {old.range_min} -> {old.range_max} ({span} ticks)")
    print("Saved to", robot.calibration_fpath)
finally:
    robot.bus.disconnect(disable_torque=True)
