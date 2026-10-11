"""First motion test for the SO-101 follower.

Connects, prints joint positions, rotates wrist_roll by +10 degrees, waits,
returns to the start position, and disconnects (torque off).

    python -m robot.scripts.tests.first_move            # real run (arm moves)
    python -m robot.scripts.tests.first_move --hold     # you pose the arm by hand first; it holds there
    python -m robot.scripts.tests.first_move --dry-run  # connect + checks only, torque stays off
"""

import os
import sys
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = os.environ.get("ROBOT_PORT", "COM7")
ROBOT_ID = "my_follower"
JOINT = "wrist_roll.pos"
NUDGE_DEG = 10.0
STEP_DEG = 0.5      # per command while ramping
STEP_DT = 0.05      # seconds between commands -> 10 deg/s
HOLD_S = 1.0
MAX_GOAL_ERROR = 2.0  # degrees; goal must match present before torque goes on


def print_positions(robot, label):
    obs = robot.get_observation()
    print(f"\n{label}")
    for key, val in obs.items():
        print(f"  {key:18s} {val:8.2f}")
    return obs


def ramp(robot, start, end):
    n = max(1, round(abs(end - start) / STEP_DEG))
    for i in range(1, n + 1):
        robot.send_action({JOINT: start + (end - start) * i / n})
        time.sleep(STEP_DT)


def main():
    dry_run = "--dry-run" in sys.argv
    hold = "--hold" in sys.argv and not dry_run
    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, max_relative_target=10.0)
    )

    robot.bus.connect()
    torque_on = False
    try:
        if not robot.bus.is_calibrated:
            raise SystemExit("Motors don't match the calibration file; run lerobot-calibrate.")

        robot.bus.disable_torque()
        if hold:
            input("\nHold the arm in the pose you want, keep holding, and press ENTER...")

        # The motors keep a stale Goal_Position while torque is off. Copy the present
        # position into it first, so enabling torque holds the arm where it is.
        present = robot.bus.sync_read("Present_Position")
        robot.bus.sync_write("Goal_Position", present)
        goal = robot.bus.sync_read("Goal_Position")
        worst = max(abs(goal[m] - present[m]) for m in present)
        if worst > MAX_GOAL_ERROR:
            raise SystemExit(f"Goal/present mismatch of {worst:.1f} deg after sync; not enabling torque.")
        print(f"Goal synced to present (max error {worst:.2f} deg).")

        start = print_positions(robot, "Start positions:")
        if dry_run:
            print("\nDry run: torque left off, nothing moved.")
            return

        robot.configure()  # enables torque; arm should stiffen in place
        torque_on = True
        time.sleep(0.5)
        if hold:
            input("Torque on. Let go of the arm, step back, and press ENTER to start the wrist move...")

        origin = start[JOINT]
        print(f"\nRotating {JOINT} {origin:.1f} -> {origin + NUDGE_DEG:.1f}")
        ramp(robot, origin, origin + NUDGE_DEG)
        time.sleep(HOLD_S)
        print_positions(robot, "After nudge:")

        print(f"\nReturning {JOINT} to {origin:.1f}")
        ramp(robot, origin + NUDGE_DEG, origin)
        time.sleep(HOLD_S)
        print_positions(robot, "Back at start:")
    finally:
        if hold and torque_on:
            try:
                input("\nSupport the arm with your hand (it goes limp next), then press ENTER...")
            except (EOFError, KeyboardInterrupt):
                pass
        robot.bus.disconnect(disable_torque=True)
        print("\nDisconnected, torque off.")


if __name__ == "__main__":
    main()
