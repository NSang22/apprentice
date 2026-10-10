"""Exercise each follower joint with small moves around a pose you set by hand.

    python -m robot.scripts.tests.joint_tour

You hold the arm in a pose and press ENTER; torque locks it there. Each step
then moves one joint relative to that pose and returns. At the end you support
the arm before torque turns off.
"""

import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM7"
ROBOT_ID = "my_follower"
SPEED = 20.0          # deg/s (gripper: percent/s)
STEP_DT = 0.05        # seconds between commands
SETTLE_S = 0.4
MAX_GOAL_ERROR = 2.0  # degrees; goal must match present before torque goes on

# (joint, offsets from the held pose, visited in order). Ends back at 0.
TOUR = [
    ("gripper", "absolute", [60.0, None]),  # None = back to held value
    ("wrist_flex", "relative", [20.0, -20.0, 0.0]),
    ("wrist_roll", "relative", [45.0, -45.0, 0.0]),
    ("elbow_flex", "relative", [15.0, 0.0]),
    ("shoulder_pan", "relative", [20.0, -20.0, 0.0]),
    ("shoulder_lift", "relative", [10.0, 0.0]),
]


def read_pos(robot):
    return robot.bus.sync_read("Present_Position")


def move_joint(robot, joint, start, end):
    n = max(1, round(abs(end - start) / (SPEED * STEP_DT)))
    for i in range(1, n + 1):
        robot.send_action({f"{joint}.pos": start + (end - start) * i / n})
        time.sleep(STEP_DT)
    time.sleep(SETTLE_S)
    actual = read_pos(robot)[joint]
    print(f"    target {end:7.1f}   actual {actual:7.1f}   error {actual - end:+5.1f}")


def main():
    robot = SO101Follower(
        SO101FollowerConfig(port=PORT, id=ROBOT_ID, max_relative_target=10.0)
    )
    robot.bus.connect()
    torque_on = False
    try:
        if not robot.bus.is_calibrated:
            raise SystemExit("Motors don't match the calibration file; run lerobot-calibrate.")

        robot.bus.disable_torque()
        input("\nHold the arm in a pose with the gripper well above the table, keep holding, press ENTER...")

        # Copy present position into Goal_Position so enabling torque holds in place.
        held = read_pos(robot)
        robot.bus.sync_write("Goal_Position", held)
        goal = robot.bus.sync_read("Goal_Position")
        worst = max(abs(goal[m] - held[m]) for m in held)
        if worst > MAX_GOAL_ERROR:
            raise SystemExit(f"Goal/present mismatch of {worst:.1f} deg after sync; not enabling torque.")

        robot.configure()  # torque on
        torque_on = True
        print("\nHeld pose:")
        for m, v in held.items():
            print(f"  {m:14s} {v:8.2f}")
        input("\nTorque on. Let go, step back, and press ENTER to start the tour (Ctrl+C to stop)...")

        for joint, mode, waypoints in TOUR:
            print(f"\n{joint}:")
            current = held[joint]
            for wp in waypoints:
                if wp is None:
                    target = held[joint]
                elif mode == "absolute":
                    target = wp
                else:
                    target = held[joint] + wp
                move_joint(robot, joint, current, target)
                current = target

        print("\nTour done; arm is back at the held pose.")
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C. Torque is still holding the current position.")
    finally:
        if torque_on:
            try:
                input("\nSupport the arm with your hand (it goes limp next), then press ENTER...")
            except (EOFError, KeyboardInterrupt):
                pass
        robot.bus.disconnect(disable_torque=True)
        print("Disconnected, torque off.")


if __name__ == "__main__":
    main()
