"""Safe joint-space motion for the SO-101 follower. All arm motion goes through here.

- Goal_Position is synced to the present position before torque turns on (no jump).
- Moves are linear joint ramps at a capped speed.
- After a move, a short outer loop corrects gravity sag on the shoulder, elbow and wrist.
"""

import os
import time

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = os.environ.get("ROBOT_PORT", "COM7")  # COM7 on Windows; a /dev/tty* path on Jetson (jetson-kushi/env.example)
ROBOT_ID = "my_follower"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SAG_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
DT = 0.05
SETTLE_S = 0.3
MAX_SAG_FIX_DEG = 6.0
PUSHED_DEG = 3.0  # an arm joint this far off its command while gripping = the jaw is shoving the arm


class ArmPushed(RuntimeError):
    pass


class Arm:
    def __init__(self, port=PORT, robot_id=ROBOT_ID, speed_deg_s=25.0):
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=10.0))
        self.speed = speed_deg_s
        self.torque_on = False
        self.goal = None   # where we want the joints
        self.sent = None   # last command actually sent (goal + sag correction)

    # -- connection -------------------------------------------------------
    def connect(self):
        self.robot.bus.connect()
        if not self.robot.bus.is_calibrated:
            self.robot.bus.disconnect(disable_torque=True)
            raise RuntimeError("Motors don't match the calibration file; run lerobot-calibrate.")
        self.robot.bus.disable_torque()

    def enable(self):
        """Hold the arm where it is right now."""
        present = self.robot.bus.sync_read("Present_Position")
        self.robot.bus.sync_write("Goal_Position", present)
        goal = self.robot.bus.sync_read("Goal_Position")
        worst = max(abs(goal[m] - present[m]) for m in present)
        if worst > 2.0:
            raise RuntimeError(f"Goal/present mismatch {worst:.1f} deg after sync; not enabling torque")
        self.robot.configure()  # torque on
        self.torque_on = True
        self.goal = dict(present)
        self.sent = dict(present)
        return dict(present)

    def disconnect(self):
        self.robot.bus.disconnect(disable_torque=True)
        self.torque_on = False

    # -- state ------------------------------------------------------------
    def read(self):
        return self.robot.bus.sync_read("Present_Position")

    # -- motion -----------------------------------------------------------
    def move_to(self, target, speed_deg_s=None, fix_sag=True):
        """Ramp all given joints (deg; gripper 0-100) to `target`. Returns final error per joint."""
        assert self.torque_on, "call enable() first"
        start = dict(self.sent)
        end = {**self.goal, **target}
        span = max(abs(end[j] - start[j]) for j in JOINTS)
        n = max(1, int(np.ceil(span / ((speed_deg_s or self.speed) * DT))))
        for i in range(1, n + 1):
            a = i / n
            self._send({j: start[j] + (end[j] - start[j]) * a for j in JOINTS})
            time.sleep(DT)
        time.sleep(SETTLE_S)
        if fix_sag:
            self._fix_sag(end)
        self.goal = end
        present = self.read()
        return {j: present[j] - end[j] for j in JOINTS}

    def _fix_sag(self, end, rounds=2):
        """Gravity pulls joints short of their goals; push the goal past by the error."""
        cmd = dict(end)
        for _ in range(rounds):
            present = self.read()
            for j in SAG_JOINTS:
                err = end[j] - present[j]
                cmd[j] = end[j] + np.clip(cmd[j] - end[j] + err, -MAX_SAG_FIX_DEG, MAX_SAG_FIX_DEG)
            self._send(cmd)
            time.sleep(SETTLE_S)

    def close_gripper_on_contact(self, rate_pct_s=40.0, lag_pct=6.0, squeeze_pct=4.0):
        """Close until the jaws stall on something, then squeeze a little past contact.

        Returns the gripper reading at contact (0 = fully closed, i.e. nothing grasped).
        LeRobot caps the gripper motor at 50% torque, so a stall can't hurt the motor.
        """
        cmd = dict(self.sent)
        g = g_open = cmd["gripper"]
        hold = {j: self.read()[j] for j in JOINTS if j != "gripper"}  # pose (incl. sag) before closing
        stalled = 0
        while g > 0:
            g = max(0.0, g - rate_pct_s * DT)
            cmd["gripper"] = g
            self._send(cmd)
            time.sleep(DT)
            now = self.read()
            pushed = max(abs(now[j] - hold[j]) for j in hold)
            if pushed > PUSHED_DEG:
                cmd["gripper"] = g_open
                self._send(cmd)
                time.sleep(SETTLE_S)
                raise ArmPushed(f"arm pushed {pushed:.1f} deg while gripping")
            present = now["gripper"]
            stalled = stalled + 1 if present - g > lag_pct else 0
            if stalled >= 3:
                cmd["gripper"] = max(0.0, present - squeeze_pct)
                self._send(cmd)
                self.goal["gripper"] = cmd["gripper"]
                time.sleep(SETTLE_S)
                return present
        self.goal["gripper"] = 0.0
        time.sleep(SETTLE_S)
        return self.read()["gripper"]

    def _send(self, q):
        self.sent = dict(q)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in q.items()})
