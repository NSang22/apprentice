"""Sheet-frame targets -> follower joint commands, using arm_sheet_calibration.json."""

import json

import numpy as np

from .arm_calibration import CALIB_FILE, OFFSET_JOINTS, sheet_to_base_xy
from .kinematics import ARM_JOINTS, SO101Kinematics

# A comfortable reach-down pose (follower readings, deg) used to seed IK.
SEED_READING = {"shoulder_pan": -20.0, "shoulder_lift": 30.0, "elbow_flex": 0.0, "wrist_flex": 85.0, "wrist_roll": 0.0}
MAX_POS_ERR_MM = 1.0
MAX_TILT_DEG = 25.0  # jaws can still close on a cube's opposite faces at this tilt
DOWN_WEIGHT = 5.0  # small: hit the position exactly, get as vertical as the joints allow
LIMIT_MARGIN_DEG = 3.0  # stay this far inside the calibrated hard stops
ROBOT_ID = "my_follower"


def calibrated_ranges(robot_id=ROBOT_ID):
    """Each joint's reading range (deg) from the LeRobot follower calibration file."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    path = HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / f"{robot_id}.json"
    cal = json.loads(path.read_text())
    return {j: (cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / 4095 for j in ARM_JOINTS}


class Workspace:
    def __init__(self, calib_path=CALIB_FILE):
        c = json.loads(open(calib_path).read())
        self.offsets = {j: c["joint_offsets_deg"].get(j, 0.0) for j in ARM_JOINTS}
        s = c["sheet_to_base"]
        self.theta, self.tx, self.ty = s["theta_deg"], s["tx_mm"], s["ty_mm"]
        self.table_z = c["table_z_mm"]
        self.tip_mm = c["tip_extension_mm"]
        self.kin = SO101Kinematics()
        # The URDF's limits are nominal; use this arm's calibrated ranges (in URDF angles).
        for j, half in calibrated_ranges().items():
            if j == "wrist_roll":
                continue  # fixed during IK; keep the URDF's cable-safe range
            lo, hi = -half + LIMIT_MARGIN_DEG, half - LIMIT_MARGIN_DEG
            self.kin.limits[j] = (lo + self.offsets[j], hi + self.offsets[j])

    # Follower readings and URDF angles differ by the fitted zero offsets.
    def to_urdf(self, reading):
        return [reading[j] + self.offsets[j] for j in ARM_JOINTS]

    def to_reading(self, q_urdf):
        return {j: q - self.offsets[j] for j, q in zip(ARM_JOINTS, q_urdf)}

    def sheet_to_base(self, sheet_xy, height_mm=0.0):
        xy = sheet_to_base_xy([sheet_xy], self.theta, self.tx, self.ty)[0]
        return np.array([xy[0], xy[1], self.table_z + height_mm])

    def base_sheet_xy(self):
        """The arm's shoulder_pan axis, in sheet mm (the URDF puts it 38.8 mm in front of base_link)."""
        t = np.radians(self.theta)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pan_xy = self.kin.chain[0]["origin"][:2, 3]
        return ((pan_xy - [self.tx, self.ty]) @ R) * [1, -1]

    def tip_sheet(self, reading):
        """Where the jaw tips are, in sheet mm, and their height above the table."""
        T = self.kin.fk(self.to_urdf(reading))
        p = T[:3, 3] + T[:3, 2] * self.tip_mm
        t = np.radians(self.theta)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        sheet = ((p[:2] - [self.tx, self.ty]) @ R) * [1, -1]
        return sheet, p[2] - self.table_z

    def arm_points_sheet(self, reading):
        """The arm's joint centres plus the jaw tips, base to tip, as (sheet xy, height above
        the table): a stick figure of the arm, for checking what it hides from the camera."""
        T, _, origins = self.kin.fk_frames(self.to_urdf(reading))
        pts = [origins[j] for j in origins] + [T[:3, 3] + T[:3, 2] * self.tip_mm]
        t = np.radians(self.theta)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return [(((p[:2] - [self.tx, self.ty]) @ R) * [1, -1], p[2] - self.table_z) for p in pts]

    def solve(self, sheet_xy, height_mm, wrist_roll_reading=0.0, seed_reading=None, max_tilt_deg=MAX_TILT_DEG,
              max_err_mm=MAX_POS_ERR_MM):
        """Joint command (follower readings, deg) to put the jaw tips at (sheet_xy, height),
        gripper pointing down. Raises ValueError if unreachable."""
        seed = self.to_urdf(seed_reading or SEED_READING)
        roll_urdf = wrist_roll_reading + self.offsets["wrist_roll"]
        q, err, tilt = self.kin.ik_down(self.sheet_to_base(sheet_xy, height_mm), seed, roll_urdf, self.tip_mm,
                                       down_weight=DOWN_WEIGHT)
        self.last_tilt_deg = tilt
        if err > max_err_mm or tilt > max_tilt_deg:
            raise ValueError(f"unreachable: {sheet_xy} at {height_mm} mm (err {err:.1f} mm, tilt {tilt:.1f} deg)")
        return self.to_reading(q)
