"""Pick-and-place skills in the sheet frame (mm). Uses Arm for motion, Workspace for IK."""

import numpy as np

from .arm import ArmPushed

PARK_XY, PARK_H = (-50.0, 20.0), 120.0  # left of the sheet, out of the camera's way
TRAVEL_H = 120.0       # tip height for moving between cells (cube is 48 mm tall)
GRASP_H = 14.0         # tip height when closing on a cube resting on the table
CUBE_MM = 48.0
GRIPPER_OPEN = 50.0    # percent (~63 deg). 70% swung the jaw out nearly horizontal (run 03:54)
BACKOFF_MM = 6.0       # after letting go, slide the fixed jaw this far off the cube face before lifting;
                       # lifting straight up dragged the foam cube along by friction (autocal 04:59, A1)
# The SO-101 has one fixed jaw and one moving jaw. The jaw-tip point (closed) is the fixed
# jaw's inner face, so the cube centre sits half a cube toward the moving jaw from it.
FIXED_JAW_CLEARANCE = 10.0  # mm; 4 was less than the corner calibration error (A3 run 04:33 landed on the cube)
MIN_GRASP = 8.0        # gripper reading below this at contact = closed on nothing
WAYPOINT_MM = 20.0
MAX_JOINT_JUMP = 25.0  # deg between waypoints; more means IK flipped arm shape -> abort
PRECISE_BELOW_MM = 60.0  # near the table the gripper must be near-vertical; higher up, tilt is fine
TILT_LOW, TILT_HIGH = 25.0, 50.0


def max_tilt(h):
    return TILT_LOW if h < PRECISE_BELOW_MM else TILT_HIGH
ROLL_SPEED = 30.0
BLOCKED_MM = 10.0      # mid-descent: tips this far above the waypoint = something is underneath
# End of descent (after sag fix). Tuned on runs 04:01-04:33: good grasps stopped 0-1 mm high and
# <=5.5 mm sideways; the failed A3 grasp stopped 3.9 mm high (jaw resting on the cube).
BOTTOM_HIGH_MM = 3.0
BOTTOM_SIDE_MM = 8.0


class SkillError(RuntimeError):
    pass


def _wrap90(a):
    return (a + 45.0) % 90.0 - 45.0


# The jaws open and close along gripper_frame_link's y axis, and the moving jaw sits on
# its -y side. Verified from the overhead photos of run 2026-10-10 03:54: the open jaw
# pointed along sheet -x while gripper_frame y mapped to sheet +x. (Using the x axis, as
# the first two attempts did, put the gap 90 deg away from the cube.)
JAW_AXIS_COL = 1
MOVING_JAW_SIGN = -1


def jaw_axis_angle(ws, q_reading):
    """Direction the jaws close along, as an angle in the sheet frame (atan2 on sheet x, y)."""
    T = ws.kin.fk(ws.to_urdf(q_reading))
    t = np.radians(ws.theta)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    s = (T[:2, JAW_AXIS_COL] @ R) * [1, -1]
    return float(np.degrees(np.arctan2(s[1], s[0])))


def moving_jaw_dir(ws, q_reading):
    """Unit vector (sheet frame) from the fixed jaw toward the moving jaw."""
    a = np.radians(jaw_axis_angle(ws, q_reading) + (0.0 if MOVING_JAW_SIGN > 0 else 180.0))
    return np.array([np.cos(a), np.sin(a)])


def tip_xy_for_cube(ws, cube_xy, h, roll, gap_mm):
    """Jaw-tip point that puts the cube centre `gap_mm` toward the moving jaw from it."""
    cube_xy = np.asarray(cube_xy, float)
    tip = cube_xy
    for _ in range(2):  # the direction barely changes with the small offset; 2 passes is plenty
        n = moving_jaw_dir(ws, ws.solve(tip, h, wrist_roll_reading=roll))
        tip = cube_xy - n * gap_mm
    return tip


def grasp_pose(ws, cube_xy, h, cube_yaw_deg, gap_mm, roll0=0.0):
    """Roll and jaw-tip point together: the roll depends on where the arm reaches, and the
    tip offset depends on the roll, so alternate until both settle."""
    tip, roll = np.asarray(cube_xy, float), roll0
    for _ in range(3):
        roll = roll_for_yaw(ws, tip, h, cube_yaw_deg, roll)
        tip = tip_xy_for_cube(ws, cube_xy, h, roll, gap_mm)
    return tip, roll


def roll_for_yaw(ws, xy, h, cube_yaw_deg, roll0=0.0):
    """Wrist roll that lines the jaws up with a pair of cube faces.

    cube_yaw_deg is counterclockwise-from-above (scene convention), so face normals sit at
    -yaw (mod 90) as sheet atan2 angles. Turning the roll by +d turns the jaw axis by about -d.
    """
    roll = roll0
    for _ in range(3):
        q = ws.solve(xy, h, wrist_roll_reading=roll)
        roll -= _wrap90(-cube_yaw_deg - jaw_axis_angle(ws, q))
    return roll


TRAVEL_ERR_MM = 5.0  # IK slack allowed on high travel waypoints; precision only matters low down


def _arc_points(ws, a, b, n):
    """Points from a to b swinging around the arm's base: distance from the pan axis,
    pan angle and height all change linearly. Avoids cutting close past the base."""
    c = ws.base_sheet_xy()
    da, db = a[:2] - c, b[:2] - c
    ra, rb = np.hypot(*da), np.hypot(*db)
    pa, pb = np.arctan2(da[1], da[0]), np.arctan2(db[1], db[0])
    dp = (pb - pa + np.pi) % (2 * np.pi) - np.pi
    out = []
    for i in range(1, n + 1):
        f = i / n
        r, ang = ra + (rb - ra) * f, pa + dp * f
        out.append(np.array([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), a[2] + (b[2] - a[2]) * f]))
    return out


def line_path(ws, a, b, roll_a, roll_b, seed):
    """IK waypoints from a=(x, y, h) to b, sheet frame. Vertical-ish moves go in a straight
    line; moves across the table at travel height swing around the base."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / WAYPOINT_MM)))
    across = np.linalg.norm(b[:2] - a[:2]) > 1.0 and min(a[2], b[2]) >= PRECISE_BELOW_MM
    points = _arc_points(ws, a, b, n) if across else [a + (b - a) * (i / n) for i in range(1, n + 1)]
    path, prev = [], seed
    for i, p in enumerate(points, 1):
        f = i / n
        try:
            q = ws.solve(p[:2], p[2], wrist_roll_reading=roll_a + (roll_b - roll_a) * f, seed_reading=prev,
                         max_tilt_deg=max_tilt(p[2]),
                         max_err_mm=TRAVEL_ERR_MM if p[2] >= PRECISE_BELOW_MM else 1.0)
        except ValueError as e:
            raise SkillError(f"no safe path: {e}") from None
        jump = max(abs(q[j] - prev[j]) for j in q if j in prev)
        if prev is not seed and jump > MAX_JOINT_JUMP:
            raise SkillError(f"IK jumped {jump:.0f} deg near {p.round(1).tolist()}; refusing to move")
        path.append(q)
        prev = q
    return path


class Skills:
    def __init__(self, arm, ws, log=print, on_phase=None, correct=None):
        """correct: maps a camera-frame sheet point to the point to command (see correction.py)."""
        self.arm, self.ws, self.log = arm, ws, log
        self.correct = correct or (lambda xy: np.asarray(xy, float))
        self.on_phase = on_phase or (lambda phase, **info: None)
        self.pos = None  # current tip target (x, y, h) in the sheet frame, None = unknown
        self.roll = 0.0

    def _follow(self, path, gripper, speed=None):
        for q in path[:-1]:
            self.arm.move_to({**q, "gripper": gripper}, speed_deg_s=speed, fix_sag=False)
        self.arm.move_to({**path[-1], "gripper": gripper}, speed_deg_s=speed)

    def go(self, xy, h, roll=None, gripper=None, speed=None):
        """Straight-line move of the jaw tips to (xy, h)."""
        roll = self.roll if roll is None else roll
        gripper = self.arm.goal["gripper"] if gripper is None else gripper
        here = self.arm.read()
        if self.pos is None:  # first move from an unknown pose: one joint-space move to the target
            try:
                q = self.ws.solve(xy, h, wrist_roll_reading=roll, seed_reading=here, max_tilt_deg=max_tilt(h),
                                  max_err_mm=TRAVEL_ERR_MM if h >= PRECISE_BELOW_MM else 1.0)
            except ValueError as e:
                raise SkillError(f"unreachable: {e}") from None
            self.arm.move_to({**q, "gripper": gripper}, speed_deg_s=speed)
        else:
            going_down = h < self.pos[2] - 1.0
            path = line_path(self.ws, self.pos, (*xy, h), self.roll, roll, here)
            if going_down:
                self._descend(path, gripper, speed, h)
            else:
                self._follow(path, gripper, speed)
        self.pos, self.roll = (float(xy[0]), float(xy[1]), float(h)), roll

    def _descend(self, path, gripper, speed, h):
        """Follow a downward path, checking after each waypoint that nothing blocks it."""
        for i, q in enumerate(path):
            last = i == len(path) - 1
            self.arm.move_to({**q, "gripper": gripper}, speed_deg_s=speed, fix_sag=last)
            want_xy, want_h = self.ws.tip_sheet(q)
            got_xy, got_h = self.ws.tip_sheet(self.arm.read())
            limit = BOTTOM_HIGH_MM if last else BLOCKED_MM
            off_side = float(np.hypot(*(got_xy - want_xy)))
            if got_h - want_h > limit or (last and off_side > BOTTOM_SIDE_MM):
                back = path[max(0, i - 2)]
                self.arm.move_to({**back, "gripper": gripper}, speed_deg_s=speed, fix_sag=False)
                self.pos = None  # we backed off mid-path; next move re-plans from here
                raise SkillError(f"blocked: tips {got_h - want_h:.0f} mm high, {off_side:.0f} mm sideways "
                                 f"while descending to {h:.0f} mm; backed off")

    def park(self):
        self.log("park")
        if self.pos is not None and self.pos[2] < TRAVEL_H:
            self.go(self.pos[:2], TRAVEL_H)
        self.go(PARK_XY, PARK_H, roll=0.0)
        self.on_phase("parked")

    def pick(self, xy, yaw_deg):
        self.log(f"pick at ({xy[0]:.1f}, {xy[1]:.1f}) yaw {yaw_deg:.1f}")
        xy = self.correct(xy)
        if self.pos is not None and self.pos[2] < TRAVEL_H:
            self.go(self.pos[:2], TRAVEL_H)
        try:
            tip, roll = grasp_pose(self.ws, xy, GRASP_H, yaw_deg, CUBE_MM / 2 + FIXED_JAW_CLEARANCE, self.roll)
        except ValueError as e:
            raise SkillError(f"can't grasp there: {e}") from None
        self.log(f"  jaw tips go to ({tip[0]:.1f}, {tip[1]:.1f}), roll {roll:.1f}")
        self.go(tip, TRAVEL_H, roll=roll, gripper=GRIPPER_OPEN)
        self.on_phase("above_grasp", cube_xy=list(xy), tip_target=tip.tolist())
        self.go(tip, GRASP_H, speed=15)
        self.on_phase("at_grasp_open", cube_xy=list(xy), tip_target=tip.tolist())
        try:
            g = self.arm.close_gripper_on_contact()
        except ArmPushed as e:
            self.on_phase("pushed", reason=str(e))
            self.go(tip, TRAVEL_H, gripper=GRIPPER_OPEN, speed=15)
            raise SkillError(f"grip aborted: {e} (jaw probably on top of the cube)") from None
        self.log(f"  gripper contact at {g:.1f}%")
        self.on_phase("closed", cube_xy=list(xy), contact_pct=g)
        if g < MIN_GRASP:
            self.go(tip, TRAVEL_H, gripper=GRIPPER_OPEN)
            raise SkillError("grasp failed: jaws closed on nothing")
        self.go(tip, TRAVEL_H, speed=15)
        self.on_phase("lifted")
        if self.arm.read()["gripper"] < MIN_GRASP:
            raise SkillError("cube slipped out while lifting")

    def place(self, xy, yaw_deg, on_height_mm=0.0):
        """Put the held cube down at xy. on_height_mm > 0 stacks it on something that tall."""
        self.log(f"place at ({xy[0]:.1f}, {xy[1]:.1f}) yaw {yaw_deg:.1f} on {on_height_mm:.0f} mm")
        xy = self.correct(xy)
        h = GRASP_H + on_height_mm + 2.0  # release ~2 mm above the surface
        try:
            tip, roll = grasp_pose(self.ws, xy, h, yaw_deg, CUBE_MM / 2, self.roll)  # cube rests against the fixed jaw
        except ValueError as e:
            raise SkillError(f"can't place there: {e}") from None
        self.go(tip, TRAVEL_H, roll=roll)
        self.on_phase("above_place", target_xy=list(xy), tip_target=tip.tolist())
        self.go(tip, h, speed=15)
        self.on_phase("at_place", target_xy=list(xy))
        self.arm.move_to({"gripper": GRIPPER_OPEN})  # full 50%: 30% didn't let go of the foam cube
        self.on_phase("released", target_xy=list(xy))
        away = tip - moving_jaw_dir(self.ws, self.arm.read()) * BACKOFF_MM  # fixed jaw moves off the cube face
        self.go(away, h, speed=10)
        self.go(away, TRAVEL_H, speed=15)
        self.on_phase("retreated", target_xy=list(xy))
