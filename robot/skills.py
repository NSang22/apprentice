"""Pick-and-place skills in the sheet frame (mm). Uses Arm for motion, Workspace for IK."""

import numpy as np

from .arm import ArmPushed, ArmStopped

PARK_XY, PARK_H = (-50.0, 20.0), 120.0  # left of the sheet, out of the camera's way
REST_XY, REST_H = PARK_XY, 8.0            # end of a run: straight down from the parking spot, on the
                                          # table left of the sheet (keep it clear). The old spot between
                                          # tags 0 and 1 was too close to the base once the arm moved nearer.
TRAVEL_SPEED = 60.0    # deg/s (largest joint) for moves between cells at travel height
NEAR_SPEED = 25.0      # deg/s for descents, lifts and anything near the table or a cube
GRIPPER_SPEED = 150.0  # percent/s for opening the jaw
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
BLOCKED_MM = 15.0      # mid-descent: tips this far above the command = something is underneath
                       # (streamed motion lags its command by a few mm, so this is looser than 10)
# End of descent (after sag fix). Tuned on runs 04:01-04:33: good grasps stopped 0-1 mm high and
# <=5.5 mm sideways; the failed A3 grasp stopped 3.9 mm high (jaw resting on the cube).
BOTTOM_HIGH_MM = 3.0
TRAVEL_CLEAR_MM = 60.0  # travel this far above the tallest stack (a tilted gripper dips; the arm sags)
MAX_TRAVEL_H = 190.0   # reachable over the whole sheet and both parking sides (A3 tops out near 190)
STACK_CLEAR_MM = 40.0  # above a stack, the open jaws stay this far over its top before moving or
                       # closing (run 19:59: closing right after a stack release knocked the tower over)
RELEASE_GAP_MM = 2.0   # release this far above the surface on the table...
RELEASE_GAP_PER_LEVEL_MM = 2.0  # ...plus this per block underneath: sizes and grip vary, and add up
LANDED_MM = 10.0       # placing: the held cube may stop this far above plan (it touched the surface)
DROP_XY = (-50.0, 85.0)  # last resort at rest: set a still-held block down here, beside the sheet
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


# Where the gripper physically goes around a block, as capsules along the jaw axis n (sheet
# frame) from the jaw-tip point: the fixed jaw sits just behind the tip, and the open moving jaw
# swings out ~82 mm on the n side at 50% (it reached nearly flat at 70%, run 03:54).
FIXED_JAW_BEHIND_MM = 12.0 + BACKOFF_MM  # fixed jaw thickness plus the slide-away after release
MOVING_JAW_REACH_MM = 85.0
JAW_HALF_WIDTH_MM = 14.0
NEIGHBOUR_RADIUS_MM = CUBE_MM / 2 + 3.0  # a neighbouring block, as a disc around its centre
SIDE_CLEARANCE_MM = 6.0                  # prefer a grip side with at least this much room
MAX_PLANS_TRIED = 6                      # grip pairings whose travel paths plan_move checks
WRIST_ROLL_RANGE = (-150.0, 155.0)       # stay inside the wrist's calibrated range


def _dist_point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.hypot(*(p - (a + t * ab))))


def side_clearance(tip, n, neighbours):
    """Smallest gap (mm) between the gripper's jaws (fixed + swung-open moving jaw) and any
    neighbouring block; positive = room to spare. neighbours: list of (x, y) centres."""
    tip, n = np.asarray(tip, float), np.asarray(n, float)
    a, b = tip - n * FIXED_JAW_BEHIND_MM, tip + n * MOVING_JAW_REACH_MM
    gaps = [_dist_point_segment(np.asarray(c, float), a, b) - NEIGHBOUR_RADIUS_MM - JAW_HALF_WIDTH_MM for c in neighbours]
    return min(gaps) if gaps else float("inf")


def grip_options(ws, cube_xy, h, cube_yaw_deg, gap_mm, roll0, neighbours):
    """Every reachable way to grip a cube square to its faces, one per wrist quarter turn.
    Each is a dict: tip, roll, jaw (moving-jaw direction, sheet frame), clear_mm (room to the
    nearest neighbour, inf if none) and turns (quarter turns from the default grip)."""
    tip, roll = grasp_pose(ws, cube_xy, h, cube_yaw_deg, gap_mm, roll0)
    options = []
    for k in (0, 1, -1, 2, -2):
        r = roll + 90.0 * k
        if not (WRIST_ROLL_RANGE[0] <= r <= WRIST_ROLL_RANGE[1]):
            continue
        try:
            t = tip if k == 0 else tip_xy_for_cube(ws, cube_xy, h, r, gap_mm)
            n = moving_jaw_dir(ws, ws.solve(t, h, wrist_roll_reading=r))
        except ValueError:
            continue
        options.append({"tip": t, "roll": r, "jaw": n, "turns": abs(k),
                        "clear_mm": side_clearance(t, n, neighbours) if neighbours else float("inf")})
    if not options:
        raise SkillError("no reachable grip side")
    return options


def _no_room(where, options):
    best = max(o["clear_mm"] for o in options)
    return SkillError(f"every side of the {where} is blocked by a neighbouring block (best gap {best:.0f} mm); "
                      "move the blocks further apart")


def clear_grasp_pose(ws, cube_xy, h, cube_yaw_deg, gap_mm, roll0, neighbours):
    """Like grasp_pose, but chooses which of the 4 cube faces to grip so the jaws stay clear of
    neighbouring blocks. Returns (tip, roll, clearance_mm); raises SkillError if no side fits."""
    options = grip_options(ws, cube_xy, h, cube_yaw_deg, gap_mm, roll0, neighbours)
    fits = [o for o in options if o["clear_mm"] >= 0]
    if not fits:
        raise _no_room("block", options)
    best = min(fits, key=lambda o: (o["clear_mm"] < SIDE_CLEARANCE_MM, o["turns"], -o["clear_mm"]))
    return best["tip"], best["roll"], best["clear_mm"]


# Placement is accurate with the wrist near the roll it was calibrated at (0): 8-12 mm at
# |roll| < 25, but 19 mm at roll 87 (B2 runs 18:34 and 18:47, with and without turning the
# cube in the hand, landed on the same spot). So grips are ranked by how far the wrist rolls.


def grip_cost(o):
    return abs(o["roll"])


LARGE_ROLL_DEG = 30.0  # beyond this the release lands ~20 mm off (runs 18:34, 18:47, 19:33)


def rank_grips(pick_options, place_options):
    """All workable (pick, place) grip pairs, best first. Any pair clear of neighbours is safe
    (the clearance model has margins built in), so accuracy comes first: a small wrist roll at
    the place, then room to spare, then a small roll at the pick. Run 19:33 ranked room first,
    took a 87 deg release to gain 1 mm of room, and missed by 23 mm. Raises SkillError if none fit."""
    pairs = []
    for a in pick_options:
        for b in place_options:
            if min(a["clear_mm"], b["clear_mm"]) < 0:
                continue
            roomy = min(a["clear_mm"], b["clear_mm"]) >= SIDE_CLEARANCE_MM
            pairs.append(((grip_cost(b) > LARGE_ROLL_DEG, grip_cost(a) > LARGE_ROLL_DEG, not roomy,
                           grip_cost(b), grip_cost(a)), a, b))
    if not pairs:
        if all(o["clear_mm"] < 0 for o in pick_options):
            raise _no_room("block", pick_options)
        raise _no_room("target", place_options)
    return [(a, b) for _, a, b in sorted(pairs, key=lambda p: p[0])]


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



def travel_path(ws, a, b, roll_a, roll_b, seed):
    """line_path, but if turning the wrist during the move makes the IK jump (run 19:07: roll
    0 -> 139 while swinging past the base), turn the wrist in place first and then travel, or
    travel first and turn at the end. Raises the original SkillError if none of these work."""
    try:
        return line_path(ws, a, b, roll_a, roll_b, seed)
    except SkillError as first:
        if abs(roll_b - roll_a) < 1.0:
            raise
        for turn_at, travel_roll in ((a, roll_b), (b, roll_a)):
            try:
                if turn_at is a:
                    turn = line_path(ws, a, a, roll_a, roll_b, seed)
                    return turn + line_path(ws, a, b, roll_b, roll_b, turn[-1])
                move = line_path(ws, a, b, roll_a, roll_a, seed)
                return move + line_path(ws, b, b, roll_a, roll_b, move[-1])
            except SkillError:
                continue
        raise first


# Parking between steps: anywhere just off the sheet's left or right edge, nearest first, as long
# as the arm then hides none of the grid from the camera (that's all parking is for).
SHEET_W_MM = 215.9
SIDE_PARK_OFFSET_MM = 35.0
SIDE_PARK_YS = np.arange(20.0, 241.0, 20.0)
VIEW_MARGIN_MM = 8.0
GRIPPER_PARKED = 10.0  # nearly closed: the open moving jaw sticks out ~85 mm


def hides_area(ws, q, camera_xyz, box, margin=VIEW_MARGIN_MM):
    """True if, seen from the camera, any part of the arm (joint centres to jaw tips, sampled
    along each link) lies over box = (x0, y0, x1, y1) sheet mm, at table or cube-top level."""
    c = np.asarray(camera_xyz, float)
    x0, y0, x1, y1 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
    pts = ws.arm_points_sheet(q)
    for (a, ha), (b, hb) in zip(pts, pts[1:]):
        for f in np.linspace(0.0, 1.0, 6):
            p, h = a + (b - a) * f, ha + (hb - ha) * f
            for plane in (0.0, CUBE_MM):
                if not plane < h < c[2]:
                    continue
                sx, sy = c[:2] + (p - c[:2]) * (c[2] - plane) / (c[2] - h)
                if x0 < sx < x1 and y0 < sy < y1:
                    return True
    return False


def side_park_spots(ws, here_xy, camera_xyz, box, h=PARK_H):
    """Parking spots off the sheet's side edges, nearest to here_xy first, that the arm can
    reach and that keep it out of the camera's view of box. Each is (xy, joint reading)."""
    spots = [np.array([x, y]) for x in (-SIDE_PARK_OFFSET_MM, SHEET_W_MM + SIDE_PARK_OFFSET_MM) for y in SIDE_PARK_YS]
    spots.sort(key=lambda xy: float(np.hypot(*(xy - here_xy))))
    out = []
    for xy in spots:
        try:
            q = ws.solve(xy, h, wrist_roll_reading=0.0, max_tilt_deg=max_tilt(h), max_err_mm=TRAVEL_ERR_MM)
        except ValueError:
            continue
        if not hides_area(ws, {**q, "gripper": GRIPPER_PARKED}, camera_xyz, box):
            out.append((xy, q))
    return out


class Skills:
    def __init__(self, arm, ws, log=print, on_phase=None, correct=None):
        """correct: maps a camera-frame sheet point to the point to command (see correction.py)."""
        self.arm, self.ws, self.log = arm, ws, log
        self.correct = correct or (lambda xy: np.asarray(xy, float))
        self.on_phase = on_phase or (lambda phase, **info: None)
        self.pos = None  # current tip target (x, y, h) in the sheet frame, None = unknown
        self.roll = 0.0
        self.holding = None  # (xy, yaw, base_mm) the held block was picked from, None = empty
        self.tallest_mm = CUBE_MM  # top of the tallest stack on the table (the executor keeps it current)

    def go(self, xy, h, roll=None, gripper=None, speed=None, land_mm=0.0, sag_fix=True):
        """Smooth move of the jaw tips to (xy, h): straight down/up, or an arc around the base
        when crossing the table. Streams as one motion."""
        roll = self.roll if roll is None else roll
        gripper = self.arm.goal["gripper"] if gripper is None else gripper
        here = self.arm.read()
        if self.pos is None:  # first move from an unknown pose: one joint-space move to the target
            try:
                q = self.ws.solve(xy, h, wrist_roll_reading=roll, seed_reading=here, max_tilt_deg=max_tilt(h),
                                  max_err_mm=TRAVEL_ERR_MM if h >= PRECISE_BELOW_MM else 1.0)
            except ValueError as e:
                raise SkillError(f"unreachable: {e}") from None
            self.arm.follow([{**q, "gripper": gripper}], speed_deg_s=speed or NEAR_SPEED)
        elif h < self.pos[2] - 1.0:
            self._descend(xy, h, roll, gripper, speed, here, land_mm, sag_fix)
        else:
            path = travel_path(self.ws, self.pos, (*xy, h), self.roll, roll, here)
            self.arm.follow([{**q, "gripper": gripper} for q in path], speed_deg_s=speed or TRAVEL_SPEED)
        self.pos, self.roll = (float(xy[0]), float(xy[1]), float(h)), roll

    def _descend(self, xy, h, roll, gripper, speed, here, land_mm=0.0, sag_fix=True):
        """Stream down, watching for anything underneath; strict check at the bottom.
        land_mm: when placing, stopping up to this much high is fine (the held cube has landed)."""
        path = line_path(self.ws, self.pos, (*xy, h), self.roll, roll, here)
        wps = [{**q, "gripper": gripper} for q in path]

        def blocked(cmd, present):
            _, want_h = self.ws.tip_sheet(cmd)
            _, got_h = self.ws.tip_sheet(present)
            if got_h - want_h > BLOCKED_MM:
                return f"blocked: tips {got_h - want_h:.0f} mm above the command while descending to {h:.0f} mm"
            return None

        try:
            self.arm.follow(wps, speed_deg_s=speed or NEAR_SPEED, check=blocked, fix_sag=sag_fix)
        except ArmStopped as e:
            self.arm.follow([wps[0]], speed_deg_s=NEAR_SPEED)  # back to where the descent started
            raise SkillError(f"{e}; backed off") from None
        want_xy, want_h = self.ws.tip_sheet(path[-1])
        got_xy, got_h = self.ws.tip_sheet(self.arm.read())
        off_side = float(np.hypot(*(got_xy - want_xy)))
        if got_h - want_h > max(BOTTOM_HIGH_MM, land_mm) or off_side > BOTTOM_SIDE_MM:
            self.arm.follow([wps[0]], speed_deg_s=NEAR_SPEED)
            raise SkillError(f"blocked: tips {got_h - want_h:.0f} mm high, {off_side:.0f} mm sideways "
                             f"at the bottom of the descent to {h:.0f} mm; backed off")

    def set_gripper(self, pct):
        self.arm.follow([{"gripper": pct}], speed_deg_s=GRIPPER_SPEED)

    def wake(self):
        """Start from wherever the arm is (torque just turned on): rise straight up, then park."""
        here = self.arm.read()
        xy, h = self.ws.tip_sheet(here)
        self.roll = 0.0
        self.log(f"wake: tips at ({xy[0]:.0f}, {xy[1]:.0f}) mm, {h:.0f} mm up")
        if h < TRAVEL_H - 5:
            try:
                path = line_path(self.ws, (*xy, max(h, 2.0)), (*xy, TRAVEL_H), here["wrist_roll"], 0.0, here)
                self.arm.follow([{**q, "gripper": here["gripper"]} for q in path], speed_deg_s=NEAR_SPEED)
                self.pos = (float(xy[0]), float(xy[1]), TRAVEL_H)
            except SkillError:
                self.pos = None  # odd starting pose: a slow joint-space move to park instead
        self.park()

    def put_back(self):
        """After a failed step with a block still in the jaws: set it down where it was picked
        up (that spot was free a moment ago)."""
        if self.holding is None:
            return
        xy, yaw, base = self.holding
        self.log(f"putting the held block back at ({xy[0]:.0f}, {xy[1]:.0f})")
        self.place(xy, yaw, base)

    def rest(self):
        """End of a run: set the arm down on its rest spot so torque can be released safely."""
        self.log("rest")
        if self.pos is not None and self.pos[2] < self.travel_h():
            self.go(self.pos[:2], self.travel_h())
        if self.holding is not None:  # never lower the arm onto the table with a block in the jaws
            self.log(f"  still holding a block: setting it down beside the sheet at ({DROP_XY[0]:.0f}, {DROP_XY[1]:.0f})")
            self.place(DROP_XY, 0.0)
            self.go(DROP_XY, self.travel_h(), speed=NEAR_SPEED)
        self.go(REST_XY, self.travel_h(), roll=0.0)
        self.set_gripper(0.0)  # off the sheet: closing can't catch a block here
        self.go(REST_XY, REST_H, speed=NEAR_SPEED)

    def park(self, camera_xyz=None, keep_clear=None):
        """Get out of the camera's way. With camera_xyz and keep_clear (the grid's box in sheet
        mm), park at the nearest spot off the sheet's side that keeps the grid in view;
        otherwise (or if none works) at the fixed parking spot."""
        if self.pos is not None and self.pos[2] < self.travel_h():
            self.go(self.pos[:2], self.travel_h())
        if camera_xyz is not None and keep_clear is not None and self.pos is not None:
            for xy, _ in side_park_spots(self.ws, np.array(self.pos[:2]), camera_xyz, keep_clear, self.travel_h())[:3]:
                try:
                    self.go(xy, self.travel_h(), roll=0.0)
                except SkillError:
                    continue
                self.set_gripper(GRIPPER_PARKED)  # off the sheet now: nothing for the jaw to catch
                self.log(f"park beside the sheet at ({xy[0]:.0f}, {xy[1]:.0f})")
                self.on_phase("parked")
                return
        self.log("park")
        self.go(PARK_XY, self.travel_h(), roll=0.0)
        self.on_phase("parked")

    def _pick_options(self, xy, yaw_deg, neighbours, base_mm=0.0):
        """base_mm: what the cube stands on (48 for a cube on top of another); blocks no taller
        than that are below the jaws and no obstacle. neighbours: (x, y) or (x, y, height)."""
        near = [self.correct(n[:2]) for n in (neighbours or []) if (n[2] if len(n) > 2 else CUBE_MM) > base_mm + 1.0]
        return grip_options(self.ws, self.correct(xy), GRASP_H + base_mm, yaw_deg,
                            CUBE_MM / 2 + FIXED_JAW_CLEARANCE, self.roll, near)

    def _place_options(self, xy, yaw_deg, on_height_mm, neighbours):
        """xy already corrected. Stacking: the jaws work above the block below, so blocks no
        taller than that are no obstacle."""
        near = [self.correct(n[:2]) for n in (neighbours or []) if (n[2] if len(n) > 2 else CUBE_MM) > on_height_mm + 1.0]
        return grip_options(self.ws, xy, self._release_h(on_height_mm), yaw_deg,
                            CUBE_MM / 2, self.roll, near)  # the cube rests against the fixed jaw

    def travel_h(self):
        """Tip height for moving across the table: the usual 120 mm, raised to clear the tallest
        stack by TRAVEL_CLEAR_MM (run 20:41: swinging back at 120 mm over a 2-block tower, 96 mm
        tall, knocked it over)."""
        return min(MAX_TRAVEL_H, max(TRAVEL_H, self.tallest_mm + TRAVEL_CLEAR_MM))

    def _clear_h(self, top_mm):
        """Tip height for moving or working the gripper above something top_mm tall."""
        return max(self.travel_h(), top_mm + STACK_CLEAR_MM) if top_mm > CUBE_MM else self.travel_h()

    @staticmethod
    def _release_h(on_height_mm):
        """Tip height for letting go: the held block's bottom just above the surface (2, 4, 6 mm
        for the table, one block, two blocks). Run 20:52 released 2 mm over a 2-block tower, the
        block touched 3.6 mm early, and the arm pressed on the tower until it let go."""
        return GRASP_H + on_height_mm + RELEASE_GAP_MM + RELEASE_GAP_PER_LEVEL_MM * round(on_height_mm / CUBE_MM)

    def plan_move(self, src_xy, src_yaw, target_xy, target_yaw, on_height_mm=0.0,
                  pick_neighbours=None, place_neighbours=None, src_base_mm=0.0):
        """Grips for a pick-then-place, chosen together: clear of neighbours, wrist roll small,
        and with travel paths the IK can follow (checked here, so a bad pairing falls back to the
        next one instead of stopping the task). Pass the results to pick(grip=) and place(grip=)."""
        try:
            pairs = rank_grips(self._pick_options(src_xy, src_yaw, pick_neighbours, src_base_mm),
                               self._place_options(self.correct(target_xy), target_yaw, on_height_mm,
                                                   place_neighbours))
        except ValueError as e:
            raise SkillError(f"can't plan the grips: {e}") from None
        first_error = None
        for a, b in pairs[:MAX_PLANS_TRIED]:
            try:
                self._check_travel(a, b)
            except SkillError as e:
                first_error = first_error or e
                continue
            if first_error:
                self.log(f"  (first choice of grip had no clean path: {first_error}; using roll {a['roll']:.0f}/{b['roll']:.0f})")
            return a, b
        raise first_error

    def _check_travel(self, a, b):
        """Raise SkillError if the travel moves (here -> above the pick -> above the place) need
        an IK jump, the way go() would."""
        start = self.pos if self.pos is not None else (*PARK_XY, PARK_H)
        start = (start[0], start[1], max(start[2], self.travel_h()))
        seed = self.arm.read() if self.arm is not None else self.ws.solve(start[:2], start[2], wrist_roll_reading=self.roll, max_tilt_deg=max_tilt(start[2]), max_err_mm=TRAVEL_ERR_MM)
        leg1 = travel_path(self.ws, start, (*a["tip"], self.travel_h()), self.roll, a["roll"], seed)
        travel_path(self.ws, (*a["tip"], self.travel_h()), (*b["tip"], self.travel_h()), a["roll"], b["roll"], leg1[-1])

    def pick(self, xy, yaw_deg, neighbours=None, grip=None, base_mm=0.0):
        """neighbours: (x, y) centres of other blocks on the table; the jaws grip from a side
        that keeps clear of them. grip: one from plan_move, which overrides that choice.
        base_mm: what the cube stands on (48 to take it off another cube)."""
        self.log(f"pick at ({xy[0]:.1f}, {xy[1]:.1f}) yaw {yaw_deg:.1f}" + (f" from {base_mm:.0f} mm up" if base_mm else ""))
        src_xy = xy
        grasp_h = GRASP_H + base_mm
        xy = self.correct(xy)
        if self.pos is not None and self.pos[2] < self.travel_h():
            self.go(self.pos[:2], self.travel_h())
        if grip is None:
            try:
                tip, roll, room = clear_grasp_pose(self.ws, xy, grasp_h, yaw_deg, CUBE_MM / 2 + FIXED_JAW_CLEARANCE,
                                                   self.roll, [self.correct(c) for c in (neighbours or [])])
            except ValueError as e:
                raise SkillError(f"can't grasp there: {e}") from None
        else:
            tip, roll, room = grip["tip"], grip["roll"], grip["clear_mm"]
        self.log(f"  jaw tips go to ({tip[0]:.1f}, {tip[1]:.1f}), roll {roll:.1f}"
                 + (f", {room:.0f} mm clear of neighbours" if room != float("inf") else ""))
        above = self._clear_h(base_mm + CUBE_MM)
        self.go(tip, above, roll=roll)  # jaws stay compact on the way; open only above the block
        self.set_gripper(GRIPPER_OPEN)
        self.on_phase("above_grasp", cube_xy=list(xy), tip_target=tip.tolist())
        self.go(tip, grasp_h)
        self.on_phase("at_grasp_open", cube_xy=list(xy), tip_target=tip.tolist())
        try:
            g = self.arm.close_gripper_on_contact()
        except ArmPushed as e:
            self.on_phase("pushed", reason=str(e))
            self.go(tip, self.travel_h(), gripper=GRIPPER_OPEN, speed=NEAR_SPEED)
            raise SkillError(f"grip aborted: {e} (jaw probably on top of the cube)") from None
        self.log(f"  gripper contact at {g:.1f}%")
        self.on_phase("closed", cube_xy=list(xy), contact_pct=g)
        if g < MIN_GRASP:
            self.go(tip, self.travel_h(), gripper=GRIPPER_OPEN, speed=NEAR_SPEED)
            raise SkillError("grasp failed: jaws closed on nothing")
        self.go(tip, self.travel_h(), speed=NEAR_SPEED)
        self.on_phase("lifted")
        if self.arm.read()["gripper"] < MIN_GRASP:
            raise SkillError("cube slipped out while lifting")
        self.holding = (np.asarray(src_xy, float), yaw_deg, base_mm)

    def place(self, xy, yaw_deg, on_height_mm=0.0, neighbours=None, grip=None):
        """Put the held cube down at xy. on_height_mm > 0 stacks it on something that tall.
        neighbours: (x, y, height_mm) of other blocks; only ones that reach the jaws count.
        grip: one from plan_move, which overrides that choice."""
        self.log(f"place at ({xy[0]:.1f}, {xy[1]:.1f}) yaw {yaw_deg:.1f} on {on_height_mm:.0f} mm")
        xy = self.correct(xy)
        h = self._release_h(on_height_mm)
        if grip is None:
            try:
                options = self._place_options(xy, yaw_deg, on_height_mm, neighbours)
            except ValueError as e:
                raise SkillError(f"can't place there: {e}") from None
            fits = [o for o in options if o["clear_mm"] >= 0]
            if not fits:
                raise _no_room("target", options)
            grip = min(fits, key=lambda o: (grip_cost(o) > LARGE_ROLL_DEG, o["clear_mm"] < SIDE_CLEARANCE_MM,
                                            grip_cost(o)))  # see rank_grips
        tip, roll, room = grip["tip"], grip["roll"], grip["clear_mm"]
        if room != float("inf"):
            self.log(f"  releasing with the jaws {room:.0f} mm clear of neighbours")
        self.go(tip, self._clear_h(on_height_mm + CUBE_MM), roll=roll)  # carried block clears a stack top
        self.on_phase("above_place", target_xy=list(xy), tip_target=tip.tolist())
        # No sag fix here: with the block resting on something it reads as sag, and the fix pushes
        # the arm harder into the stack (every release in run 20:52 sprang down 2.7-5.5 mm).
        self.go(tip, h, land_mm=LANDED_MM, sag_fix=False)
        self.arm.hold()  # stop straining toward the command: rest where the block actually stopped
        self.on_phase("at_place", target_xy=list(xy))
        self.set_gripper(GRIPPER_OPEN)  # full 50%: 30% didn't let go of the foam cube
        self.holding = None
        self.on_phase("released", target_xy=list(xy))
        away = tip - moving_jaw_dir(self.ws, self.arm.read()) * BACKOFF_MM  # fixed jaw moves off the cube face
        self.go(away, h, speed=NEAR_SPEED)
        self.go(away, self._clear_h(on_height_mm + CUBE_MM), speed=NEAR_SPEED)  # still open: clear the stack first
        self.on_phase("retreated", target_xy=list(xy))
