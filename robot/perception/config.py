"""Rig constants for the overhead camera and test objects. Measure, don't guess."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # robot/
SHEET_SPEC = ROOT / "assets" / "workspace_sheet.json"

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 1))  # kit USB camera, mounted top-down (index 0 is the user-facing webcam on Windows; the Jetson sets 0 in jetson-kushi/.env)
FRAME_W, FRAME_H = 1920, 1080

# Lens height above the table surface, mm. Used to correct parallax for raised objects
# (cube tops). None = no correction (positions of raised objects shift away from the camera's nadir).
CAMERA_HEIGHT_MM: float | None = 432.0  # 9 cubes, measured 2026-10-10 at the new location (was 468)

# Camera centre in sheet mm (x, y, height), pinned. None = estimate it from the tags every frame.
# The per-frame estimate is ill-conditioned for a tilted wide-angle camera: on 2026-10-11 it jumped
# 177 mm in y between a 4-tag and a 3-tag view of the same setup, which moved every cube reading
# by 10-20 mm. A pinned value that is slightly wrong only shifts all readings by the same amount,
# and auto_calibrate's correction absorbs that. Re-pin after moving the camera:
#   python -m robot.scripts.calibration.pin_camera
CAMERA_XYZ_MM: tuple[float, float, float] | None = (105.7, 226.2, 423.2)  # pin_camera, 2026-10-11 02:43

# Test objects: name -> edge length in mm. None until measured.
CUBE_SIZE_MM: float | None = 48.0  # foam cubes, measured
OBJECTS = ["red_cube", "blue_cube", "yellow_cube"]

# The block colours this kit actually has. Guesses only choose from these. The order is also
# colour calibration's default spot order (A1, A3, B2, C1, C2, C3). Six blocks on a 3x3 grid
# can't all avoid neighbours, so the only neighbours (row C, and B2 above C2) are dissimilar
# colours: similar colours that touch merge and can't be split (teal + light blue at 17:24).
BLOCK_COLORS = ["orange", "yellow", "purple", "teal", "red", "light blue"]

# The colours blocks may be named, out of BLOCK_COLORS. In this room's glare orange and yellow
# share a hue (105 deg; the yellow block was called "orange" 53 times out of 59), and teal and
# light blue sit 6 deg apart, so only one of each pair is in play. The colour memory keeps all of
# them; this only narrows the choice. Use only blocks of these colours in tasks.
ACTIVE_COLORS = ["red", "purple", "yellow", "teal"]  # perception/ reads this too: team-agreed demo colours

# Colours the robot (not perception/) may also name, for the cube bank's 5th block. Jetson table,
# 2026-10-11, after color_calibrate in this room: orange sits 11.7 from its own calibration and 43.8
# from yellow (yellow: 5.9 vs 58.5). Teal vs light blue is still the closest pair (10.2 vs 29.3), so
# light blue stays out. Re-check after moving rooms.
ROBOT_EXTRA_COLORS = ["orange", "light blue"]  # light blue: tested apart from teal on 2026-10-11 03:4x, see below

# Cube bank: spare blocks on the mat right of the sheet, for building from scratch. Where block
# centres may sit, as (x0, y0, x1, y1) sheet mm; None = no bank. Inside the green tape box and
# where the gripper can reach straight down (a block centred at y -13 could not be gripped).
# TWO blocks at a time, refilled as the arm works (the executor waits a few seconds for a block
# it can't see yet): one at the top of the bank, square to the sheet, one at the bottom, set down
# as a diamond (turned ~40 deg). Reason: the arm is 15-20 mm off whenever the wrist rolls more
# than ~30 deg, and a square block gets a small roll only in the top third of the bank (grip
# planner, 2026-10-11: roll 14-30 for y <= 100, but 36-44 from y 125 down). Keep blocks ~9 cm
# apart so the small-roll grip's moving jaw (it swings ~85 mm) clears the other one.
# Everything outside this box is invisible to the robot, even as an obstacle: keep it clear.
BANK_ZONE_MM: tuple[float, float, float, float] | None = (238.0, 32.0, 310.0, 262.0)
