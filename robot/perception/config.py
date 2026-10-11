"""Rig constants for the overhead camera and test objects. Measure, don't guess."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # robot/
SHEET_SPEC = ROOT / "assets" / "workspace_sheet.json"

CAMERA_INDEX = 1          # kit USB camera, mounted top-down (index 0 is the user-facing webcam)
FRAME_W, FRAME_H = 1920, 1080

# Lens height above the table surface, mm. Used to correct parallax for raised objects
# (cube tops). None = no correction (positions of raised objects shift away from the camera's nadir).
CAMERA_HEIGHT_MM: float | None = 432.0  # 9 cubes, measured 2026-10-10 at the new location (was 468)

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
ACTIVE_COLORS = ["red", "purple", "yellow", "teal"]
