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
