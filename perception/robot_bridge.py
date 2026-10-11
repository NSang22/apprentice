"""Taher's robot.perception (segment.detect, cubefit.fit_cube) with perception's block size.

robot/perception/config.py CUBE_SIZE_MM is the robot's value (48 mm, the old foam cubes). The
Playskool blocks are bigger, and perception must size blobs and fit cube outlines with the real
block, but robot/ files are the robot owner's. set_block_mm() overrides the size inside the
modules perception imports, in this process only: segment's and cubefit's module constants and
cubefit.cube_corners' default size. The robot's own processes keep reading config.py.

ACTIVE_COLORS also comes from robot/perception/config.py: under our lighting orange/yellow and
teal/light blue are not separable, so only red, purple, yellow, teal are in play.
"""

from __future__ import annotations

from typing import Optional

from robot.perception import config as robot_config
from robot.perception import cubefit, segment

ROBOT_NAMES = {"light blue": "skyblue", "light_blue": "skyblue"}  # robot spelling -> shared COLORS


def default_active_colors() -> list[str]:
    return [ROBOT_NAMES.get(c, c) for c in robot_config.ACTIVE_COLORS]


def set_block_mm(block_mm: Optional[float]) -> tuple[float, str]:
    """Use block_mm (or the robot's CUBE_SIZE_MM) for segmentation and cube fitting. Returns the
    size in use and a start-up line saying where it came from."""
    robot_mm = robot_config.CUBE_SIZE_MM
    size = float(block_mm) if block_mm else float(robot_mm or 48.0)
    segment.CUBE_SIZE_MM = size
    cubefit.CUBE_SIZE_MM = size
    cubefit.cube_corners.__defaults__ = (0.0, size)
    if block_mm:
        line = f"block size {size:g} mm (--block-mm; robot/perception/config.py CUBE_SIZE_MM is {robot_mm})"
    else:
        line = (f"WARNING: block size {size:g} mm from robot/perception/config.py CUBE_SIZE_MM, the old foam "
                "cubes. Measure a Playskool block and pass --block-mm <edge mm>.")
    return size, line
