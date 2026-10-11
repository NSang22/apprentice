"""robot.perception.cubefit: recover a cube's position and yaw from its projected outline."""

import cv2
import numpy as np

from robot.perception import cubefit
from robot.perception.config import CUBE_SIZE_MM
from robot.perception.sheet import SheetPose

# 3 px per mm, camera beyond the sheet's far edge like the real rig (run 19:14)
POSE = SheetPose(H=np.diag([1 / 3, 1 / 3, 1.0]), tag_ids=[0, 1, 2, 3], fit_error_mm=0.5,
                 camera_xyz=np.array([143.0, 285.0, 415.0]))


def blob(xy, yaw, base=0.0):
    sq = cubefit.cube_corners(xy, yaw)
    pts = np.vstack([POSE.to_image(sq, base), POSE.to_image(sq, base + CUBE_SIZE_MM)])
    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)


def test_recovers_a_turned_cube():
    xy, yaw, overlap = cubefit.fit_cube(POSE, blob((108, 120), 12.0), (114, 126), 0.0)
    assert np.hypot(*(xy - [108, 120])) < 1.0 and abs(yaw - 12.0) < 1.5 and overlap > 0.95


def test_square_cube_with_a_glare_cut_corner_does_not_read_as_turned():
    # run 19:14: glare ate one corner, the box rotation said 27 deg for a square cube
    c = blob((50, 84), 0.0)
    keep = np.array([p for p in c if not (p[0] < c[:, 0].min() + 40 and p[1] < c[:, 1].min() + 40)])
    xy, yaw, overlap = cubefit.fit_cube(POSE, cv2.convexHull(keep.astype(np.float32)).reshape(-1, 2), (50, 84), 27.0)
    assert abs(yaw) < 10.0


def test_stacked_cube_at_its_height():
    xy, yaw, overlap = cubefit.fit_cube(POSE, blob((165, 131), 0.0, base=48.0), (170, 125), 0.0, base_mm=48.0)
    assert np.hypot(*(xy - [165, 131])) < 1.0 and overlap > 0.95
