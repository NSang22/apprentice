"""Fit a cube's 3D outline to its segmented blob.

The camera sits off to one side of the sheet, so a blob is a cube's top face plus the sides
facing the camera, and glare can eat a corner of it. The rotation of the blob's bounding box
is then not the cube's yaw (run 19:14: an L-shaped glare-cut blob read as 27 deg for a cube
sitting square, and the gripper twisted it). Here we project a real 48 mm cube with the
camera pose and search for the position and yaw whose outline overlaps the blob best.
"""

import cv2
import numpy as np

from .config import CUBE_SIZE_MM

PAD_PX = 60
STEPS = ((8.0, 8.0), (4.0, 4.0), (2.0, 2.0), (1.0, 1.0), (0.5, 0.5))  # (mm, deg) per search stage


def cube_corners(xy, yaw_deg, base_mm=0.0, size_mm=CUBE_SIZE_MM):
    """The 4 corners (sheet mm) of a square of side size_mm centred at xy. yaw follows
    scene.py: counterclockwise seen from above is positive (sheet y points down the page)."""
    a = -np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    h = size_mm / 2
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return sq @ np.array([[c, s], [-s, c]]) + np.asarray(xy, float)


def _outline_mask(pose, xy, yaw_deg, base_mm, shape, offset):
    sq = cube_corners(xy, yaw_deg)
    pts = np.vstack([pose.to_image(sq, base_mm), pose.to_image(sq, base_mm + CUBE_SIZE_MM)]) - offset
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.float32)).astype(np.int32), 1)
    return m


def fit_cube(pose, contour_px, xy0, yaw0_deg, base_mm=0.0):
    """Returns (xy, yaw_deg, overlap): the cube pose whose projected outline best matches the
    blob, and the intersection-over-union of the two (1.0 = perfect)."""
    contour = np.asarray(contour_px, np.float64).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    offset = np.array([x - PAD_PX, y - PAD_PX])
    shape = (h + 2 * PAD_PX, w + 2 * PAD_PX)
    blob = np.zeros(shape, np.uint8)
    cv2.fillPoly(blob, [(contour - offset).astype(np.int32)], 1)

    def overlap(p):
        m = _outline_mask(pose, p[:2], p[2], base_mm, shape, offset)
        return np.count_nonzero(m & blob) / max(np.count_nonzero(m | blob), 1)

    p = np.array([xy0[0], xy0[1], yaw0_deg], float)
    best = overlap(p)
    for step_mm, step_deg in STEPS:  # pattern search: try each direction, keep what helps
        improved = True
        while improved:
            improved = False
            for d in ((step_mm, 0, 0), (-step_mm, 0, 0), (0, step_mm, 0), (0, -step_mm, 0),
                      (0, 0, step_deg), (0, 0, -step_deg)):
                q = p + d
                v = overlap(q)
                if v > best + 1e-6:
                    p, best, improved = q, v, True
    yaw = float((p[2] + 45) % 90 - 45)
    return p[:2], yaw, float(best)
