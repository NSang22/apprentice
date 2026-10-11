"""Helpers around Taher's robot.perception pipeline (segment.detect + cubefit), used by
perception/calibrate.py to find the one block in B2, and by perception/cellmodel.py (crop,
pose, sheet mask, leftover hands). The live board reading is perception/cellmodel.py; the
notes below describe the blob-first pipeline calibration still uses.

Per frame, with the locked sheet pose and the empty-board reference (perception/grid_lock.py):
  1. segment.detect finds colorful blobs on the sheet and splits touching blocks by hue. It runs
     on a crop around the locked sheet, scaled to SEG_WIDTH pixels wide (the full frame took
     ~0.75 s); its blobs are mapped back to full-frame pixels for everything after.
  2. A blob counts only if at least CHANGED_SHARE of it changed since the empty board
     (perception/change.py), so print, tape, or anything already there when the board was
     empty is ignored. Changed area that no blob explains (a hand, a sleeve) is a hand or object.
  3. The blob is fitted as a column of 1, 2, and 3 blocks with cubefit (perception/stacks.py)
     and the best-fitting height is kept (or none, if two heights are too close to call). The
     column's footprint centre, corrected for that height, names the cell (perception/cells.py).
     For a stack, the visible side band of each block under the top is classified too.
  4. The top color is sampled only from the central TOP_CORE of the fitted top face (highlights
     dropped) and named by the nearest calibrated reference among the active colors. Anything
     else reads "unknown" (an unknown block, never a hand: the brain asks to nudge it). Only a
     blob bigger than ~3 blocks, or changed area no blob explains, is a hand or object.
     No vision model: nothing here touches the network.
Confidence is min(color margin, fit score). A corrected centre less than EDGE_MARGIN_MM inside
its cell caps it at STRADDLE_CONFIDENCE (the brain then reads that cell as unknown).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.change import warp_matrix
from perception.colors import to_lab
from robot.perception import cubefit
from robot.perception.sheet import SheetPose

SHEET_W_MM, SHEET_H_MM = 215.9, 279.4
SHEET_MARGIN_MM = 5.0
CROP_MARGIN_MM = 15.0      # segmentation sees only the sheet plus this margin, not the whole frame
SEG_WIDTH = 640            # ... scaled down to at most this many pixels wide
CHANGED_SHARE = 0.5        # a blob counts only if this share of it changed since the empty board
FIT_MIN_OVERLAP = 0.6      # fitted cube outline vs blob IoU needed to trust the fit's position
FIT_FULL = 0.8             # IoU at or above this scores 1.0 (clean blobs fit 0.87-0.92)
EDGE_MARGIN_MM = 10.0      # corrected centre must sit this far inside its cell
STRADDLE_CONFIDENCE = 0.5
TOP_CORE = 0.5             # sample color from this central fraction of the top face's edge
LEFTOVER_FACES = 0.5       # unexplained changed area this many block faces big is a hand or object
HAND_FACES = 2.8           # bigger than this many faces: a hand (segment.MERGED_FACES)
EXPLAINED_PAD_PX = 15      # grow explained blobs by this before taking leftovers (rims, halos)


@dataclass
class Region:
    kind: str                            # "hand" or "object"
    contour_px: np.ndarray
    area_mm2: float
    why: str


def px_per_mm(pose: SheetPose) -> float:
    return float(np.linalg.norm(pose.to_image([(100, 100)])[0] - pose.to_image([(110, 100)])[0]) / 10)


def sheet_roi(pose: SheetPose, shape: tuple[int, ...], margin_mm: float = SHEET_MARGIN_MM) -> np.ndarray:
    m = margin_mm
    poly = pose.to_image([(-m, -m), (SHEET_W_MM + m, -m), (SHEET_W_MM + m, SHEET_H_MM + m), (-m, SHEET_H_MM + m)])
    roi = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(roi, [poly.astype(np.int32)], 255)
    return roi


def changed_in_image(mask_sheet: np.ndarray, H: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    inv = np.linalg.inv(warp_matrix(H))
    return cv2.warpPerspective(mask_sheet, inv, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)


def measure(frame_bgr: np.ndarray, pose: SheetPose, contour_px: np.ndarray, center_px: tuple[float, float],
            block_mm: float) -> tuple[tuple[float, float], float, float, np.ndarray, np.ndarray]:
    """(corrected centre mm, yaw, fit overlap, fitted outline px, central top-face polygon px)."""
    mid = pose.to_sheet([center_px], block_mm / 2.0)[0]
    xy, yaw, overlap = cubefit.fit_cube(pose, contour_px, mid, 0.0)
    if overlap < FIT_MIN_OVERLAP:
        xy, yaw = mid, 0.0
    sq = cubefit.cube_corners(xy, yaw, size_mm=block_mm)
    outline = cv2.convexHull(np.vstack([pose.to_image(sq, 0.0), pose.to_image(sq, block_mm)]).astype(np.float32))
    top = pose.to_image(cubefit.cube_corners(xy, yaw, size_mm=block_mm * TOP_CORE), block_mm)
    return (float(xy[0]), float(xy[1])), float(yaw), float(overlap), outline.reshape(-1, 2), top


def top_face_lab(frame_bgr: np.ndarray, top_px: np.ndarray) -> np.ndarray:
    """Lab pixels inside the top-face polygon (cropped first: no full-frame Lab per block)."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(top_px.astype(np.int32))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), np.float32)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [(top_px - [x0, y0]).astype(np.int32)], 255)
    return to_lab(frame_bgr[y0:y1, x0:x1])[mask > 0]


def crop_box(pose: SheetPose, shape: tuple[int, ...], margin_mm: float = CROP_MARGIN_MM) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the image rectangle around the sheet plus margin_mm, clipped to the frame."""
    m = margin_mm
    pts = pose.to_image([(-m, -m), (SHEET_W_MM + m, -m), (SHEET_W_MM + m, SHEET_H_MM + m), (-m, SHEET_H_MM + m)])
    h, w = shape[:2]
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def crop_pose(pose: SheetPose, x0: int, y0: int, scale: float = 1.0) -> SheetPose:
    """The same sheet pose for an image cropped at (x0, y0) and then resized by `scale`."""
    to_full = np.array([[1.0 / scale, 0.0, x0], [0.0, 1.0 / scale, y0], [0.0, 0.0, 1.0]])
    return SheetPose(np.asarray(pose.H, np.float64) @ to_full, pose.tag_ids, pose.fit_error_mm, pose.camera_xyz)


def _leftovers(changed: np.ndarray, explained: np.ndarray, face_px: float, ppm: float) -> list[Region]:
    """Changed area no blob explains: a hand or object if it is at least LEFTOVER_FACES of a block."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EXPLAINED_PAD_PX + 1, 2 * EXPLAINED_PAD_PX + 1))
    rest = cv2.bitwise_and(changed, cv2.bitwise_not(cv2.dilate(explained, k)))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    out = []
    for c in cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area >= LEFTOVER_FACES * face_px:
            kind = "hand" if area > HAND_FACES * face_px else "object"
            out.append(Region(kind, c.reshape(-1, 2), area / ppm ** 2, "changed, but not a colorful block"))
    return out
