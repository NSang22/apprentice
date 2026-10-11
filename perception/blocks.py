"""Blocks, hands, and unknown objects from one frame, with Taher's robot.perception pipeline.

Per frame, with the locked sheet pose and the empty-board reference (perception/grid_lock.py):
  1. segment.detect finds colorful blobs on the sheet and splits touching blocks by hue. It runs
     on a crop around the locked sheet, scaled to SEG_WIDTH pixels wide (the full frame took
     ~0.75 s); its blobs are mapped back to full-frame pixels for everything after.
  2. A blob counts only if at least CHANGED_SHARE of it changed since the empty board
     (perception/change.py), so print, tape, or anything already there when the board was
     empty is ignored. Changed area that no blob explains (a hand, a sleeve) is a hand or object.
  3. The blob centre goes to sheet mm at the block's mid-height (pose.to_sheet(c, size/2), as
     scene.build_scene does), then cubefit.fit_cube refines it into the position and yaw of a
     real cube outline. That corrected centre names the cell (perception/cells.py).
  4. The color is sampled only from the central TOP_CORE of the fitted top face (highlights
     dropped) and named by the nearest calibrated reference among the active colors. Anything
     else reads "unknown" (an unknown block, never a hand: the brain asks to nudge it). Only a
     blob bigger than ~3 blocks, or changed area no blob explains, is a hand or object.
     No vision model: nothing here touches the network.
Confidence is min(color margin, fit score). A corrected centre less than EDGE_MARGIN_MM inside
its cell caps it at STRADDLE_CONFIDENCE (the brain then reads that cell as unknown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from perception.cells import edge_margin_mm, mm_to_cell
from perception.change import change_mask, warp_matrix, warp_to_sheet
from perception.colors import ColorRef, classify, sample_stats, to_lab
from robot.perception import cubefit, segment
from robot.perception.sheet import SheetPose
from shared.config import CELLS, UNKNOWN

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
class Block:
    color: str
    center_mm: tuple[float, float]       # corrected centre on the sheet
    center_px: tuple[float, float]       # that centre drawn at the block's top
    contour_px: np.ndarray               # the blob
    outline_px: np.ndarray               # the fitted cube's silhouette
    top_px: np.ndarray                   # where the color was sampled
    yaw_deg: float
    fit_overlap: float
    lab: tuple[float, float, float]
    d_nearest: float
    d_second: float
    margin_score: float
    cell: Optional[str] = None
    edge_mm: float = 0.0

    @property
    def fit_score(self) -> float:
        return max(0.0, min(1.0, self.fit_overlap / FIT_FULL))

    @property
    def score(self) -> float:
        return min(self.margin_score, self.fit_score)


@dataclass
class Region:
    kind: str                            # "hand" or "object"
    contour_px: np.ndarray
    area_mm2: float
    why: str


@dataclass
class Analysis:
    blocks: list[Block]
    others: list[Region]
    mask: np.ndarray                     # changed pixels, sheet warp
    live_sheet: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def empty(self) -> bool:
        return not self.blocks and not self.others


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


def detect_blocks(frame_bgr: np.ndarray, pose: SheetPose, reference_sheet: np.ndarray,
                  refs: dict[str, ColorRef], block_mm: float, seg_width: Optional[int] = SEG_WIDTH) -> Analysis:
    """seg_width: segment.detect runs on the sheet crop scaled down to at most this many pixels
    wide (None: full resolution). Everything after segmentation (cube fit, color, cells) works on
    the full-resolution frame."""
    live_sheet = warp_to_sheet(frame_bgr, pose.H)
    mask = change_mask(live_sheet, reference_sheet)
    x0, y0, x1, y1 = crop_box(pose, frame_bgr.shape)
    crop = frame_bgr[y0:y1, x0:x1]
    scale = min(1.0, seg_width / crop.shape[1]) if seg_width else 1.0
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
    small_pose = crop_pose(pose, x0, y0, scale)
    detections = segment.detect(small, None, roi_mask=sheet_roi(small_pose, small.shape), pose=small_pose)
    changed = changed_in_image(mask, crop_pose(pose, x0, y0).H, crop.shape)  # crop coordinates, full resolution
    ppm = px_per_mm(pose)
    face_px = (block_mm * ppm) ** 2
    offset = np.array([x0, y0], np.float64)
    blocks: list[Block] = []
    others: list[Region] = []
    explained = np.zeros(crop.shape[:2], np.uint8)
    for d in detections:
        contour_crop = np.asarray(d.contour, np.float64).reshape(-1, 2) / scale
        contour = contour_crop + offset                       # full-frame pixels
        center = (d.center_px[0] / scale + x0, d.center_px[1] / scale + y0)
        blob = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(blob, [contour_crop.astype(np.int32)], 255)
        area = np.count_nonzero(blob)
        if area == 0 or np.count_nonzero(changed[blob > 0]) < CHANGED_SHARE * area:
            continue  # it was there on the empty board
        explained |= blob
        area_mm2 = area / ppm ** 2
        if d.score < 0:  # segment.detect: much bigger than one cube
            others.append(Region("hand", contour, area_mm2, f"{area / face_px:.1f} block faces big"))
            continue
        xy, yaw, overlap, outline, top = measure(frame_bgr, pose, contour, center, block_mm)
        try:
            median, _, _ = sample_stats(top_face_lab(frame_bgr, top))
            color, margin, d1, d2 = classify(median, refs)
        except ValueError:
            median, color, margin, d1, d2 = (0.0, 0.0, 0.0), UNKNOWN, 0.0, 0.0, 0.0
        cx, cy = pose.to_image([xy], block_mm)[0]
        blocks.append(Block(color, xy, (float(cx), float(cy)), contour, outline, top, yaw, overlap,
                            median, d1, d2, margin, mm_to_cell(*xy), edge_margin_mm(*xy)))
    others += [Region(r.kind, r.contour_px + offset, r.area_mm2, r.why)
               for r in _leftovers(changed, explained, face_px, ppm)]
    return Analysis(blocks, others, mask, live_sheet)


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


def board_from(analysis: Analysis) -> tuple[dict[str, list[str]], dict[str, float], bool]:
    """(board, confidence, crowded). Each cell gets its best block; a second block in one cell
    sets crowded. A centre near a cell line caps confidence. Empty cells are 1.0."""
    board = {c: [] for c in CELLS}
    confidence = {c: 1.0 for c in CELLS}
    crowded = False
    for b in sorted(analysis.blocks, key=lambda b: b.score, reverse=True):
        if b.cell is None:
            continue
        conf = min(b.score, STRADDLE_CONFIDENCE) if b.edge_mm < EDGE_MARGIN_MM else b.score
        if board[b.cell]:
            crowded = True
            confidence[b.cell] = min(confidence[b.cell], conf)
            continue
        board[b.cell] = [b.color]
        confidence[b.cell] = conf
    return board, confidence, crowded
