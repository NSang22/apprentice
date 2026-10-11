"""Find blocks and hands by what changed since the empty board.

Both the live frame and the empty-board reference are warped onto the sheet (SHEET_PX_PER_MM
pixels per mm, table plane) with the locked homography, so the same pixel is the same spot on
the paper. A pixel has changed when its color differs from the reference by more than
CHANGE_DE (Lab, lightness at half weight, after a small blur), except where it only got darker
and stayed colorless: that is a block's shadow on the paper, not an object. A block lying on
a tag changes those pixels too (black and white print vs a colored block), so tags never hide
a block.

Each changed region is then sorted out:
  block    block-sized (the calibrated footprint range) and its colorful pixels sit near a
           calibrated block color. Named by the nearest color, its cell from its centre
           (corrected for the block's height), scored with the usual area / margin / fill
           confidence. Two touching blocks of different colors are split by color.
  hand     far bigger than any block (HAND_OF_MAX_AREA x the largest footprint)
  object   block-sized or bigger but not a block color (skin, a sleeve, a phone)
A hand or object anywhere on the sheet means hands_clear false. Small specks are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from perception.cells import edge_margin_mm, grid_geometry, mm_to_cell
from perception.colors import Calibration, chroma, classify, delta_e, label_pixels, specular, to_lab
from shared.config import CELLS, COLORS, UNKNOWN

SHEET_MM = (215.9, 279.4)    # US Letter, the printed sheet (workspace_sheet.json frame)
SHEET_PX_PER_MM = 2.0
CHANGE_DE = 15.0             # color difference from the reference that counts as a change
SHADOW_CHROMA = 12.0         # paper in a block's shadow stays this colorless ...
SHADOW_MIN_L_RATIO = 0.45    # ... and keeps at least this share of its lightness (a dark sleeve does not)
NOISE_OF_BLOCK = 0.25        # regions smaller than this share of the smallest block are ignored
PART_OF_BLOCK = 0.5          # a color patch this share of the smallest block counts as its own block
MIN_COLORFUL = 0.35          # a block's region is at least this colorful (not white, gray, or black)
HAND_OF_MAX_AREA = 2.5       # this many times the largest block footprint is a hand or arm
FILL_FULL = 0.5
EDGE_MARGIN_MM = 8.0         # block centre must sit this far inside its cell
STRADDLE_CONFIDENCE = 0.5    # confidence cap for a block too close to a cell line


@dataclass
class Block:
    color: str
    center_mm: tuple[float, float]   # height-corrected centre on the sheet
    contour: np.ndarray              # outline in sheet-warp pixels
    area_mm2: float
    fill_ratio: float
    lab: tuple[float, float, float]
    d_nearest: float
    d_second: float
    area_score: float
    margin_score: float
    fill_score: float
    cell: Optional[str] = None
    edge_mm: float = 0.0

    @property
    def score(self) -> float:
        return min(self.area_score, self.margin_score, self.fill_score)


@dataclass
class Region:
    kind: str                        # "hand" or "object"
    contour: np.ndarray              # sheet-warp pixels
    area_mm2: float
    why: str


@dataclass
class Analysis:
    blocks: list[Block]
    others: list[Region]             # hands and unknown objects
    mask: np.ndarray                 # changed pixels, sheet-warp

    @property
    def empty(self) -> bool:
        return not self.blocks and not self.others


# ---------------------------------------------------------------- geometry


def sheet_size_px() -> tuple[int, int]:
    return int(round(SHEET_MM[0] * SHEET_PX_PER_MM)), int(round(SHEET_MM[1] * SHEET_PX_PER_MM))


def warp_matrix(H: np.ndarray) -> np.ndarray:
    """Image px -> sheet-warp px."""
    return np.diag([SHEET_PX_PER_MM, SHEET_PX_PER_MM, 1.0]) @ np.asarray(H, np.float64)


def warp_to_sheet(frame_bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame_bgr, warp_matrix(H), sheet_size_px(), flags=cv2.INTER_LINEAR)


def sheet_to_image(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Sheet-warp pixels (N,2) back to camera image pixels, for drawing."""
    inv = np.linalg.inv(warp_matrix(H))
    return cv2.perspectiveTransform(np.asarray(points_px, np.float64).reshape(-1, 1, 2), inv).reshape(-1, 2)


def lift(xy_mm: tuple[float, float], camera_xyz: Optional[np.ndarray], height_mm: float) -> tuple[float, float]:
    """A table-plane point under a camera ray, moved to where that ray meets height_mm (the
    block's middle), so a tall block's centre lands in the cell it stands in."""
    if camera_xyz is None or not height_mm:
        return xy_mm
    c = np.asarray(camera_xyz, np.float64)
    xy = c[:2] + (np.asarray(xy_mm) - c[:2]) * (c[2] - height_mm) / c[2]
    return float(xy[0]), float(xy[1])


def block_area_mm2(calib: Calibration) -> tuple[float, float]:
    """(smallest, largest) believable block footprint on the sheet, from calibration."""
    cell = grid_geometry()[2] ** 2
    return calib.area_frac[0] * cell, calib.area_frac[2] * cell


# ---------------------------------------------------------------- change


def change_mask(live_sheet: np.ndarray, ref_sheet: np.ndarray, threshold: float = CHANGE_DE) -> np.ndarray:
    """uint8 mask of pixels that differ from the empty board, shadows left out."""
    live = to_lab(cv2.GaussianBlur(live_sheet, (5, 5), 0))
    ref = to_lab(cv2.GaussianBlur(ref_sheet, (5, 5), 0))
    changed = delta_e(live, ref) > threshold
    shadow = ((chroma(live) < SHADOW_CHROMA) & (chroma(ref) < SHADOW_CHROMA)
              & (live[..., 0] < ref[..., 0]) & (live[..., 0] >= SHADOW_MIN_L_RATIO * ref[..., 0]))
    mask = (changed & ~shadow).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def fill_score(fill_ratio: float) -> float:
    return max(0.0, min(1.0, fill_ratio / FILL_FULL))


def area_score(area: float, expected: tuple[float, float]) -> float:
    lo, hi = expected
    if area < lo:
        return max(0.0, area / lo)
    if area > hi:
        return hi / area
    return 1.0


def analyze(live_sheet: np.ndarray, ref_sheet: np.ndarray, calib: Calibration,
            camera_xyz: Optional[np.ndarray] = None) -> Analysis:
    """Blocks, hands, and unknown objects on the sheet, from the change against the empty board."""
    ppm2 = SHEET_PX_PER_MM ** 2
    lo, hi = block_area_mm2(calib)
    mask = change_mask(live_sheet, ref_sheet)
    lab = to_lab(live_sheet)
    colorful = (chroma(lab) >= calib.fg_chroma) & ~specular(lab)
    blocks: list[Block] = []
    others: list[Region] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour) / ppm2
        if area < NOISE_OF_BLOCK * lo:
            continue
        region = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)  # holes (highlights) filled
        parts = _split_by_color(region, lab, colorful, calib, lo * ppm2)
        if len(parts) > 1:  # touching blocks of different colors, maybe with a hand on them
            covered = np.zeros(mask.shape, np.uint8)
            for part in parts:
                found = _block_or_object(part, lab, colorful, calib, lo, hi, camera_xyz)
                if isinstance(found, Block):
                    blocks.append(found)
                elif found is not None:
                    others.append(found)
                covered |= part  # judged already, block or not
            others += _leftover(region, covered, lo, hi)
            continue
        if area > HAND_OF_MAX_AREA * hi:
            others.append(Region("hand", contour, area, f"{area:.0f} mm2, bigger than any block"))
            continue
        inside = region > 0
        share = np.count_nonzero(colorful & inside) / max(np.count_nonzero(inside), 1)
        if share < MIN_COLORFUL:
            if area >= PART_OF_BLOCK * lo:
                others.append(Region("object", contour, area, f"only {share:.0%} colorful: not a block"))
            continue
        found = _block_or_object(region, lab, colorful, calib, lo, hi, camera_xyz)
        if isinstance(found, Block):
            blocks.append(found)
        elif found is not None:
            others.append(found)
    return Analysis(blocks, others, mask)


def _leftover(region: np.ndarray, covered: np.ndarray, lo: float, hi: float) -> list[Region]:
    """What is left of a region once its blocks are taken out: block-sized or bigger means a hand
    or an object touching them. Thin rims around the blocks (edges, shadows) are opened away."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    rest = cv2.morphologyEx(cv2.bitwise_and(region, cv2.bitwise_not(cv2.dilate(covered, kernel))), cv2.MORPH_OPEN, kernel)
    out = []
    for c in cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c) / SHEET_PX_PER_MM ** 2
        if area >= PART_OF_BLOCK * lo:
            kind = "hand" if area > HAND_OF_MAX_AREA * hi else "object"
            out.append(Region(kind, c, area, f"{area:.0f} mm2 touching the blocks, not a block"))
    return out


def _split_by_color(region: np.ndarray, lab: np.ndarray, colorful: np.ndarray, calib: Calibration,
                    min_block_px: float) -> list[np.ndarray]:
    """The region as one block, or one part per color when touching blocks of different colors
    each cover PART_OF_BLOCK of a block (a shaded side or an edge never does)."""
    pixels = colorful & (region > 0)
    labels = label_pixels(lab, pixels.astype(np.uint8), calib.refs)
    counts = np.bincount(labels[pixels].astype(np.int64) + 1, minlength=len(COLORS) + 2)[1:]
    need = [PART_OF_BLOCK * min_block_px] * len(COLORS) + [min_block_px]
    major = [k for k, c in enumerate(counts) if c >= need[k]]
    if len(major) <= 1:
        return [region]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return [cv2.morphologyEx(((labels == k) & (region > 0)).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
            for k in major]


def _block_or_object(part: np.ndarray, lab: np.ndarray, colorful: np.ndarray, calib: Calibration,
                     lo: float, hi: float, camera_xyz: Optional[np.ndarray]):
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour) / SHEET_PX_PER_MM ** 2
    if area < PART_OF_BLOCK * lo:
        return None
    pixels = lab[(part > 0) & colorful]
    if len(pixels) == 0:
        return None
    median = tuple(float(v) for v in np.median(pixels, axis=0))
    color, margin, d1, d2 = classify(median, calib.refs)
    if color == UNKNOWN:
        return Region("object", contour, area, f"color {d1:.0f} dE from the nearest block color")
    m = cv2.moments(contour)
    cx, cy = m["m10"] / m["m00"] / SHEET_PX_PER_MM, m["m01"] / m["m00"] / SHEET_PX_PER_MM
    center = lift((cx, cy), camera_xyz, calib.side_mm / 2.0)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    fill = float(min(1.0, cv2.contourArea(contour) / max(w * h, 1.0)))
    return Block(color, center, contour, area, fill, median, d1, d2,
                 area_score(area, (lo, hi)), margin, fill_score(fill),
                 cell=mm_to_cell(*center), edge_mm=edge_margin_mm(*center))


# ---------------------------------------------------------------- board


def board_from(analysis: Analysis) -> tuple[dict[str, list[str]], dict[str, float], bool]:
    """(board, confidence, crowded). Each cell gets its best block; a second block in one cell
    sets crowded (something is wrong there, e.g. a hand holding one). Empty cells are 1.0."""
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
