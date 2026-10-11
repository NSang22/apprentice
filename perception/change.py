"""What changed on the sheet since the empty board.

Both the live frame and the empty-board reference are warped onto the sheet (SHEET_PX_PER_MM
pixels per mm, table plane) with the locked homography, so the same pixel is the same spot on
the paper. A pixel has changed when its color differs from the reference by more than
CHANGE_DE (Lab, lightness at half weight, after a small blur), except where it only got darker
and stayed colorless: that is a block's shadow on the paper, not an object. A block lying on a
tag changes those pixels too (black and white print vs a colored block), so tags never hide a
block. perception/blocks.py decides what each changed area is.
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.colors import chroma, delta_e, to_lab

SHEET_MM = (215.9, 279.4)    # US Letter, the printed sheet (workspace_sheet.json frame)
SHEET_PX_PER_MM = 2.0
CHANGE_DE = 15.0             # color difference from the reference that counts as a change
SHADOW_CHROMA = 12.0         # paper in a block's shadow stays this colorless ...
SHADOW_MIN_L_RATIO = 0.45    # ... and keeps at least this share of its lightness (a dark sleeve does not)


def sheet_size_px() -> tuple[int, int]:
    return int(round(SHEET_MM[0] * SHEET_PX_PER_MM)), int(round(SHEET_MM[1] * SHEET_PX_PER_MM))


def warp_matrix(H: np.ndarray) -> np.ndarray:
    """Image px -> sheet-warp px."""
    return np.diag([SHEET_PX_PER_MM, SHEET_PX_PER_MM, 1.0]) @ np.asarray(H, np.float64)


def warp_to_sheet(frame_bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(as_bgr(frame_bgr), warp_matrix(H), sheet_size_px(), flags=cv2.INTER_LINEAR)


def sheet_to_image(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Sheet-warp pixels (N,2) back to camera image pixels, for drawing."""
    inv = np.linalg.inv(warp_matrix(H))
    return cv2.perspectiveTransform(np.asarray(points_px, np.float64).reshape(-1, 1, 2), inv).reshape(-1, 2)


def as_bgr(img: np.ndarray) -> np.ndarray:
    """3-channel uint8 whatever came in (gray or BGRA frames, a reference saved differently)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def change_mask(live_sheet: np.ndarray, ref_sheet: np.ndarray, threshold: float = CHANGE_DE) -> np.ndarray:
    """uint8 mask of pixels that differ from the empty board, shadows left out."""
    live_sheet, ref_sheet = as_bgr(live_sheet), as_bgr(ref_sheet)
    if ref_sheet.shape[:2] != live_sheet.shape[:2]:  # a reference from another warp size: compare anyway
        ref_sheet = cv2.resize(ref_sheet, (live_sheet.shape[1], live_sheet.shape[0]))
    live = to_lab(cv2.GaussianBlur(live_sheet, (5, 5), 0))
    ref = to_lab(cv2.GaussianBlur(ref_sheet, (5, 5), 0))
    changed = delta_e(live, ref) > threshold
    shadow = ((chroma(live) < SHADOW_CHROMA) & (chroma(ref) < SHADOW_CHROMA)
              & (live[..., 0] < ref[..., 0]) & (live[..., 0] >= SHADOW_MIN_L_RATIO * ref[..., 0]))
    mask = (changed & ~shadow).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
