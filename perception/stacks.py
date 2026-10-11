"""Stacks: how tall each block pile is, and which colors are under the top.

Three signals, combined so that disagreement reads "unknown" and is never guessed:

  geometry   fit_column fits a column of 1, 2, or 3 blocks (footprint on the table, top at
             (level + 1) x block size) to the blob with robot/perception/cubefit.py, and
             choose_level keeps the level whose outline overlaps the blob best, if it beats the
             next best by LEVEL_MARGIN. A taller stack's top sits closer to the camera, so its
             blob is bigger and pushed further from the camera's nadir.
  side bands with level >= 1, the part of each lower block's side that the blocks above do not
             hide (projected from the fitted column) is sampled and classified. A band counts only
             with MIN_BAND_PX visible pixels and a clear color margin: near the camera's nadir the
             sides are barely visible, so this signal is often missing there.
  memory     StackMemory follows each cell across stable readings: the same cell reading one
             level higher with a new top is a push, one level lower with the old second color on
             top is a pop, empty resets it. Anything else (a stack swapped in while hands were
             over the board) marks the cell's history as conflicting until it reads empty.

resolve(): the stack is the memory when it matches the geometry and every counted side band;
with no usable memory, the side bands plus the top when all of them were seen; otherwise
"unknown" with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from perception.colors import ColorRef, classify, sample_stats, to_lab
from robot.perception import cubefit
from robot.perception.sheet import SheetPose
from shared.config import CELLS, UNKNOWN

LEVEL_MARGIN = 0.10        # chosen level must beat the next best by this (ideal stacks: >= 0.25, clean single cubes ~0.2)
MIN_BAND_PX = 150          # a side band needs this many visible pixels to count
BAND_MIN_MARGIN = 0.5      # ... and this color margin (perception/colors.margin_score)
BAND_ERODE_PX = 3          # keep away from edges, where neighbours and the paper bleed in
MEMORY_STABLE_FRAMES = 3   # a cell's (level, top) must hold this many usable frames to update memory


# ---------------------------------------------------------------- geometry


def choose_level(overlaps: list[float], min_overlap: float, margin: float = LEVEL_MARGIN) -> tuple[Optional[int], str]:
    """(level, "") for the best-fitting level, or (None, why) when nothing fits well enough or
    the two best levels are too close to call."""
    order = sorted(range(len(overlaps)), key=lambda i: -overlaps[i])
    best = order[0]
    if overlaps[best] < min_overlap:
        return None, f"no stack height fits (best overlap {overlaps[best]:.2f})"
    if len(order) > 1 and overlaps[best] - overlaps[order[1]] < margin:
        return None, (f"height unclear: {best + 1} or {order[1] + 1} high "
                      f"({overlaps[best]:.2f} vs {overlaps[order[1]]:.2f})")
    return best, ""


def fit_column(pose: SheetPose, contour_px: np.ndarray, center_px: tuple[float, float], block_mm: float,
               level: int) -> tuple[np.ndarray, float, float]:
    """cubefit.fit_cube for a column of level + 1 blocks: (footprint centre mm, yaw, overlap)."""
    height = (level + 1) * block_mm
    start = pose.to_sheet([center_px], height / 2.0)[0]
    saved = cubefit.CUBE_SIZE_MM
    cubefit.CUBE_SIZE_MM = height  # the outline's height; its width stays block_mm (robot_bridge)
    try:
        return cubefit.fit_cube(pose, contour_px, start, 0.0, base_mm=0.0)
    finally:
        cubefit.CUBE_SIZE_MM = saved


def segment_hull(pose: SheetPose, xy, yaw: float, block_mm: float, lo_mm: float, hi_mm: float) -> np.ndarray:
    sq = cubefit.cube_corners(xy, yaw, size_mm=block_mm)
    pts = np.vstack([pose.to_image(sq, lo_mm), pose.to_image(sq, hi_mm)]).astype(np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


def side_bands(frame_bgr: np.ndarray, pose: SheetPose, xy, yaw: float, block_mm: float, level: int,
               blob_px: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """For each block under the top (bottom first): (its visible Lab pixels, how many). Visible =
    inside the blob, inside that block's projected column segment, outside every segment above."""
    hulls = [segment_hull(pose, xy, yaw, block_mm, k * block_mm, (k + 1) * block_mm) for k in range(level + 1)]
    pts = np.vstack(hulls + [np.asarray(blob_px, np.float64).reshape(-1, 2)])
    h, w = frame_bgr.shape[:2]
    x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + 2, [w, h])
    if x1 <= x0 or y1 <= y0:
        return [(np.zeros((0, 3), np.float32), 0) for _ in range(level)]
    shape, off = (y1 - y0, x1 - x0), np.array([x0, y0])

    def fill(poly):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [(np.asarray(poly) - off).astype(np.int32)], 255)
        return m

    blob = fill(blob_px)
    lab = to_lab(frame_bgr[y0:y1, x0:x1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_ERODE_PX + 1, 2 * BAND_ERODE_PX + 1))
    out = []
    for k in range(level):
        above = np.zeros(shape, np.uint8)
        for j in range(k + 1, level + 1):
            above |= fill(hulls[j])
        band = cv2.erode(fill(hulls[k]) & ~above & blob, kernel)
        px = lab[band > 0]
        out.append((px, int(len(px))))
    return out


def band_color(pixels: np.ndarray, count: int, refs: dict[str, ColorRef]) -> Optional[str]:
    """A side band's color, or None when too few pixels or no clear winner."""
    if count < MIN_BAND_PX:
        return None
    try:
        median, _, _ = sample_stats(pixels)
    except ValueError:
        return None
    color, margin, _, _ = classify(median, refs)
    return None if color == UNKNOWN or margin < BAND_MIN_MARGIN else color


# ---------------------------------------------------------------- memory


@dataclass(frozen=True)
class Reading:
    """One cell in one frame: the stack height index (0 = one block), the top color, and the
    colors seen on the side bands under it (bottom first, None = not seen)."""
    level: Optional[int]
    top: str
    sides: tuple[Optional[str], ...] = ()


_NO_HISTORY = object()


@dataclass
class StackMemory:
    stable_frames: int = MEMORY_STABLE_FRAMES
    stacks: dict = field(default_factory=dict)    # cell -> list of colors, or None (conflict)
    pending: dict = field(default_factory=dict)   # cell -> ((level, top) or None, frames held)

    def update(self, readings: dict[str, Optional[Reading]], usable: bool) -> None:
        """Feed one frame's readings (None = the cell is empty). Unusable frames (hands, grid
        not locked) teach nothing and restart every cell's count."""
        if not usable:
            self.pending.clear()
            return
        for cell in CELLS:
            r = readings.get(cell)
            key = None if r is None else (r.level, r.top)
            held = self.pending.get(cell)
            count = held[1] + 1 if held and held[0] == key else 1
            self.pending[cell] = (key, count)
            if count == self.stable_frames:  # just became stable: act once
                self._transition(cell, r)

    def _transition(self, cell: str, r: Optional[Reading]) -> None:
        if r is None:
            self.stacks[cell] = []        # empty: forget
            return
        m = self.stacks.get(cell, [])
        if m is None or r.level is None or r.top == UNKNOWN:
            return                        # conflicting history waits for empty; no information
        level, top = r.level, r.top
        if not m:
            if level == 0:
                self.stacks[cell] = [top]
            elif all(s is not None for s in r.sides):
                self.stacks[cell] = list(r.sides) + [top]   # first seen as a stack: trust the sides
            else:
                self.stacks.pop(cell, None)                  # no history to go on
        elif len(m) == level + 1 and m[-1] == top:
            pass                                             # unchanged
        elif level == len(m):
            self.stacks[cell] = m + [top]                    # push
        elif level == len(m) - 2 and m[-2] == top:
            self.stacks[cell] = m[:-1]                       # pop
        else:
            self.stacks[cell] = None                         # conflict until the cell reads empty

    def resolve(self, cell: str, r: Reading) -> tuple[Optional[list[str]], str]:
        """(stack bottom to top, "") or (None, why the cell is unknown)."""
        if r.level is None:
            return None, "stack height unclear"
        if r.top == UNKNOWN:
            return None, "top color unclear"
        m = self.stacks.get(cell, _NO_HISTORY)
        if m is None:
            return None, "stack history conflicts: clear this cell"
        seen = [(k, c) for k, c in enumerate(r.sides) if c is not None]
        if m is _NO_HISTORY or len(m) != r.level + 1 or m[-1] != r.top:
            if len(seen) == r.level:
                return list(r.sides) + [r.top], ""
            return None, f"{r.level + 1} high but the colors under the top are hidden"
        for k, c in seen:
            if m[k] != c:
                return None, f"level {k + 1} looks {c}, but {m[k]} was placed there"
        return list(m), ""


def unknown_stack(level: Optional[int]) -> list[str]:
    return [UNKNOWN] * ((level if level is not None else 0) + 1)

