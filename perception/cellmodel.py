"""Cell-centric board reading: blocks only sit in the 9 cells, so read each cell, not each blob.

Blob-first detection (segment.detect, then fit each blob) breaks when neighbouring blocks merge
or hide each other. Here every cell is a hypothesis: empty, or a stack of 1-3 blocks of a block
color. All of it runs on the sheet crop scaled to WORK_WIDTH pixels, with the locked pose.

  geometry   Geometry projects, for every cell and level, the block column (footprint on the
             table up to (level + 1) x block size), its top face, and each block's segment, the
             way robot/perception/cubefit.py projects a cube. Cells are drawn farthest from the
             camera first, so nearer stacks cover farther ones (painter's order).
  observed   C: pixels that changed since the empty board (the reference warped back into the
             camera view, perception/change.change_mask). P: each changed, colorful pixel's
             nearest calibrated block color, or "far" (skin, a phone), or colorless.
  render     render() paints a hypothesized board; cost() compares it with C and P. A painted
             pixel that did not change, or changed to a different block color, costs; a
             changed pixel nothing is painted on costs.
  search     Starting from the previous frame's board, each cell tries every alternative
             (empty; each color at each level) with the other cells held, keeping the cheapest.
             Two passes, since neighbours shade each other. A cell whose best alternative is not
             DECISION_MARGIN (per pixel of a block footprint) cheaper than the runner-up reads
             "unknown". So does a block whose visible top face (the part nearer stacks leave in
             view) is too small, mostly unchanged, or not clearly one color.
  stacks     Under the top, each block's visible side band is classified (bottom first).
  hands      Changed pixels that the final board does not explain, block-sized or bigger, are a
             hand or object (perception/blocks._leftovers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from perception.blocks import HAND_FACES, Region, _leftovers, crop_box, crop_pose, px_per_mm, sheet_roi
from perception.cells import cell_center_mm
from perception.change import change_mask, warp_matrix
from perception.colors import ColorRef, chroma, classify, label_pixels, sample_stats, specular, to_lab
from robot.perception import cubefit
from robot.perception.sheet import SheetPose
from shared.config import CELLS, COLORS, MAX_LEVEL, UNKNOWN

WORK_WIDTH = 480            # the sheet crop is read at this width (px)
DECISION_MARGIN = 0.12      # best alternative must beat the runner-up by this, per footprint pixel
MARGIN_FULL = 0.20          # ... and scores confidence 1.0 at this margin (clean blocks: 0.27-0.45)
OCCUPIED_SHARE = 0.5        # share of a block's visible top face that must have changed
MIN_TOP_PX = 40             # visible top-face pixels needed to read a block (work scale)
MIN_BAND_PX = 30            # visible side-band pixels needed to name a lower block (work scale)
BAND_MIN_MARGIN = 0.5
COST_WRONG_COLOR = 0.6      # painted block color, observed another block color
COST_COLORLESS = 0.5        # painted block, observed a changed but colorless pixel (sheen, shade)
SEARCH_PASSES = 2
FAR = len(COLORS)           # pixel label: changed and colorful, but no block color
NONE = -1                   # pixel label: not changed, or colorless

EMPTY = None                # a cell hypothesis: None, or (level, color index)


@dataclass
class Mask:
    """A boolean mask kept only inside its bounding box: rows y0:y0+h, cols x0:x0+w."""
    y0: int
    x0: int
    m: np.ndarray

    @classmethod
    def from_poly(cls, poly: np.ndarray, shape: tuple[int, int]) -> "Mask":
        poly = np.asarray(poly, np.float64).reshape(-1, 2)
        x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(poly.max(axis=0)).astype(int) + 1, [shape[1], shape[0]])
        m = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), np.uint8)
        if m.size:
            cv2.fillPoly(m, [(poly - [x0, y0]).astype(np.int32)], 1)
        return cls(int(y0), int(x0), m.astype(bool))

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.y0, self.y0 + self.m.shape[0], self.x0, self.x0 + self.m.shape[1]

    def into(self, win: tuple[int, int, int, int]):
        """(slices into a window array, the matching part of this mask), or None if disjoint."""
        wy0, wy1, wx0, wx1 = win
        y0, y1, x0, x1 = self.box
        a0, a1, b0, b1 = max(y0, wy0), min(y1, wy1), max(x0, wx0), min(x1, wx1)
        if a0 >= a1 or b0 >= b1:
            return None
        return (slice(a0 - wy0, a1 - wy0), slice(b0 - wx0, b1 - wx0)), self.m[a0 - y0:a1 - y0, b0 - x0:b1 - x0]

    def full(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, bool)
        y0, y1, x0, x1 = self.box
        out[y0:y1, x0:x1] = self.m
        return out


class Geometry:
    """Projected cells for one locked pose and block size, at work scale. Built once per lock."""

    def __init__(self, pose: SheetPose, shape: tuple[int, int], block_mm: float):
        self.shape, self.block_mm = shape, block_mm
        cam = pose.camera_xyz
        far_first = sorted(CELLS, key=lambda c: -np.hypot(*(np.asarray(cell_center_mm(c)) - cam[:2]))) \
            if cam is not None else list(CELLS)
        self.order = far_first                       # painter's order
        self.seg, self.top, self.col, self.top_poly = {}, {}, {}, {}
        for cell in CELLS:
            sq = cubefit.cube_corners(cell_center_mm(cell), 0.0, size_mm=block_mm)

            def hull(lo, hi):
                return cv2.convexHull(np.vstack([pose.to_image(sq, lo), pose.to_image(sq, hi)]).astype(np.float32))

            self.seg[cell] = [Mask.from_poly(hull(k * block_mm, (k + 1) * block_mm), shape) for k in range(MAX_LEVEL + 1)]
            self.col[cell] = [Mask.from_poly(hull(0.0, (k + 1) * block_mm), shape) for k in range(MAX_LEVEL + 1)]
            self.top_poly[cell] = [pose.to_image(sq, (k + 1) * block_mm) for k in range(MAX_LEVEL + 1)]
            self.top[cell] = [Mask.from_poly(p, shape) for p in self.top_poly[cell]]
        # Each cell's search window: everything its tallest column can touch.
        self.window = {c: self.col[c][MAX_LEVEL].box for c in CELLS}
        self.footprint_px = {c: max(int(self.col[c][0].m.sum()), 1) for c in CELLS}


def render(geo: Geometry, hyp: dict, win: tuple[int, int, int, int], colors: Optional[dict] = None):
    """Paint the board into a window: (label per pixel, NONE where nothing is painted; owner
    cell index per pixel, -1 where nothing). colors: cell -> per-level color index (stacks)."""
    shape = (win[1] - win[0], win[3] - win[2])
    label = np.full(shape, NONE, np.int16)
    owner = np.full(shape, -1, np.int8)
    for cell in geo.order:
        h = hyp.get(cell)
        if h is None:
            continue
        level, color = h
        per_level = colors.get(cell) if colors else None
        for k in range(level + 1):
            hit = geo.seg[cell][k].into(win)
            if hit is None:
                continue
            (sy, sx), m = hit
            label[sy, sx][m] = per_level[k] if per_level else color
            owner[sy, sx][m] = CELLS.index(cell)
    return label, owner


def cost(label: np.ndarray, changed: np.ndarray, observed: np.ndarray, roi: np.ndarray) -> float:
    painted = label != NONE
    c = np.where(~painted, changed, 0.0).astype(np.float32)
    on = painted & roi
    c[on & ~changed] = 1.0
    hit = on & changed
    c[hit & (observed == label)] = 0.0
    c[hit & (observed != label) & (observed >= 0) & (observed < FAR)] = COST_WRONG_COLOR
    c[hit & (observed == NONE)] = COST_COLORLESS
    c[hit & (observed == FAR)] = 1.0
    return float(c[roi].sum())


@dataclass
class CellRead:
    cell: str
    state: str                               # "empty", "block", "unknown"
    level: Optional[int] = None              # 0 = one block; set for blocks (and unknown blocks)
    color: str = UNKNOWN                     # top color
    sides: tuple[Optional[str], ...] = ()    # colors under the top, bottom first (None = unseen)
    confidence: float = 0.0
    reason: str = ""
    top_px: list = field(default_factory=list)  # outline(s) of the visible top face, full-frame px
    center_px: tuple[float, float] = (0.0, 0.0)


@dataclass
class CellAnalysis:
    cells: dict[str, CellRead]
    others: list[Region]
    hypothesis: dict                         # cell -> EMPTY or (level, color index): next frame's start
    render_bgr: np.ndarray                   # the final board painted in block colors, work scale
    to_full: tuple[float, int, int]          # (scale, x0, y0): work px -> full px = p / scale + (x0, y0)
    mask: Optional[np.ndarray] = None        # changed pixels, work scale
    live_sheet: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def empty(self) -> bool:
        return not self.others and all(r.state == "empty" for r in self.cells.values())

    def to_full_px(self, pts: np.ndarray) -> np.ndarray:
        s, x0, y0 = self.to_full
        return np.asarray(pts, np.float64).reshape(-1, 2) / s + [x0, y0]


_geo_cache: dict = {}


def geometry_for(pose: SheetPose, shape: tuple[int, int], block_mm: float) -> Geometry:
    key = (np.asarray(pose.H).tobytes(), shape, round(block_mm, 3),
           None if pose.camera_xyz is None else np.asarray(pose.camera_xyz).tobytes())
    if key not in _geo_cache:
        _geo_cache.clear()
        _geo_cache[key] = Geometry(pose, shape, block_mm)
    return _geo_cache[key]


def read_cells(frame_bgr: np.ndarray, pose: SheetPose, reference_sheet: np.ndarray, refs: dict[str, ColorRef],
               block_mm: float, prior: Optional[dict] = None, fg_chroma: float = 15.0,
               live_sheet: Optional[np.ndarray] = None) -> CellAnalysis:
    # Work space: the sheet crop at WORK_WIDTH, and the same pose for it.
    x0, y0, x1, y1 = crop_box(pose, frame_bgr.shape)
    crop = frame_bgr[y0:y1, x0:x1]
    scale = min(1.0, WORK_WIDTH / max(crop.shape[1], 1))
    work = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
    wpose = crop_pose(pose, x0, y0, scale)
    shape = work.shape[:2]
    geo = geometry_for(wpose, shape, block_mm)

    # Observed: change against the empty board seen from the camera, and a color label per pixel.
    inv = np.linalg.inv(warp_matrix(wpose.H))
    ref_img = cv2.warpPerspective(reference_sheet, inv, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(reference_sheet.shape[:2], 255, np.uint8), inv, (shape[1], shape[0]),
                                flags=cv2.INTER_NEAREST) > 0
    ref_img[~valid] = work[~valid]               # off the sheet: no reference, so no change
    changed = change_mask(work, ref_img) > 0
    roi = (sheet_roi(wpose, shape) > 0) & valid
    changed &= roi
    lab = to_lab(work)
    colorful = (chroma(lab) >= fg_chroma) & ~specular(lab) & changed
    observed = label_pixels(lab, colorful.astype(np.uint8), refs).astype(np.int16)
    active = [COLORS.index(c) for c in refs]

    # Search.
    hyp = {c: (prior or {}).get(c) for c in CELLS}
    hyp = {c: (h if h is None or h[1] in active else None) for c, h in hyp.items()}
    alternatives = [EMPTY] + [(lv, k) for lv in range(MAX_LEVEL + 1) for k in active]
    costs: dict = {}
    for _ in range(SEARCH_PASSES):
        for cell in CELLS:
            win = geo.window[cell]
            sl = (slice(win[0], win[1]), slice(win[2], win[3]))
            ch, ob, ro = changed[sl], observed[sl], roi[sl]
            trial = {}
            for alt in alternatives:
                hyp[cell] = alt
                trial[alt] = cost(render(geo, hyp, win)[0], ch, ob, ro)
            hyp[cell] = min(trial, key=trial.get)
            costs[cell] = trial

    # Final render with stack colors from the side bands, then read each cell.
    full = (0, shape[0], 0, shape[1])
    _, owner = render(geo, hyp, full)
    cells: dict[str, CellRead] = {}
    stack_colors: dict = {}
    for cell in CELLS:
        cells[cell], stack_colors[cell] = _read_cell(cell, geo, hyp, costs[cell], owner, changed, lab, observed,
                                                     refs, wpose, block_mm)
    label, owner = render(geo, hyp, full, stack_colors)
    by_blocks = owner >= 0
    explained = by_blocks.copy()
    for cell, r in cells.items():
        if r.state == "unknown":  # an unreadable block is not a hand: it reads "unknown" instead
            explained |= geo.col[cell][r.level or 0].full(shape)
    ppm = px_per_mm(wpose)
    face_px = (block_mm * ppm) ** 2
    s_to_full = (scale, x0, y0)
    found = _leftovers(changed.astype(np.uint8) * 255, explained.astype(np.uint8) * 255, face_px, ppm)
    # Too large: changed area no rendered block explains, bigger than any block, is a hand even
    # where the cells under it read unknown (a hand over the board makes them unreadable).
    rest = (changed & ~cv2.dilate(by_blocks.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).astype(np.uint8)
    for c in cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) > HAND_FACES * face_px:
            found.append(Region("hand", c.reshape(-1, 2), cv2.contourArea(c) / ppm ** 2, "changed area bigger than any block"))
    others = [Region(o.kind, np.asarray(o.contour_px, np.float64) / scale + [x0, y0], o.area_mm2, o.why)
              for o in found]
    for r in cells.values():
        r.top_px = [np.asarray(p, np.float64).reshape(-1, 2) / scale + [x0, y0] for p in r.top_px]
        r.center_px = (r.center_px[0] / scale + x0, r.center_px[1] / scale + y0)
    return CellAnalysis(cells, others, dict(hyp), _paint(label), s_to_full, changed, live_sheet)


def _read_cell(cell, geo: Geometry, hyp: dict, trial: dict, owner, changed, lab, observed, refs, wpose, block_mm):
    """The cell's reading from the search result and its visible top face; (CellRead, per-level colors)."""
    best = min(trial, key=trial.get)
    runner = min((v for k, v in trial.items() if k != best), default=trial[best] + 1e9)
    margin = (runner - trial[best]) / geo.footprint_px[cell]
    render_conf = max(0.0, min(1.0, margin / MARGIN_FULL))
    if margin < DECISION_MARGIN:
        level = None if best is None else best[0]
        return CellRead(cell, "unknown", level, reason=f"two readings fit almost equally (margin {margin:.2f})"), None
    if best is None:
        return CellRead(cell, "empty", confidence=render_conf), None
    level, color_idx = best
    idx = CELLS.index(cell)
    top = geo.top[cell][level]
    (sy, sx), m = top.into((0, geo.shape[0], 0, geo.shape[1]))
    visible = m & (owner[sy, sx] == idx)
    cx, cy = wpose.to_image([cell_center_mm(cell)], (level + 1) * block_mm)[0]
    outlines = [(c.reshape(-1, 2) + [top.x0, top.y0]) for c in
                cv2.findContours(visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
    base = dict(level=level, top_px=outlines, center_px=(float(cx), float(cy)))
    n_visible = int(visible.sum())
    if n_visible < MIN_TOP_PX:
        return CellRead(cell, "unknown", reason=f"top face hidden ({n_visible} px in view)", **base), None
    share = float(changed[sy, sx][visible].mean())
    if share < OCCUPIED_SHARE:
        return CellRead(cell, "unknown", reason=f"only {share:.0%} of the top face changed", **base), None
    px = lab[sy, sx][visible & changed[sy, sx]]
    try:
        median, _, _ = sample_stats(px)
        color, color_margin, _, _ = classify(median, refs)
    except ValueError:
        color, color_margin = UNKNOWN, 0.0
    if color != COLORS[color_idx]:
        return CellRead(cell, "unknown", reason=f"top reads {color}, the board fits {COLORS[color_idx]}", **base), None
    sides, per_level = [], []
    for k in range(level):  # each lower block's side band: its segment minus everything above it
        seg = geo.seg[cell][k]
        (by, bx), bm = seg.into((0, geo.shape[0], 0, geo.shape[1]))
        band = bm & (owner[by, bx] == idx) & changed[by, bx]
        for j in range(k + 1, level + 1):
            hit = geo.seg[cell][j].into(seg.box)
            if hit is not None:
                (jy, jx), jm = hit
                band[jy, jx] &= ~jm
        sides.append(_band_color(lab[by, bx][band], refs))
        per_level.append(COLORS.index(sides[-1]) if sides[-1] else color_idx)
    per_level.append(color_idx)
    return CellRead(cell, "block", color=color, sides=tuple(sides), confidence=min(render_conf, color_margin),
                    **base), per_level


def _band_color(px: np.ndarray, refs) -> Optional[str]:
    if len(px) < MIN_BAND_PX:
        return None
    try:
        median, _, _ = sample_stats(px)
    except ValueError:
        return None
    color, margin, _, _ = classify(median, refs)
    return None if color == UNKNOWN or margin < BAND_MIN_MARGIN else color


def _paint(label: np.ndarray) -> np.ndarray:
    """The rendered board in block colors (BGR), black where nothing is painted."""
    from perception.colors import DEFAULT_HEX
    lut = np.zeros((FAR + 2, 3), np.uint8)
    for i, c in enumerate(COLORS):
        h = DEFAULT_HEX.get(c, "#808080")
        lut[i] = (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))
    out = lut[np.where(label == NONE, FAR + 1, label)]
    out[label == NONE] = 0
    return out


# ---------------------------------------------------------------- temporal vote


@dataclass
class CellVote:
    """Each cell's published reading is the majority of its last `frames` readings, else unknown."""
    frames: int = 5
    history: dict = field(default_factory=dict)

    def add(self, cell: str, value) -> None:
        h = self.history.setdefault(cell, [])
        h.append(value)
        del h[:-self.frames]

    def winner(self, cell: str):
        """(value, True) for a strict majority of the window, else (None, False)."""
        h = self.history.get(cell, [])
        if not h:
            return None, False
        best = max(set(h), key=h.count)
        return (best, True) if h.count(best) * 2 > len(h) else (None, False)
