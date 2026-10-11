"""Block colors in CIELAB: calibration data, sampling, and nearest-color classification.

perception.calibrate measures one block of each color on the real sheet and saves, per color,
the median Lab of its top face and the spread of its pixels around that median, plus the
block's footprint as a fraction of a cell. perception.publish then labels every block by the
nearest reference color. Pure functions only (numpy in, numbers out), so they are testable
without a camera.

Lab here is OpenCV's float Lab: L 0-100, a and b about -128..127. Distances are Euclidean
with lightness counted at half weight (the 2:1 lightness tolerance of CMC l:c): a block's
shaded side is mostly darker, not a different hue, so shading must not flip orange to red.
About 3 is barely visible, 10+ is an obvious color difference.

Silicone blocks have sheen. A highlight is very bright and nearly colorless (the light's
color, not the block's), so sampling drops pixels with L > SPECULAR_L and chroma below
SPECULAR_CHROMA before taking medians.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.config import COLORS, UNKNOWN

CALIB_PATH = Path(__file__).resolve().parent / "calib" / "colors.json"

SPECULAR_L = 80.0          # highlights are brighter than this ...
SPECULAR_CHROMA = 20.0     # ... and less colorful than this
MAX_DISTANCE_FLOOR = 18.0  # a blob farther than this (or SPREAD_K x the color's spread) from
SPREAD_K = 3.0             # its nearest reference color is "unknown"
L_WEIGHT = 0.5             # lightness counts half in color distance (see module docstring)
MARGIN_FULL = 10.0         # second-nearest this much farther than the nearest: margin score 1.0
FG_CHROMA_FLOOR = 10.0     # foreground chroma threshold never drops below this (paper sits ~0-8)
FG_CHROMA_CAP = 25.0       # ... and never rises above this
FG_CHROMA_OF_MIN = 0.5     # default: half the least colorful block's chroma


@dataclass(frozen=True)
class ColorRef:
    name: str
    lab: tuple[float, float, float]   # median Lab of the block's top face
    spread: float                     # 90th percentile delta E of its pixels from that median
    hex: str = "#808080"              # sRGB of the median, for the UI

    def cutoff(self) -> float:
        return max(MAX_DISTANCE_FLOOR, SPREAD_K * self.spread)


@dataclass
class Calibration:
    refs: dict[str, ColorRef]
    area_frac: tuple[float, float, float] = (0.45, 0.65, 1.25)  # block footprint / cell area: min, median, max
    side_mm: float = 48.0                                       # block edge, from the top-face footprint
    fg_chroma: float = 15.0                                     # min chroma of a block pixel
    source: str = "default"                                     # "default" or the calib file path
    meta: dict = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.source != "default"


# ---------------------------------------------------------------- Lab helpers


def to_lab(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> float32 Lab (L 0-100)."""
    return cv2.cvtColor(frame_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def chroma(lab: np.ndarray) -> np.ndarray:
    return np.hypot(lab[..., 1], lab[..., 2])


def specular(lab: np.ndarray) -> np.ndarray:
    """True where a pixel looks like a highlight: very bright and nearly colorless."""
    return (lab[..., 0] > SPECULAR_L) & (chroma(lab) < SPECULAR_CHROMA)


def lab_to_hex(lab: tuple[float, float, float] | np.ndarray) -> str:
    px = np.asarray(lab, np.float32).reshape(1, 1, 3)
    bgr = np.clip(cv2.cvtColor(px, cv2.COLOR_Lab2BGR)[0, 0] * 255.0 + 0.5, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])


def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    lab = to_lab(np.array([[[b, g, r]]], np.uint8))[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def sample_stats(lab_pixels: np.ndarray) -> tuple[tuple[float, float, float], float, int]:
    """Median Lab and spread of a block's pixels (N, 3), highlights dropped first.
    Returns (median, spread, pixels used)."""
    px = np.asarray(lab_pixels, np.float32).reshape(-1, 3)
    px = px[~specular(px)]
    if len(px) == 0:
        raise ValueError("no usable pixels (all highlight)")
    median = np.median(px, axis=0)
    spread = float(np.percentile(delta_e(px, median), 90))
    return (float(median[0]), float(median[1]), float(median[2])), spread, int(len(px))


# ---------------------------------------------------------------- classification


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Color distance over the last axis, lightness at L_WEIGHT."""
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    return np.sqrt((L_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def distances(lab: tuple[float, float, float] | np.ndarray, refs: dict[str, ColorRef]) -> list[tuple[float, str]]:
    """(delta E, color) to every reference, nearest first."""
    return sorted((float(delta_e(np.asarray(lab), np.asarray(r.lab))), name) for name, r in refs.items())


def margin_score(d_nearest: float, d_second: float, full: float = MARGIN_FULL) -> float:
    """1.0 when the second-nearest color is at least `full` delta E farther than the nearest,
    falling to 0.0 when the two are equally close (teal vs skyblue, orange vs red/yellow)."""
    return max(0.0, min(1.0, (d_second - d_nearest) / full))


def classify(lab: tuple[float, float, float] | np.ndarray, refs: dict[str, ColorRef]) -> tuple[str, float, float, float]:
    """(color or "unknown", margin score, nearest delta E, second-nearest delta E).
    Too far from every reference is "unknown" with margin 0."""
    ranked = distances(lab, refs)
    (d1, name), d2 = ranked[0], (ranked[1][0] if len(ranked) > 1 else float("inf"))
    if d1 > refs[name].cutoff():
        return UNKNOWN, 0.0, d1, d2
    return name, margin_score(d1, d2), d1, d2


def label_pixels(lab: np.ndarray, mask: np.ndarray, refs: dict[str, ColorRef]) -> np.ndarray:
    """Per pixel: index into COLORS of the nearest reference, len(COLORS) for a colorful pixel
    far from every reference, -1 outside `mask`. Splits touching blocks of different colors."""
    labels = np.full(mask.shape, -1, np.int16)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return labels
    px = lab[ys, xs].astype(np.float32)
    names = [c for c in COLORS if c in refs]
    centers = np.array([refs[c].lab for c in names], np.float32)
    d = delta_e(px[:, None, :], centers[None, :, :])
    nearest = d.argmin(axis=1)
    far = d[np.arange(len(px)), nearest] > np.array([refs[c].cutoff() for c in names])[nearest] * 2.0
    idx = np.array([COLORS.index(c) for c in names], np.int16)[nearest]
    idx[far] = len(COLORS)
    labels[ys, xs] = idx
    return labels


def foreground(lab: np.ndarray, roi: np.ndarray, fg_chroma: float) -> np.ndarray:
    """uint8 mask of colorful, non-highlight pixels inside roi (paper, lines, tags, shadows are out)."""
    fg = (chroma(lab) >= fg_chroma) & ~specular(lab) & (roi > 0)
    return fg.astype(np.uint8) * 255


def fg_chroma_for(refs: dict[str, ColorRef]) -> float:
    least = min(float(np.hypot(r.lab[1], r.lab[2])) for r in refs.values())
    return max(FG_CHROMA_FLOOR, min(FG_CHROMA_CAP, FG_CHROMA_OF_MIN * least))


# ---------------------------------------------------------------- calibration file

# Nominal Playskool colors, used only until perception.calibrate has run (publish warns).
DEFAULT_HEX = {"red": "#e2383f", "orange": "#f5862a", "yellow": "#f6c933",
               "teal": "#16a59b", "skyblue": "#62bdea", "purple": "#8a5cc7"}


def default_calibration() -> Calibration:
    refs = {c: ColorRef(c, hex_to_lab(h), 6.0, h) for c, h in DEFAULT_HEX.items()}
    return Calibration(refs=refs, fg_chroma=fg_chroma_for(refs))


def area_stats(fracs: list[float]) -> tuple[float, float, float]:
    """(min, median, max) footprint fractions with headroom: a block lands anywhere in a cell
    and shows more side face away from the camera's nadir than the calibration spots did."""
    f = np.asarray(fracs, np.float64)
    return float(f.min() * 0.7), float(np.median(f)), float(f.max() * 1.3)


def save_calibration(path: Path, refs: dict[str, ColorRef], fracs: list[float], cell_mm: float,
                     meta: Optional[dict] = None) -> dict:
    lo, med, hi = area_stats(fracs)
    data = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "colors": {c: {"lab": [round(v, 2) for v in r.lab], "spread": round(r.spread, 2), "hex": r.hex}
                   for c, r in refs.items()},
        "block": {"area_frac": [round(lo, 4), round(med, 4), round(hi, 4)],
                  "side_mm": round(float(np.sqrt(min(fracs))) * cell_mm, 1)},
        "fg_chroma": round(fg_chroma_for(refs), 2),
        **({"meta": meta} if meta else {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
    return data


def load_calibration(path: Path = CALIB_PATH) -> Calibration:
    """The saved calibration, or the nominal defaults if there is no file yet."""
    if not path.exists():
        return default_calibration()
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = {c: ColorRef(c, tuple(float(v) for v in d["lab"]), float(d["spread"]), d.get("hex", "#808080"))
            for c, d in data["colors"].items() if c in COLORS}
    missing = [c for c in COLORS if c not in refs]
    if missing:
        raise ValueError(f"{path}: no calibration for {missing}; rerun perception.calibrate")
    block = data.get("block", {})
    return Calibration(
        refs=refs,
        area_frac=tuple(block.get("area_frac", (0.45, 0.65, 1.25))),
        side_mm=float(block.get("side_mm", 48.0)),
        fg_chroma=float(data.get("fg_chroma", fg_chroma_for(refs))),
        source=str(path),
        meta=data.get("meta", {}),
    )
