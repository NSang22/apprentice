"""Colour identity for segments, learned per session instead of hardcoded per room.

Each segment gets a Lab signature from its most colourful pixels, after white-balancing the
frame against the paper. Measured on 2026-10-10 in a dim and a bright room, the Lab *hue angle*
of a block barely moves with lighting (red 23-30 deg, yellow 102-105) while lightness and
chroma do, so matching leans on hue.

ColorMemory remembers what each colour name looked like in the current setup. Names come from
a vision model (gemini_namer) when available; PRIOR is only a rough last-resort guess.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import BLOCK_COLORS, ROOT

MEMORY_FILE = ROOT / "outputs" / "color_memory.json"
TOP_CHROMA_FRACTION = 0.3  # use the most colourful 30% of a segment's pixels (sides beat washed-out tops)

# Distance between signatures: degrees of hue, plus looser terms for chroma and lightness,
# which shift more with lighting than hue does.
W_HUE, W_CHROMA, W_L = 1.0, 0.4, 0.2
KNOWN_MAX = 18.0     # nearest remembered colour must be at least this close...
MIN_MARGIN = 8.0     # ...and this much closer than the second-nearest, or we ask the namer
LOW_CHROMA = 12.0    # below this a segment is essentially grey; hue is meaningless
MAX_SAMPLES = 8      # remembered signatures per colour

# Rough Lab hue-angle centres. Only used when there is no memory and no vision model, and only
# for colours the kit has (config.BLOCK_COLORS), so a guess can't name a block the kit lacks.
# "light blue" is this kit's block (measured 253 in the bright room, 265-272 in the dim one),
# which sits where a generic blue would; the kit has no separate blue.
HUE_CENTERS = {"red": 27, "orange": 80, "yellow": 104, "green": 150, "teal": 195,
               "light blue": 250, "blue": 268, "purple": 310, "pink": 352}
PRIOR = {n: h for n, h in HUE_CENTERS.items() if n in BLOCK_COLORS} or HUE_CENTERS


def paper_gains(frame_bgr, paper_mask):
    """Per-channel gains that make the paper neutral (1.0 when it's clipped white or unknown)."""
    px = frame_bgr[paper_mask > 0]
    if len(px) < 500:
        return np.ones(3)
    p = np.median(px.reshape(-1, 3).astype(float), axis=0)
    return p.max() / np.maximum(p, 1.0)


def signature(frame_bgr, contour, gains=None):
    m = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(contour, np.int32).reshape(-1, 1, 2)], 255)
    px = frame_bgr[m > 0].astype(float)
    if gains is not None:
        px = np.clip(px * gains, 0, 255)
    lab = cv2.cvtColor(px.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    L, a, b = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128
    chroma = np.hypot(a, b)
    top = chroma >= np.percentile(chroma, 100 * (1 - TOP_CHROMA_FRACTION))
    A, B = float(np.median(a[top])), float(np.median(b[top]))
    sig = {"L": round(float(np.median(L[top])), 1), "a": round(A, 1), "b": round(B, 1),
           "hue": round(float(np.degrees(np.arctan2(B, A)) % 360), 1), "chroma": round(float(np.hypot(A, B)), 1)}
    # Hue quartiles over all the block's colourful pixels. Patterned blocks mix colours: an
    # orange block with yellow stars has hue quartiles 81/93/100 vs a plain yellow block's
    # 99/105/106, while its most chromatic pixels (the stars) read almost yellow (17:01 snapshot).
    colorful = chroma >= max(LOW_CHROMA, 0.3 * float(np.percentile(chroma, 90)))
    if colorful.sum() >= 50:
        hue = np.degrees(np.arctan2(b[colorful], a[colorful]))
        center = np.degrees(np.arctan2(np.sin(np.radians(hue)).mean(), np.cos(np.radians(hue)).mean()))
        rel = (hue - center + 180) % 360 - 180  # unwrap around the block's mean hue (red spans 0/360)
        sig["hue_q"] = [round(float((center + q) % 360), 1) for q in np.percentile(rel, [25, 50, 75])]
    return sig


def _dhue(x, y):
    return abs((x - y + 180) % 360 - 180)


def distance(s, t):
    if s["chroma"] < LOW_CHROMA or t["chroma"] < LOW_CHROMA:
        dh = 0.0 if (s["chroma"] < LOW_CHROMA) == (t["chroma"] < LOW_CHROMA) else 90.0
    elif "hue_q" in s and "hue_q" in t:
        dh = float(np.mean([_dhue(x, y) for x, y in zip(s["hue_q"], t["hue_q"])]))
    else:
        dh = _dhue(s["hue"], t["hue"])
    return W_HUE * dh + W_CHROMA * abs(s["chroma"] - t["chroma"]) + W_L * abs(s["L"] - t["L"])


def prior_name(sig):
    if sig["chroma"] < LOW_CHROMA:
        return "white" if sig["L"] > 75 else ("black" if sig["L"] < 30 else "grey")
    return min(PRIOR, key=lambda n: abs((sig["hue"] - PRIOR[n] + 180) % 360 - 180))


class ColorMemory:
    """What each colour name looks like in this setup. Persisted so later runs in the same room
    don't need the vision model; entries from another room simply won't match and get relearned."""

    def __init__(self, path=MEMORY_FILE):
        self.path = Path(path)
        self.colors = {}
        if self.path.exists():
            try:
                self.colors = json.loads(self.path.read_text()).get("colors", {})
            except (ValueError, OSError):
                self.colors = {}

    def classify(self, sig):
        """(name, distance, margin) of the nearest remembered colour; name None if unsure."""
        scored = sorted((min(distance(sig, s) for s in samples), name) for name, samples in self.colors.items() if samples)
        if not scored:
            return None, None, None
        best_d, best = scored[0]
        margin = (scored[1][0] - best_d) if len(scored) > 1 else float("inf")
        if best_d > KNOWN_MAX or margin < MIN_MARGIN:
            return None, best_d, margin
        return best, best_d, margin

    def learn(self, name, sig):
        samples = self.colors.setdefault(name, [])
        samples.append(sig)
        del samples[:-MAX_SAMPLES]

    def forget_all(self):
        self.colors = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"updated": time.time(), "colors": self.colors}, indent=1))
