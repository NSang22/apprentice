"""Give each segment a colour name, learning the setup's colours as it goes.

Order of preference for every frame:
1. ColorMemory (what each colour looked like earlier in this setup): instant, offline.
2. The vision model (gemini_namer), only if some block is unfamiliar, ambiguous between two
   remembered colours, or two blocks would get the same name (tasks use all-different colours).
   Its answer is learned, so the next frames are local again.
3. PRIOR hue guess, if the model isn't available. Logged, because it can be wrong.
"""

import cv2
import numpy as np

from . import color, gemini_namer
from .config import ACTIVE_COLORS
from .segment import detection_roi, object_mask

PRIOR_PENALTY = 15.0  # a calibrated colour beats an uncalibrated prior guess at equal distance


class ColorNamer:
    def __init__(self, memory=None, use_model=True, log=print):
        self.memory = memory if memory is not None else color.ColorMemory()
        self.use_model = use_model
        self.log = log
        self.last_source = None

    def name(self, frame_bgr, contours, pose=None, tags=None):
        """Colour names (one per contour), e.g. ['red', 'light blue']."""
        if not len(contours):
            return []
        gains = None
        if pose is not None and tags is not None:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            roi = detection_roi(pose, frame_bgr.shape, tags)
            objects, _ = object_mask(frame_bgr, roi)
            paper = (roi > 0) & (objects == 0) & (hsv[..., 1] < 40) & (hsv[..., 2] > 120)
            gains = color.paper_gains(frame_bgr, paper.astype(np.uint8))
        sigs = [color.signature(frame_bgr, c, gains) for c in contours]
        names = [self.memory.classify(s, ACTIVE_COLORS)[0] for s in sigs]

        unsure = [i for i, n in enumerate(names) if n is None]
        clashes = {n for n in names if n is not None and names.count(n) > 1}
        if not unsure and not clashes:
            self.last_source = "memory"
            return names

        if self.use_model and gemini_namer.available():
            try:
                answer = {i: normalize(n) for i, n in gemini_namer.name_segments(frame_bgr, contours).items()}
                if len(answer) == len(contours):
                    for i, s in enumerate(sigs):
                        self.memory.learn(answer[i], s)
                    self.memory.save()
                    info = getattr(gemini_namer.name_segments, "last", {})
                    self.log(f"  (colours named by {info.get('model', 'the vision model')} in {info.get('seconds', '?')} s "
                             f"and remembered for this setup)")
                    self.last_source = "model"
                    return [answer[i] for i in range(len(contours))]
                self.log(f"  (vision model named only {len(answer)} of {len(contours)} blocks; using guesses)")
            except gemini_namer.NamerUnavailable as e:
                self.log(f"  ({e}; using colour guesses)")

        # No model: name the rest jointly, each name used once (blocks in a task are all different
        # colours). Candidates are the calibrated colours first (memory distance), then the rough
        # prior for colours never calibrated. Closest pairs claim their names first, so two
        # similar blocks split correctly: calibrated teal vs light blue in glare (distance 6,
        # "unsure" one at a time, 17:21), or orange-with-stars vs yellow (hue 97 vs 101, 17:01).
        guessed = [i for i, n in enumerate(names) if n is None or n in clashes]
        taken = {n for i, n in enumerate(names) if i not in guessed}
        pairs = []
        done = set()
        for i in guessed:
            if sigs[i]["chroma"] < color.LOW_CHROMA:
                names[i] = color.prior_name(sigs[i])  # greys: lightness decides, no hue to compare
                done.add(i)
                continue
            for n, samples in self.memory.colors.items():
                if samples and n in ACTIVE_COLORS:
                    pairs.append((min(color.distance(sigs[i], t) for t in samples), i, n))
            for n, h in color.PRIOR.items():
                if n not in self.memory.colors and n in ACTIVE_COLORS:
                    pairs.append((PRIOR_PENALTY + abs((sigs[i]["hue"] - h + 180) % 360 - 180), i, n))
        for _, i, n in sorted(pairs):
            if i not in done and n not in taken:
                names[i] = n
                done.add(i)
                taken.add(n)
        self.last_source = "prior" if guessed else "memory"
        if guessed and not getattr(self, "_warned_guess", False):
            self._warned_guess = True  # once per run: observe() names every photo
            self.log(f"  (guessed colours for {len(guessed)} block(s) without the vision model; may be wrong)")
        return names


def normalize(colour):
    """'Light-Blue ' -> 'light blue'."""
    return " ".join(colour.strip().lower().replace("-", " ").replace("_", " ").split())


def object_name(colour):
    return normalize(colour).replace(" ", "_") + "_cube"
