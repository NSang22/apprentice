"""Find objects on the sheet without fixed colour thresholds.

An object is anything on the sheet that is more colourful (Lab chroma) than the paper. The
paper's chroma is measured in every frame, so the threshold follows the room. Lab chroma, not
HSV saturation: HSV saturation inflates in shadows (shaded paper read S ~40 in the dim room,
as high as a washed-out cube top in the bright one), while paper chroma stays 0-3 everywhere
and blocks read 25-70. Touching objects are split by hue *relative to each other*. Names come
later, from scene naming (colour calibration / memory / vision model).

Same interface as detect_color.detect, so scene.build_scene can use either.
"""

import cv2
import numpy as np

from .config import BANK_ZONE_MM, CUBE_SIZE_MM
from .detect_color import Detection

MIN_DELTA_C = 12.0       # object pixels must beat the paper's chroma by at least this (Lab units)
MAD_K = 5.0              # ...or by this many median-absolute-deviations, whichever is larger
HUE_BIN = 3              # OpenCV hue units (0-179) per histogram bin
MIN_PEAK_FRACTION = 0.15  # without a known cube size: a hue peak must hold this share of a segment
MIN_PEAK_FACES = 0.35     # with one: a hue peak needs this many cube faces' worth of pixels. A share of
                          # the segment failed when glow linked three diagonal blocks into one blob and
                          # the washed-out yellow among them fell under 15% (snapshot 17:27).
MIN_PEAK_SEPARATION = 9   # hue units; a single cube's lit top and shaded side differ by ~5

MIN_AREA_FACES = 0.5     # smaller than half a cube face = fragment (cable, LED glint, tape sliver)
MERGED_FACES = 2.8       # one cube seen at an angle (top + two sides) reaches ~2.2 faces
SHEET_W_MM, SHEET_H_MM = 215.9, 279.4
CENTER_TOL_MM = 0.0      # an object's centre must lie on the paper (or in the cube bank)
BANK_PAD_MM = 30.0       # search this far around the bank zone, so blocks at its edge are seen whole
IGNORED_HUE = (40, 74)   # no green blocks in the kit: this is the green tape around the sheet and the bank.
                         # Measured 2026-10-11: tape 64-68, the teal (mint) block 80-84, light blue 97-105.
MIN_ASPECT = 0.45        # a block seen from above is squarish (top + one side ~0.6); thin strips are
                         # glare or the mat's sheen along the sheet edge (snapshot 16:55)

# Rough names from the hue of a segment's most saturated pixels (its sides, usually: tops wash
# out in bright light). Measured: orange 18 (bright); yellow 30 (bright), 23-24 (dim); blue 100-107;
# red 177-2. The orange/yellow margin is narrow -- vision-model naming (step 2) is the real fix.
HUE_NAMES = [(0, 7, "red"), (7, 21, "orange"), (21, 38, "yellow"), (38, 85, "green"),
             (85, 130, "blue"), (130, 165, "purple"), (165, 180, "red")]


def _hue_name(h):
    return next(n for lo, hi, n in HUE_NAMES if lo <= h < hi)


def _circ_dist(a, b):
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


def _hue_peaks(hues, min_count=None):
    """Peaks of a circular hue histogram that each hold enough pixels to be a block of their own
    (min_count pixels, or MIN_PEAK_FRACTION of the segment when no count is given)."""
    nb = 180 // HUE_BIN
    hist = np.bincount((hues // HUE_BIN).astype(int) % nb, minlength=nb).astype(float)
    smooth = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3
    peaks = [i for i in range(nb) if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[(i + 1) % nb]]
    peaks.sort(key=lambda i: -smooth[i])
    kept = []
    for i in peaks:
        center = i * HUE_BIN + HUE_BIN / 2
        mass = hist[[j % nb for j in range(i - 3, i + 4)]].sum()
        enough = mass >= min_count if min_count else mass / max(len(hues), 1) >= MIN_PEAK_FRACTION
        if enough and all(_circ_dist(center, k) >= MIN_PEAK_SEPARATION for k in kept):
            kept.append(center)
    return kept or [float(np.median(hues))]


def lab_chroma(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def object_mask(frame_bgr, roi_mask, chroma=None):
    """Pixels on the sheet more colourful than the paper. Returns (mask, threshold)."""
    c = lab_chroma(frame_bgr) if chroma is None else chroma
    sheet = c[roi_mask > 0]
    # The paper is the least colourful part of the sheet: use the lowest 30% of pixels. A plain
    # median assumed paper covers most of the sheet; with six blocks out (over half of it) the
    # old saturation threshold jumped to S>354 and nothing was detected (colour calibration 17:1x).
    paper = sheet[sheet <= np.percentile(sheet, 30)]
    med = float(np.median(paper))
    mad = float(np.median(np.abs(paper - med)))
    thr = med + max(MIN_DELTA_C, MAD_K * mad)
    mask = ((c > thr) & (roi_mask > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask, thr


CORE_CHROMA = 0.5   # a segment's core: pixels at least this colourful relative to its 90th percentile
KEEP_CHROMA = 0.3   # grow the core into connected pixels at least this colourful
# Under bright light a block tints the paper and its own shadow (Lab chroma 16-18 next to a
# yellow block whose sides are 62-68). That halo dragged a segment's centre and turned its
# measured yaw to -34 deg (snapshot 16:58). A washed-out top is still ~24, so relative
# thresholds with growth from the core keep the top and drop the halo.


def refine(mask, chroma):
    """Trim the colour halo off one segment's mask, keeping the block body."""
    c = chroma[mask > 0]
    if len(c) == 0:
        return mask
    c90 = np.percentile(c, 90)
    core = (mask > 0) & (chroma >= CORE_CHROMA * c90)
    keep = ((mask > 0) & (chroma >= KEEP_CHROMA * c90)).astype(np.uint8)
    n, labels = cv2.connectedComponents(keep)
    hit = np.unique(labels[core])
    out = np.isin(labels, hit[hit > 0]).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def in_bank(xy, zone=None):
    """True if a block centred at xy (sheet mm) sits in the cube bank beside the sheet (or in
    `zone`, (x0, y0, x1, y1) sheet mm, instead)."""
    zone = BANK_ZONE_MM if zone is None else zone
    if zone is None:
        return False
    x0, y0, x1, y1 = zone
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def detection_roi(pose, shape, spec_tags, margin_mm=5.0, corner_pad_mm=6.0, bank_zone=None):
    """Sheet (plus a small margin) and the cube bank, minus the corner squares holding the tags:
    objects must not cover the tags anyway, and that is where the tape holding the sheet down sits."""
    m = margin_mm
    roi = np.zeros(shape[:2], np.uint8)
    poly = pose.to_image([(-m, -m), (SHEET_W_MM + m, -m), (SHEET_W_MM + m, SHEET_H_MM + m), (-m, SHEET_H_MM + m)])
    cv2.fillPoly(roi, [poly.astype(np.int32)], 255)
    bank_zone = BANK_ZONE_MM if bank_zone is None else bank_zone
    if bank_zone is not None:
        x0, y0, x1, y1 = bank_zone
        p = BANK_PAD_MM
        bank = pose.to_image([(x0 - p, y0 - p), (x1 + p, y0 - p), (x1 + p, y1 + p), (x0 - p, y1 + p)])
        cv2.fillPoly(roi, [bank.astype(np.int32)], 255)
    for corners in spec_tags.values():
        c = np.asarray(corners, float)
        x0, y0 = c.min(0) - corner_pad_mm
        x1, y1 = c.max(0) + corner_pad_mm
        # extend each tag box out to its nearest sheet corner (and past the margin)
        x0, x1 = (-m - 5, x1) if x0 < SHEET_W_MM / 2 else (x0, SHEET_W_MM + m + 5)
        y0, y1 = (-m - 5, y1) if y0 < SHEET_H_MM / 2 else (y0, SHEET_H_MM + m + 5)
        box = pose.to_image([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        cv2.fillPoly(roi, [box.astype(np.int32)], 0)
    return roi


def detect(frame_bgr, names=None, roi_mask=None, pose=None, tags=None, bank_zone=None):
    """Segments on the sheet, named '<colour>_cube' (or '_cube_2' etc. for repeats).

    With `pose` (and the sheet's tag corners in `tags`) it restricts itself to the sheet minus the
    tag corners, filters by size in mm, and drops anything centred off the sheet.
    A segment much bigger than one cube is marked merged (score < 0) so callers can refuse it.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    px_per_mm = None
    if pose is not None:
        px_per_mm = np.linalg.norm(pose.to_image([(100, 100)])[0] - pose.to_image([(110, 100)])[0]) / 10
        if tags is not None:
            roi_mask = detection_roi(pose, frame_bgr.shape, tags, bank_zone=bank_zone)
    roi = roi_mask if roi_mask is not None else np.full(hsv.shape[:2], 255, np.uint8)
    face_px = (CUBE_SIZE_MM * px_per_mm) ** 2 if (px_per_mm and CUBE_SIZE_MM) else None
    min_area_px = MIN_AREA_FACES * face_px if face_px else 3000
    chroma = lab_chroma(frame_bgr)
    mask, _ = object_mask(frame_bgr, roi, chroma)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    pieces = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        comp = labels == i
        hues = hsv[..., 0][comp].astype(float)
        peaks = _hue_peaks(hues, MIN_PEAK_FACES * face_px if face_px else None)
        if len(peaks) == 1:
            pieces.append(comp.astype(np.uint8) * 255)
            continue
        # Touching objects of different colours: give each pixel to its nearest hue peak.
        d = np.stack([_circ_dist(hsv[..., 0].astype(float), p) for p in peaks])
        owner = np.argmin(d, axis=0)
        for j, p in enumerate(peaks):
            sub = (comp & (owner == j)).astype(np.uint8) * 255
            sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            m2, l2, s2, _ = cv2.connectedComponentsWithStats(sub)
            for k in range(1, m2):
                if s2[k, cv2.CC_STAT_AREA] >= min_area_px:
                    pieces.append((l2 == k).astype(np.uint8) * 255)

    pieces = [refine(m, chroma) for m in pieces]
    pieces = [m for m in pieces if cv2.countNonZero(m) >= min_area_px]

    out, seen = [], {}
    for m in sorted(pieces, key=lambda t: -cv2.countNonZero(t)):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        mo = cv2.moments(c)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < MIN_ASPECT * max(rw, rh):
            continue
        x, y, w, h = cv2.boundingRect(c)
        center = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
        if pose is not None:
            cx, cy = pose.to_sheet([center])[0]
            # The bank test uses the cube's mid-height point, like the scene's centre_mm (and
            # BANK_ZONE_MM): the table-plane point sits up to 12 mm further out at the bank's edge.
            mid = pose.to_sheet([center], CUBE_SIZE_MM / 2 if CUBE_SIZE_MM else 0.0)[0]
            if not (-CENTER_TOL_MM <= cx <= SHEET_W_MM + CENTER_TOL_MM and -CENTER_TOL_MM <= cy <= SHEET_H_MM + CENTER_TOL_MM
                    or in_bank(mid, bank_zone)):
                continue
        sat = hsv[..., 1][m > 0]
        hue = float(np.median(hsv[..., 0][m > 0][sat >= np.percentile(sat, 75)]))
        if IGNORED_HUE[0] <= hue <= IGNORED_HUE[1]:
            continue
        color = _hue_name(hue)
        seen[color] = seen.get(color, 0) + 1
        name = f"{color}_cube" + ("" if seen[color] == 1 else f"_{seen[color]}")
        score = float(area / max(rw * rh, 1.0))
        if face_px and area > MERGED_FACES * face_px:
            score = -score
        out.append(Detection(name=name, bbox_px=(x, y, x + w, y + h), center_px=center,
                             score=score, contour=c.reshape(-1, 2)))
    if names:
        out = [d for d in out if d.name in names]
    return out
