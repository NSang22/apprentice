"""Find the foam cubes by colour. Offline fallback for the Gemini detector; same interface."""

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV HSV: H 0-179, S 0-255, V 0-255. Red wraps around 0, so it gets two ranges.
HSV_RANGES = {
    # Saturation floors sit well above the white paper (S 0-10) but below washed-out cube tops.
    # Dim room (2026-10-10 night): blue top S ~110, yellow ~134, red ~166.
    # Bright room (new location, 15:30): blue top S ~62 (H ~90), yellow top S ~50, red ~161.
    # An orange cube (top H ~24) overlaps yellow's hue in bright light; keep it off the sheet.
    "red_cube": [((0, 100, 70), (8, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "blue_cube": [((88, 35, 60), (125, 255, 255))],
    "yellow_cube": [((22, 35, 110), (35, 255, 255))],
}


@dataclass
class Detection:
    name: str
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1
    center_px: tuple[float, float]      # blob centroid
    score: float                        # fraction of the min-area rect the blob fills
    contour: np.ndarray                 # (N, 2) pixel outline


def color_mask(frame_bgr, name, roi_mask=None):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[name]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    if roi_mask is not None:
        mask &= roi_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drops thin wires and specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fills the raised letters
    return mask


def detect(frame_bgr, names=None, roi_mask=None, min_area_px=2000) -> list[Detection]:
    """Largest blob per colour. One cube per colour is assumed."""
    out = []
    for name in names or list(HSV_RANGES):
        mask = color_mask(frame_bgr, name, roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        out.append(Detection(
            name=name,
            bbox_px=(x, y, x + w, y + h),
            center_px=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
            score=float(cv2.contourArea(c) / max(rw * rh, 1.0)),
            contour=c.reshape(-1, 2),
        ))
    return out
