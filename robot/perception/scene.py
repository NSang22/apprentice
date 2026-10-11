"""Frame -> scene: object positions in sheet millimetres."""

import time

import cv2
import numpy as np

from . import cubefit, detect_color, segment
from . import naming as naming_mod
from .config import CUBE_SIZE_MM

SHEET_W_MM, SHEET_H_MM = 215.9, 279.4  # US Letter
ROI_MARGIN_MM = 40.0                   # objects may hang a little off the sheet


OUTLINE_COLORS = {"red_cube": (0, 0, 255), "orange_cube": (0, 140, 255), "yellow_cube": (0, 220, 255),
                  "green_cube": (0, 180, 0), "teal_cube": (160, 160, 0), "light_blue_cube": (255, 200, 80),
                  "blue_cube": (255, 0, 0), "purple_cube": (200, 0, 140), "pink_cube": (180, 120, 255)}


def sheet_roi_mask(pose, shape, margin_mm=ROI_MARGIN_MM):
    m = margin_mm
    poly = pose.to_image([(-m, -m), (SHEET_W_MM + m, -m), (SHEET_W_MM + m, SHEET_H_MM + m), (-m, SHEET_H_MM + m)])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


YAW_FIT_MIN_OVERLAP = 0.85  # cube-outline fit good enough to trust its yaw (clean blobs score 0.87-0.92)


def _yaw_deg(points_mm):
    """Cube yaw in (-45, 45]: counterclockwise seen from above is positive."""
    (_, _), (_, _), _ = rect = cv2.minAreaRect(points_mm.astype(np.float32))
    box = cv2.boxPoints(rect)
    dx, dy = box[1] - box[0]
    # Sheet y points down the page, so a visually counterclockwise turn is a negative atan2.
    yaw = -np.degrees(np.arctan2(dy, dx))
    return float((yaw + 45) % 90 - 45)


YAW_AGREE_DEG = 12.0  # a rejected cube fit is kept if the blob's box angle agrees this closely


def _same_yaw(a, b):
    """Yaws agree, allowing for the 90 deg wrap of a square."""
    d = abs(a - b) % 90
    return min(d, 90 - d) <= YAW_AGREE_DEG


_namers = {}


def _namer(naming):
    """One ColorNamer per mode per process, sharing the session's colour memory."""
    if naming not in _namers:
        shared = next(iter(_namers.values())).memory if _namers else None
        quiet = (lambda m: None) if naming == "local" else print  # local naming runs once per saved photo
        _namers[naming] = naming_mod.ColorNamer(memory=shared, use_model=(naming == "auto"), log=quiet)
    return _namers[naming]


def build_scene(frame_bgr, localizer, detector=segment.detect, names=None, naming="auto", bank_zone=None):
    """Returns (scene dict, pose, detections). scene is None when the sheet isn't visible.

    naming: "auto" names blocks from the session's colour memory and asks the vision model when
    unsure; "local" never calls the model (memory, then a rough guess); "off" keeps the
    segmenter's own hue labels.
    """
    pose = localizer.locate(frame_bgr)
    if pose is None:
        return None, None, []
    if detector is segment.detect:
        tags = {k: v["corners_mm"] for k, v in localizer.spec["tags"].items()}
        extra = {} if bank_zone is None else {"bank_zone": bank_zone}  # e.g. a wider look for a missing block
        detections = detector(frame_bgr, None, pose=pose, tags=tags, **extra)
        if naming != "off" and detections:
            colours = _namer(naming).name(frame_bgr, [d.contour for d in detections], pose, tags)
            seen = {}
            for d, c in zip(detections, colours):
                base = naming_mod.object_name(c)
                seen[base] = seen.get(base, 0) + 1
                d.name = base if seen[base] == 1 else f"{base}_{seen[base]}"
        if names:
            detections = [d for d in detections if d.name in names]
    else:  # colour-threshold fallback
        detections = detector(frame_bgr, names, roi_mask=sheet_roi_mask(pose, frame_bgr.shape))
    # A cube's blob is its top face plus visible sides; its centroid sits near the cube's
    # mid-height point, so map it at half the cube height.
    mid_h = CUBE_SIZE_MM / 2 if CUBE_SIZE_MM else 0.0
    objects = []
    for d in detections:
        center = pose.to_sheet([d.center_px], mid_h)[0]
        outline = pose.to_sheet(d.contour, mid_h)
        yaw = _yaw_deg(outline)
        if pose.camera_xyz is not None and getattr(d, "contour", None) is not None and len(d.contour) >= 3:
            # The blob's box rotation is unreliable (sides in view, glare-cut corners): fit a real
            # cube outline, and if even that fits poorly, assume the cube sits square to the sheet.
            _, fit_yaw, overlap = cubefit.fit_cube(pose, d.contour, center, yaw)
            if overlap >= YAW_FIT_MIN_OVERLAP or _same_yaw(fit_yaw, yaw):
                yaw = fit_yaw  # a weaker fit still counts when the blob's own box turns the same way
            else:  # (bank blocks at the image edge fit 0.81-0.84 while turned 20 deg, 2026-10-11)
                yaw = 0.0
        objects.append({
            "name": d.name,
            "center_mm": [round(float(center[0]), 1), round(float(center[1]), 1)],
            "yaw_deg": round(yaw, 1),
            "height_mm": CUBE_SIZE_MM,
            "score": round(d.score, 2),
        })
    scene = {
        "timestamp": time.time(),
        "sheet": {"tags": pose.tag_ids, "fit_error_mm": round(pose.fit_error_mm, 2),
                  "camera_xyz_mm": None if pose.camera_xyz is None else [round(float(v), 1) for v in pose.camera_xyz]},
        "objects": objects,
    }
    return scene, pose, detections


def annotate(frame_bgr, scene, pose, detections):
    out = frame_bgr.copy()
    if pose is None:
        cv2.putText(out, "SHEET NOT FOUND (need 2+ tags)", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        return out
    corners = pose.to_image([(0, 0), (SHEET_W_MM, 0), (SHEET_W_MM, SHEET_H_MM), (0, SHEET_H_MM)])
    cv2.polylines(out, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    colors = OUTLINE_COLORS
    for d, o in zip(detections, scene["objects"]):
        col = colors.get(d.name, (255, 255, 255))
        cv2.drawContours(out, [d.contour.reshape(-1, 1, 2)], -1, col, 3)
        u, v = map(int, d.center_px)
        cv2.drawMarker(out, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        label = f"{d.name} ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm  yaw {o['yaw_deg']:.0f}"
        cv2.putText(out, label, (u - 160, v - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(out, label, (u - 160, v - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out
