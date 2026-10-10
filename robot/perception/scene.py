"""Frame -> scene: object positions in sheet millimetres."""

import time

import cv2
import numpy as np

from . import detect_color
from .config import CUBE_SIZE_MM

SHEET_W_MM, SHEET_H_MM = 215.9, 279.4  # US Letter
ROI_MARGIN_MM = 40.0                   # objects may hang a little off the sheet


def sheet_roi_mask(pose, shape, margin_mm=ROI_MARGIN_MM):
    m = margin_mm
    poly = pose.to_image([(-m, -m), (SHEET_W_MM + m, -m), (SHEET_W_MM + m, SHEET_H_MM + m), (-m, SHEET_H_MM + m)])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def _yaw_deg(points_mm):
    """Cube yaw in (-45, 45]: counterclockwise seen from above is positive."""
    (_, _), (_, _), _ = rect = cv2.minAreaRect(points_mm.astype(np.float32))
    box = cv2.boxPoints(rect)
    dx, dy = box[1] - box[0]
    # Sheet y points down the page, so a visually counterclockwise turn is a negative atan2.
    yaw = -np.degrees(np.arctan2(dy, dx))
    return float((yaw + 45) % 90 - 45)


def build_scene(frame_bgr, localizer, detector=detect_color.detect, names=None):
    """Returns (scene dict, pose, detections). scene is None when the sheet isn't visible."""
    pose = localizer.locate(frame_bgr)
    if pose is None:
        return None, None, []
    detections = detector(frame_bgr, names, roi_mask=sheet_roi_mask(pose, frame_bgr.shape))
    # A cube's blob is its top face plus visible sides; its centroid sits near the cube's
    # mid-height point, so map it at half the cube height.
    mid_h = CUBE_SIZE_MM / 2 if CUBE_SIZE_MM else 0.0
    objects = []
    for d in detections:
        center = pose.to_sheet([d.center_px], mid_h)[0]
        outline = pose.to_sheet(d.contour, mid_h)
        objects.append({
            "name": d.name,
            "center_mm": [round(float(center[0]), 1), round(float(center[1]), 1)],
            "yaw_deg": round(_yaw_deg(outline), 1),
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
    colors = {"red_cube": (0, 0, 255), "blue_cube": (255, 0, 0), "yellow_cube": (0, 220, 255)}
    for d, o in zip(detections, scene["objects"]):
        col = colors.get(d.name, (255, 255, 255))
        cv2.drawContours(out, [d.contour.reshape(-1, 1, 2)], -1, col, 3)
        u, v = map(int, d.center_px)
        cv2.drawMarker(out, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        label = f"{d.name} ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm  yaw {o['yaw_deg']:.0f}"
        cv2.putText(out, label, (u - 160, v - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(out, label, (u - 160, v - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out
