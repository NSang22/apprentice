"""Locate the AprilTag sheet in a frame and convert between pixels and sheet millimetres.

Sheet frame (see workspace_sheet.json): origin at the paper's top-left corner, x right,
y down the page, in mm. Heights are mm above the table.

A pixel maps to the point where its viewing ray meets the table (homography). For a point
at height h (e.g. a cube's top face) the ray meets that height closer to the camera's
nadir, so we slide it toward the camera centre by h / camera_height.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CAMERA_HEIGHT_MM, SHEET_SPEC


def cell_center(name, spec_path=SHEET_SPEC):
    """Sheet mm at the centre of a named grid cell ("A1".."C3")."""
    g = json.loads(open(spec_path).read())["grid"]
    r, c = "ABC".index(name[0].upper()), int(name[1]) - 1
    return (g["origin_mm"][0] + (c + 0.5) * g["cell_mm"], g["origin_mm"][1] + (r + 0.5) * g["cell_mm"])


_DARKEN_LUT = (np.linspace(0, 1, 256) ** 3 * 255).astype(np.uint8)


@dataclass
class SheetPose:
    H: np.ndarray                 # image px -> sheet mm, on the table plane
    tag_ids: list[int]
    fit_error_mm: float           # mean tag-corner residual
    camera_xyz: np.ndarray | None  # camera centre in sheet mm; z = height above table

    def to_sheet(self, uv, height_mm=0.0):
        """Pixel(s) (N,2) -> sheet mm (N,2) for points `height_mm` above the table."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        xy = cv2.perspectiveTransform(uv, self.H).reshape(-1, 2)
        if height_mm and self.camera_xyz is not None:
            c = self.camera_xyz
            xy = c[:2] + (xy - c[:2]) * (c[2] - height_mm) / c[2]
        return xy

    def to_image(self, xy, height_mm=0.0):
        """Sheet mm (N,2) at `height_mm` -> pixels (N,2). Inverse of to_sheet."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        if height_mm and self.camera_xyz is not None:
            c = self.camera_xyz
            xy = c[:2] + (xy - c[:2]) * c[2] / (c[2] - height_mm)
        uv = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), np.linalg.inv(self.H))
        return uv.reshape(-1, 2)


class SheetLocalizer:
    def __init__(self, spec_path=SHEET_SPEC, camera_height_mm=CAMERA_HEIGHT_MM, min_tags=2):
        self.spec = json.loads(open(spec_path).read())
        self.tags = {int(k): np.float64(v["corners_mm"]) for k, v in self.spec["tags"].items()}
        self.camera_height_mm = camera_height_mm
        self.min_tags = min_tags
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.spec["dictionary"]))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect_tags(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found = self._detect(gray)
        if len(found) < len(self.tags):
            # Bright rooms overexpose the paper and lift the tags' black to grey, with dark
            # sharpening halos that break detection (2026-10-10 new location: 0 of 4 tags).
            # A gamma curve pushes the grey back toward black; white stays white.
            found = {**self._detect(cv2.LUT(gray, _DARKEN_LUT)), **found}
        return found

    def _detect(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten()) if int(i) in self.tags}

    def locate(self, frame_bgr) -> SheetPose | None:
        found = self.detect_tags(frame_bgr)
        if len(found) < self.min_tags:
            return None
        ids = sorted(found)
        img = np.concatenate([found[i] for i in ids])
        obj = np.concatenate([self.tags[i] for i in ids])
        H, _ = cv2.findHomography(img, obj)
        resid = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2) - obj
        camera_xyz = self._camera_centre(H, img, obj, frame_bgr.shape) if self.camera_height_mm else None
        return SheetPose(H, ids, float(np.linalg.norm(resid, axis=1).mean()), camera_xyz)

    def _camera_centre(self, H, img, obj, shape):
        """Camera centre in sheet mm, from the tags plus the measured lens height.

        A near top-down view can't separate focal length from height, so the measured
        height fixes the scale: f = height * (pixels per mm at the image centre).
        """
        h, w = shape[:2]
        centre = np.array([[w / 2, h / 2]])
        Hinv = np.linalg.inv(H)
        p0 = cv2.perspectiveTransform(centre.reshape(-1, 1, 2), H).reshape(2)
        px = cv2.perspectiveTransform(np.array([[p0, p0 + [10, 0], p0 + [0, 10]]]), Hinv)[0]
        px_per_mm = (np.linalg.norm(px[1] - px[0]) + np.linalg.norm(px[2] - px[0])) / 20
        f = self.camera_height_mm * px_per_mm
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        obj3 = np.column_stack([obj, np.zeros(len(obj))])
        ok, rvec, tvec = cv2.solvePnP(obj3, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        c = (-R.T @ tvec).reshape(3)
        # Sheet z = x cross y points into the table, so the camera sits at negative z.
        return np.array([c[0], c[1], -c[2]])
