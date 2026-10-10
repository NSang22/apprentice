"""Camera-closed-loop placement correction.

From placements where we commanded the cube to sheet point c and the camera then saw it
at o, fit o ~= A c + b (where the arm really puts things, as the camera sees it). To get a
cube to camera point t, command c = A^-1 (t - b). The same mapping serves picks: the camera
says the cube is at o, so reach for A^-1 (o - b).
"""

import json
from pathlib import Path

import numpy as np

CORRECTION_FILE = Path(__file__).resolve().parent / "calibration" / "placement_correction.json"


def fit(commanded, observed):
    """Least-squares affine fit. Returns (A 2x2, b 2, rms before mm, rms after mm)."""
    C, O = np.asarray(commanded, float), np.asarray(observed, float)
    X = np.column_stack([C, np.ones(len(C))])
    M, *_ = np.linalg.lstsq(X, O, rcond=None)  # O = X @ M
    A, b = M[:2].T, M[2]
    before = float(np.sqrt(np.mean(np.sum((O - C) ** 2, axis=1))))
    after = float(np.sqrt(np.mean(np.sum((O - (C @ A.T + b)) ** 2, axis=1))))
    return A, b, before, after


GROSS_MM = 30.0      # a placement this far off is a failure (or the cube was moved by hand), not data
OUTLIER_MM = 10.0    # after fitting, drop the worst point while its residual exceeds this
MIN_INLIERS = 5
MAX_RMS_AFTER_MM = 6.0


def fit_robust(commanded, observed):
    """Affine fit that ignores failed placements. Returns (A, b, before, after, inlier mask, reason).
    reason is None when the fit is trustworthy, else why it must not be saved.

    The 10:31 run on 2026-10-10 had two 90 mm 'placements' where a hand moved the cube; a
    plain least-squares fit swallowed them and produced a correction that scaled x by 0.35."""
    C, O = np.asarray(commanded, float), np.asarray(observed, float)
    keep = np.hypot(*(O - C).T) < GROSS_MM
    while keep.sum() >= MIN_INLIERS:
        A, b, _, _ = fit(C[keep], O[keep])
        resid = np.hypot(*(O - (C @ A.T + b)).T)
        worst = np.argmax(np.where(keep, resid, -1))
        if resid[worst] <= OUTLIER_MM or keep.sum() == MIN_INLIERS:
            break
        keep[worst] = False
    if keep.sum() < MIN_INLIERS:
        return None, None, None, None, keep, f"only {keep.sum()} usable placements (need {MIN_INLIERS})"
    A, b, before, after = fit(C[keep], O[keep])
    if not (np.all(np.abs(np.diag(A) - 1) < 0.15) and abs(A[0, 1]) < 0.15 and abs(A[1, 0]) < 0.15):
        return A, b, before, after, keep, f"fit is implausible (A = {np.round(A, 2).tolist()}); expected close to identity"
    if after > MAX_RMS_AFTER_MM:
        return A, b, before, after, keep, f"residual {after:.1f} mm after fitting is too large"
    return A, b, before, after, keep, None


def save(A, b, before, after, n, path=CORRECTION_FILE):
    path.write_text(json.dumps({"A": np.round(A, 6).tolist(), "b": np.round(b, 3).tolist(),
                                "rms_before_mm": round(before, 2), "rms_after_fit_mm": round(after, 2),
                                "n_placements": n}, indent=2))


class Correction:
    """Maps a camera-frame target to the sheet point to command. Identity if no file."""

    def __init__(self, path=CORRECTION_FILE):
        self.A, self.b = np.eye(2), np.zeros(2)
        self.loaded = False
        if Path(path).exists():
            c = json.loads(Path(path).read_text())
            self.A, self.b = np.array(c["A"]), np.array(c["b"])
            self.loaded = True
        self.Ainv = np.linalg.inv(self.A)

    def __call__(self, xy):
        return self.Ainv @ (np.asarray(xy, float) - self.b)
