"""Fit the sheet <-> arm-base transform plus small joint zero offsets from touch points.

Model: tip_base(q + offsets) = transform(sheet_xy, table_z), where
  - offsets: zero corrections (deg) for pan, lift, elbow, wrist_flex
  - transform: sheet mm -> base mm. The sheet's y runs down the page, which seen from
    above is a mirror of the base's z-up frame, hence the y flip.
  - table_z: height of the sheet surface in the base frame
  - tip_mm: extra tip length along the gripper axis (where the jaw tips really are)
"""

import json
from pathlib import Path

import numpy as np

from .kinematics import ARM_JOINTS, SO101Kinematics

CALIB_FILE = Path(__file__).resolve().parent / "calibration" / "arm_sheet_calibration.json"
OFFSET_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
# param vector: 4 offsets (deg), theta (deg), tx, ty, table_z, tip_mm
N_PARAMS = len(OFFSET_JOINTS) + 5


def _q(joints_deg, params):
    q = dict(joints_deg)
    for j, off in zip(OFFSET_JOINTS, params[:4]):
        q[j] = q[j] + off
    return [q[j] for j in ARM_JOINTS]


def tip_with_extension(kin, q, tip_mm):
    T = kin.fk(q)
    return T[:3, 3] + T[:3, 2] * tip_mm


def sheet_to_base_xy(sheet_xy, theta_deg, tx, ty):
    t = np.radians(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(sheet_xy, float) * [1, -1]) @ R.T + [tx, ty]


def residuals(params, kin, points):
    theta, tx, ty, table_z, tip_mm = params[4:]
    out = []
    for p in points:
        tip = tip_with_extension(kin, _q(p["joints_deg"], params), tip_mm)
        target = sheet_to_base_xy(p["sheet_mm"], theta, tx, ty)
        out += [tip[0] - target[0], tip[1] - target[1], tip[2] - table_z]
    return np.array(out)


PLACEMENT_ONLY = [4, 5, 6, 7]  # theta, tx, ty, table_z: what changes when the setup moves


def fit(points, kin=None, init=None, iters=100, free=None):
    """Levenberg-Marquardt with a numeric Jacobian. Returns (params, rms residual mm).
    free: indices of params to fit (default all); the rest stay at their `init` values."""
    kin = kin or SO101Kinematics()
    p = np.zeros(N_PARAMS) if init is None else np.array(init, float)
    free = list(range(N_PARAMS)) if free is None else list(free)
    lam = 1e-2
    r = residuals(p, kin, points)
    for _ in range(iters):
        J = np.empty((len(r), len(free)))
        for k, i in enumerate(free):
            dp = np.zeros(N_PARAMS)
            dp[i] = 1e-4
            J[:, k] = (residuals(p + dp, kin, points) - r) / 1e-4
        A = J.T @ J
        step = np.zeros(N_PARAMS)
        step[free] = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
        r_new = residuals(p + step, kin, points)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-6:
                break
        else:
            lam *= 10
    return p, float(np.sqrt(np.mean(r ** 2)))


def initial_guess(points, kin):
    """Rigid 2D fit with zero offsets, to start LM near the right rotation."""
    S = np.array([p["sheet_mm"] for p in points]) * [1, -1]
    P = np.array([kin.tip(_q(p["joints_deg"], np.zeros(N_PARAMS))) for p in points])
    mu_s, mu_p = S.mean(0), P[:, :2].mean(0)
    U, _, Vt = np.linalg.svd((S - mu_s).T @ (P[:, :2] - mu_p))
    R = (U @ Vt).T
    theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    t = mu_p - mu_s @ R.T
    return np.array([0, 0, 0, 0, theta, t[0], t[1], P[:, 2].mean(), 0.0])


def load_params(path=CALIB_FILE):
    c = json.loads(Path(path).read_text())
    s = c["sheet_to_base"]
    return np.array([*(c["joint_offsets_deg"][j] for j in OFFSET_JOINTS),
                     s["theta_deg"], s["tx_mm"], s["ty_mm"], c["table_z_mm"], c["tip_extension_mm"]])


def save(params, rms, n_points, path=CALIB_FILE, mode="full"):
    theta, tx, ty, table_z, tip_mm = params[4:]
    path.write_text(json.dumps({
        "joint_offsets_deg": {j: round(float(v), 3) for j, v in zip(OFFSET_JOINTS, params[:4])},
        "sheet_to_base": {"theta_deg": round(float(theta), 3), "tx_mm": round(float(tx), 2),
                          "ty_mm": round(float(ty), 2), "mirror_y": True},
        "table_z_mm": round(float(table_z), 2),
        "tip_extension_mm": round(float(tip_mm), 2),
        "fit_rms_mm": round(rms, 2),
        "n_points": n_points,
        "mode": mode,
    }, indent=2))
