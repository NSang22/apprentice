"""SO-101 forward/inverse kinematics from the URDF, in plain numpy.

Joint angles are LeRobot follower readings in degrees (use_degrees=True); LeRobot feeds
them to this URDF as plain deg->rad, no sign flips. Positions are in mm in the arm's
base_link frame.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent / "urdf" / "so101_new_calib.urdf"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LINK = "gripper_frame_link"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, q):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class SO101Kinematics:
    def __init__(self, urdf_path=URDF, tip_link=TIP_LINK):
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip_link
        while link in by_child:  # walk tip -> base
            j = by_child[link]
            o = j.find("origin")
            T = np.eye(4)
            T[:3, :3] = _rpy(*map(float, o.get("rpy", "0 0 0").split()))
            T[:3, 3] = np.array(list(map(float, o.get("xyz", "0 0 0").split()))) * 1000.0  # m -> mm
            axis = j.find("axis")
            lim = j.find("limit")
            chain.append({
                "name": j.get("name"),
                "type": j.get("type"),
                "origin": T,
                "axis": None if axis is None else list(map(float, axis.get("xyz").split())),
                "limits_deg": None if lim is None else (np.degrees(float(lim.get("lower"))), np.degrees(float(lim.get("upper")))),
            })
            link = j.find("parent").get("link")
        self.chain = chain[::-1]  # base -> tip
        self.limits = {c["name"]: c["limits_deg"] for c in self.chain if c["name"] in ARM_JOINTS}

    def fk(self, q_deg):
        """q_deg: dict name->deg or sequence in ARM_JOINTS order. Returns 4x4 tip pose (mm)."""
        if not isinstance(q_deg, dict):
            q_deg = dict(zip(ARM_JOINTS, q_deg))
        T = np.eye(4)
        for c in self.chain:
            T = T @ c["origin"]
            if c["type"] == "revolute" and c["name"] in q_deg:
                R = np.eye(4)
                R[:3, :3] = _axis_angle(c["axis"], np.radians(q_deg[c["name"]]))
                T = T @ R
        return T

    def tip(self, q_deg):
        return self.fk(q_deg)[:3, 3]

    def fk_frames(self, q_deg):
        """Tip pose plus, for each ARM_JOINTS joint, its world axis and origin (for Jacobians)."""
        if not isinstance(q_deg, dict):
            q_deg = dict(zip(ARM_JOINTS, q_deg))
        T = np.eye(4)
        axes, origins = {}, {}
        for c in self.chain:
            T = T @ c["origin"]
            if c["type"] == "revolute" and c["name"] in q_deg:
                axes[c["name"]] = T[:3, :3] @ (np.asarray(c["axis"], float) / np.linalg.norm(c["axis"]))
                origins[c["name"]] = T[:3, 3].copy()
                R = np.eye(4)
                R[:3, :3] = _axis_angle(c["axis"], np.radians(q_deg[c["name"]]))
                T = T @ R
        return T, axes, origins

    def ik_down(self, target_mm, q0_deg, wrist_roll_deg=0.0, tip_mm=0.0, down_weight=100.0, iters=200):
        """Joints putting the tip (extended `tip_mm` along the gripper axis) at `target_mm`
        with the gripper pointing straight down.

        Solves pan/lift/elbow/wrist_flex by damped least squares; wrist_roll is fixed.
        Returns (q_deg list in ARM_JOINTS order, position error mm, tilt from vertical deg).
        """
        target = np.asarray(target_mm, float)
        q = np.array(q0_deg, float)
        q[4] = wrist_roll_deg
        free = [0, 1, 2, 3]
        lo = np.array([self.limits[j][0] for j in ARM_JOINTS])
        hi = np.array([self.limits[j][1] for j in ARM_JOINTS])

        def resid_jac(qv):
            # One forward pass gives everything: for a revolute joint with world axis a at o, a
            # point p on the tip moves dp/dq = a x (p - o) and a tip direction z turns dz/dq = a x z.
            T, axes, origins = self.fk_frames(qv)
            z = T[:3, 2]
            p = T[:3, 3] + z * tip_mm
            # Pointing down means the gripper axis is (0, 0, -1): penalize its x and y.
            r = np.concatenate([p - target, down_weight * z[:2]])
            J = np.empty((5, len(free)))
            for k, i in enumerate(free):
                a, o = axes[ARM_JOINTS[i]], origins[ARM_JOINTS[i]]
                J[:3, k] = np.cross(a, p - o)
                J[3:, k] = down_weight * np.cross(a, z)[:2]
            return r, J * (np.pi / 180.0)  # per degree

        lam = 1e-2
        r, J = resid_jac(q)
        for _ in range(iters):
            step = np.linalg.solve(J.T @ J + lam * np.eye(len(free)), -J.T @ r)
            q_new = q.copy()
            q_new[free] = np.clip(q[free] + step, lo[free], hi[free])
            r_new, J_new = resid_jac(q_new)
            if r_new @ r_new < r @ r:
                gain = r @ r - r_new @ r_new
                moved = np.abs(q_new - q).max()
                q, r, J, lam = q_new, r_new, J_new, max(lam * 0.3, 1e-6)
                # Done when steps are tiny, or when creeping along a joint limit gains nothing.
                if moved < 1e-3 or gain < 1e-9 * max(r @ r, 1e-6):
                    break
            else:
                lam *= 4
                if lam > 1e8:
                    break
        T = self.fk(q)
        tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
        return q.tolist(), float(np.linalg.norm(r[:3])), tilt
