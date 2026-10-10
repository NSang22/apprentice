"""Per-run evidence: an annotated overhead photo and a JSON event at every skill phase.

Capturing is instant (the joint reading and the camera's latest frame are just stored), and
the slow part -- finding objects, drawing, writing files -- runs in save() after the arm has
finished, so evidence never stalls or jitters the motion.

outputs/runs/<YYYYmmdd_HHMMSS>/
    NN_<phase>.jpg   photo with the model's jaw-tip crosshair, moving-jaw arrow, and cubes
    events.jsonl     one line per phase: joints (present + goal), model tip, extra info
"""

import json
import time

import cv2
import numpy as np

from robot.perception.config import ROOT
from robot.perception.scene import build_scene

from .skills import moving_jaw_dir

COLORS = {"red_cube": (0, 0, 255), "blue_cube": (255, 0, 0), "yellow_cube": (0, 220, 255), "orange_cube": (0, 140, 255)}


class RunRecorder:
    def __init__(self, cam, loc, ws, arm, label=""):
        self.cam, self.loc, self.ws, self.arm = cam, loc, ws, arm
        self.dir = ROOT / "outputs" / "runs" / (time.strftime("%Y%m%d_%H%M%S") + (f"_{label}" if label else ""))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.t0 = time.time()
        self.pending = []

    def __call__(self, phase, **info):
        try:
            frame = self.cam.latest()
            self.pending.append((phase, info, self.arm.read(), dict(self.arm.goal or {}), frame,
                                 time.time() - self.t0))
        except Exception as e:  # evidence capture must never break the run
            print(f"  (recorder: {phase} failed: {e})")

    def save(self):
        """Process everything captured so far. Call after the arm has stopped."""
        for item in self.pending:
            try:
                self._record(*item)
            except Exception as e:
                print(f"  (recorder: {item[0]} failed: {e})")
        self.pending = []
        return self.dir

    def _record(self, phase, info, q, goal, frame, t):
        tip_xy, tip_h = self.ws.tip_sheet(q)
        jaw = moving_jaw_dir(self.ws, q)
        if frame is None:
            frame = np.zeros((1080, 1920, 3), np.uint8)
        scene, pose, dets = build_scene(frame, self.loc)

        if pose is not None:
            for d in dets:
                cv2.drawContours(frame, [d.contour.reshape(-1, 1, 2)], -1, COLORS.get(d.name, (255, 255, 255)), 2)
            u, v = pose.to_image([tip_xy], tip_h)[0].astype(int)
            a = pose.to_image([tip_xy + jaw * 40.0], tip_h)[0].astype(int)
            cv2.drawMarker(frame, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.arrowedLine(frame, (u, v), tuple(a), (255, 0, 255), 3, tipLength=0.3)
            cv2.putText(frame, "moving jaw", tuple(a + [8, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            for key in ("cube_xy", "target_xy", "tip_target"):
                if key in info:
                    p = pose.to_image([info[key]], 0.0)[0].astype(int)
                    cv2.drawMarker(frame, tuple(p), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 25, 2)
                    cv2.putText(frame, key, tuple(p + [8, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        lines = [f"{self.n:02d} {phase}  t={t:.1f}s",
                 f"model tips ({tip_xy[0]:.0f}, {tip_xy[1]:.0f}) mm  h {tip_h:.0f} mm  gripper {q['gripper']:.0f}%"]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(frame, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(str(self.dir / f"{self.n:02d}_{phase}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        event = {
            "n": self.n, "phase": phase, "t": round(t, 2),
            "joints": {k: round(float(v), 2) for k, v in q.items()},
            "goal": {k: round(float(v), 2) for k, v in goal.items()},
            "model_tip_sheet_mm": [round(float(tip_xy[0]), 1), round(float(tip_xy[1]), 1)],
            "model_tip_height_mm": round(float(tip_h), 1),
            "moving_jaw_dir": [round(float(jaw[0]), 3), round(float(jaw[1]), 3)],
            "objects": (scene or {}).get("objects", []),
            **{k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()},
        }
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
        self.n += 1
