"""Overhead camera capture (OpenCV, DirectShow on Windows).

A background thread keeps reading so read() always returns the newest frame. Without it,
DirectShow can hand back frames buffered seconds ago when the camera sat idle (e.g. while
the arm was moving), which showed the arm where it no longer was.
"""

import sys
import threading
import time

import cv2

from .config import CAMERA_INDEX, FRAME_H, FRAME_W

# DirectShow on Windows; Video4Linux2 elsewhere (e.g. Jetson). Index numbering and
# available controls differ by backend - re-verify CAMERA_INDEX on each platform
# (`v4l2-ctl --list-devices` on Linux), don't assume it carries over.
CAPTURE_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2


class Camera:
    def __init__(self, index=CAMERA_INDEX, width=FRAME_W, height=FRAME_H, warmup_s=1.5):
        self.cap = cv2.VideoCapture(index, CAPTURE_BACKEND)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Can't open camera {index}. Close the Windows Camera app or anything else using it."
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame, self._stamp = None, 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(warmup_s)  # let auto-exposure settle

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._stamp = frame, time.time()

    def read(self, max_age_s=0.5, timeout_s=3.0):
        """Newest frame captured after this call started (and at most max_age_s old)."""
        asked = time.time()
        deadline = asked + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp >= asked - max_age_s and self._stamp > asked:
                    return self._frame.copy()
            time.sleep(0.01)
        raise RuntimeError("Camera returned no fresh frame")

    def latest(self):
        """The most recent frame, without waiting (None before the first frame arrives)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
