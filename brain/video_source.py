"""GridState frames from a recorded video, through perception, for brain.extractor --video.

This is the only brain file that imports perception/ and robot/perception/, and the extractor
imports it only for --video, so the live brain never loads them. It calls perception's own
functions (Perceiver, the grid lock, segment.detect); it adds no detection or color logic.

Frames are decoded from the file, never a camera. About HZ frames per second of video are run
through perception.perceive.Perceiver, and every timestamp comes from the video: t is the frame's
own position (CAP_PROP_POS_MSEC, or frame_index / fps if that reads 0), and the Perceiver's clock
(its empty-board refresh timer) reads the same video time. Same video in, same frames out.

Grid lock and empty board, two modes:
  --calib <folder>  its grid_lock.json, empty_board.png and colors.json. Stops if the lock was
                    taken at another frame size; a lock with no saved size is used with a warning.
  first frame       (default) the first frame in the first FIRST_FRAME_SEARCH_S seconds that shows
                    all 4 tags becomes the grid lock and the empty-board picture. That picture
                    could hide a block already on the board (perception only sees what changed
                    since it), so it is checked on its own: segment.detect, which finds colorful
                    blobs without any reference, must find nothing on the sheet. reference_empty
                    holds the answer for the gate's starts_empty check.
                    Colors come from perception/calib/colors.json if present, else perception's
                    nominal colors (uncalibrated).
Either way the files are copied to a temp folder first: Perceiver writes its lock and empty
board back while it runs, and nothing here may change the team's calibration.

For each settled board the 9 cells of that frame are cropped (on_settle), so save_pictures()
can write one picture per step.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np

from perception.blocks import sheet_roi
from perception.cells import grid_geometry
from perception.colors import CALIB_PATH, load_calibration, parse_colors
from perception.grid_lock import load_lock
from perception.perceive import Perceiver
from perception.robot_bridge import default_active_colors, set_block_mm
from robot.perception import segment, sheet
from shared.config import CELLS
from shared.schemas import GridState

HZ = 10.0                     # frames per second of video run through perception (the live feed's rate)
FIRST_FRAME_SEARCH_S = 1.0    # first-frame mode: look this far into the video for all 4 tags
PROGRESS_EVERY_S = 5.0
CROP_MARGIN = 0.15            # step pictures: this share of a cell around it on every side
LOCK, EMPTY, COLORS_JSON = "grid_lock.json", "empty_board.png", "colors.json"


class VideoSourceError(RuntimeError):
    """The video cannot be turned into frames (file, calibration, or no grid in view)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VideoSource:
    def __init__(self, path: Path, colors: Optional[str], block_mm: Optional[float],
                 calib_dir: Optional[Path] = None, hz: float = HZ, log: Callable[[str], None] = print):
        self.path = Path(path)
        self.log = log
        if not self.path.is_file():
            raise VideoSourceError(f"no video file at {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise VideoSourceError(f"could not open {self.path} as a video")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.stride = max(1, round(self.fps / hz)) if self.fps > 0 else 1
        self.active = parse_colors(colors) if colors else default_active_colors()
        self.block_mm, line = set_block_mm(block_mm)
        log(f"[extract] {line}")
        log(f"[extract] colors in play: {', '.join(self.active)} (anything else reads unknown)")

        self.tmp = Path(tempfile.mkdtemp(prefix="apprentice_extract_"))
        # SheetLocalizer loads robot/outputs/camera_pose.json when it is created and rewrites it on
        # every 3+ tag fit. Pointing it at the temp folder, in this process only, keeps the video's
        # camera pose out of the robot's cache and keeps an earlier run's cache out of this one.
        self._robot_cache = sheet.CAMERA_CACHE
        sheet.CAMERA_CACHE = self.tmp / "camera_pose.json"

        self.calib_dir = None if calib_dir is None else Path(calib_dir)
        calibration: dict[str, Any] = {"mode": "calib" if self.calib_dir else "first_frame",
                                       "calib_dir": None if self.calib_dir is None else str(self.calib_dir)}
        files: dict[str, Any] = {}
        colors_path = self.tmp / COLORS_JSON
        if self.calib_dir is not None:
            for name in (LOCK, EMPTY):
                if not (self.calib_dir / name).is_file():
                    raise VideoSourceError(f"--calib {self.calib_dir} has no {name}")
            for name in (LOCK, EMPTY, COLORS_JSON):
                src = self.calib_dir / name
                if src.is_file():
                    shutil.copyfile(src, self.tmp / name)
                    files[name] = {"from": str(src), "sha256": _sha256(src)}
            lock = load_lock(self.tmp / LOCK)
            lock_size = tuple(lock.frame_size) if lock is not None else (0, 0)
            calibration["lock_frame_size"] = list(lock_size)
            if lock_size == (0, 0):
                warning = f"{LOCK} has no saved frame size; using it anyway (the video is {self.size[0]}x{self.size[1]})"
                calibration["lock_size_warning"] = warning
                log(f"[extract] WARNING: {warning}")
            elif lock_size != self.size:
                raise VideoSourceError(f"--calib grid lock was taken at {lock_size[0]}x{lock_size[1]}, but the video is "
                                       f"{self.size[0]}x{self.size[1]}. Calibrate at the video's resolution, or leave "
                                       "out --calib to lock the grid from the video's first second.")
        elif CALIB_PATH.is_file():
            shutil.copyfile(CALIB_PATH, colors_path)
            files[COLORS_JSON] = {"from": str(CALIB_PATH), "sha256": _sha256(CALIB_PATH)}
        calib = load_calibration(colors_path)
        if not calib.calibrated:
            log("[extract] WARNING: no colors.json, so block colors are perception's nominal (uncalibrated) colors")
        calibration.update({"files": files, "colors_calibrated": calib.calibrated, "colors_in_play": self.active,
                            "block_mm": self.block_mm})

        self.now = 0.0  # video time of the current frame: the Perceiver's clock
        self.perceiver = Perceiver(calib, lock_path=self.tmp / LOCK, ref_path=self.tmp / EMPTY,
                                   clock=lambda: self.now, log=lambda m: log(f"[extract] {m}"),
                                   active=self.active, block_mm=self.block_mm)
        if self.calib_dir is not None and self.perceiver.lock is None:
            raise VideoSourceError(f"--calib {self.calib_dir}: perception could not load the grid lock and empty board")
        self.reference_empty: Optional[bool] = None  # first-frame mode: set once the reference is checked
        self.info: dict[str, Any] = {
            "source": {"kind": "video", "path": str(self.path), "sha256": _sha256(self.path), "fps": self.fps,
                       "frames": frame_count, "width": self.size[0], "height": self.size[1],
                       "duration_s": round(frame_count / self.fps, 3) if self.fps else None,
                       "every_nth_frame": self.stride},
            "calibration": calibration,
        }
        self._frame: Optional[np.ndarray] = None
        self._frame_id = -1
        self._crops: dict[int, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------ frames

    def _time(self, index: int) -> float:
        ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if ms > 0:
            return ms / 1000.0
        return index / self.fps if self.fps > 0 else 0.0

    def frames(self) -> Iterator[GridState]:
        started = self.calib_dir is not None
        index, next_index, last_why, next_progress = -1, 0, "no frames", 0.0
        reasons: Counter[str] = Counter()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            index += 1
            t = self._time(index)
            if not started:
                if t > FIRST_FRAME_SEARCH_S:
                    break
                ok_lock, why = self.perceiver.capture(frame)
                if not ok_lock:
                    last_why = why
                    continue
                started, next_index = True, index
                self._check_reference(frame, index, t)
            if index < next_index:
                continue
            next_index = index + self.stride
            self.now = t
            result = self.perceiver.step(frame, index, timestamp=t)
            if not result.state.hands_clear:
                reasons[result.reason or result.status] += 1
            self._frame, self._frame_id = frame, index
            if t >= next_progress:
                self.log(f"[extract] {t:5.1f} s  {result.status}")
                next_progress += PROGRESS_EVERY_S
            yield result.state
        self.info["frames"] = {"perception_not_clear": dict(sorted(reasons.items()))}
        if not started:
            raise VideoSourceError(f"no frame in the first {FIRST_FRAME_SEARCH_S:g} s shows all 4 AprilTags "
                                   f"({last_why}). Start the recording on an empty board with all 4 tags in view, "
                                   "or pass --calib <folder>.")

    def _check_reference(self, frame: np.ndarray, index: int, t: float) -> None:
        """First-frame mode: is the frame used as the empty board really empty? segment.detect
        finds colorful blobs without any reference, so it can see a block the picture would hide."""
        pose = self.perceiver.lock.pose()
        found = segment.detect(frame, None, roi_mask=sheet_roi(pose, frame.shape), pose=pose)
        self.reference_empty = not found
        self.info["calibration"]["reference"] = {
            "frame": index, "t": round(t, 3), "empty_check": "segment.detect on the reference frame",
            "blobs_found": [{"name": d.name, "center_px": [round(float(v), 1) for v in d.center_px]} for d in found],
        }
        self.log(f"[extract] grid locked from frame {index} ({t:.2f} s); "
                 + ("no block on it" if not found else f"WARNING: {len(found)} colorful blob(s) on the empty-board picture"))

    # ------------------------------------------------------------ step pictures

    def on_settle(self, settled: Any) -> None:
        """Crop every cell of the frame that settled a board (the current frame)."""
        if settled.frame_id != self._frame_id or self._frame is None or self.perceiver.lock is None:
            return
        self._crops[settled.frame_id] = {cell: self._crop(self._frame, cell) for cell in CELLS}

    def _crop(self, frame: np.ndarray, cell: str) -> np.ndarray:
        x0, y0, size = grid_geometry()
        col, row = "ABC".index(cell[0]), int(cell[1]) - 1
        m = CROP_MARGIN * size
        a, b = x0 + col * size - m, y0 + row * size - m
        square = [(a, b), (a + size + 2 * m, b), (a + size + 2 * m, b + size + 2 * m), (a, b + size + 2 * m)]
        pose = self.perceiver.lock.pose()
        pts = np.vstack([pose.to_image(square, 0.0), pose.to_image(square, self.block_mm)])
        h, w = frame.shape[:2]
        u0, v0 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [w, h])
        u1, v1 = np.clip(np.ceil(pts.max(axis=0)).astype(int), 0, [w, h])
        return frame[v0:v1, u0:u1].copy()

    def save_pictures(self, folder: Path, steps: list[tuple[int, str, int]]) -> list[str]:
        """step_<n>_<cell>.png for each (step number, cell, frame id). Returns the files written."""
        written = []
        for number, cell, frame_id in steps:
            crop = self._crops.get(frame_id, {}).get(cell)
            if crop is None or crop.size == 0:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"step_{number}_{cell}.png"
            cv2.imwrite(str(path), crop)
            written.append(path.as_posix())
        return written

    def close(self) -> None:
        sheet.CAMERA_CACHE = self._robot_cache
        self.cap.release()
        shutil.rmtree(self.tmp, ignore_errors=True)
