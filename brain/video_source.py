"""GridState frames from a recorded video, through perception, for brain.extractor --video.

This is the only brain file that imports perception/ and robot/perception/, and the extractor
imports it only for --video, so the live brain never loads them. It calls perception's own
functions (Perceiver, the grid lock, segment.detect); it adds no detection or color logic.

Frames are decoded from the file, never a camera. About SAMPLE_HZ frames per second of video are run
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

Own hand check: perception's hands_clear misses a dark sleeve and counts a hand resting on a cell
as part of a block. So every frame is also checked here: a band of sheet around the 3x3 grid, outside
everything a 2-high stack on any cell can cover in the picture (each cell's column projected from
the locked camera position, plus HAND_BAND_CLEAR_MM), is compared with the empty-board picture
(perception's change_mask). Anything reaching into the grid crosses the band; when more than
HAND_BAND_MAX of it changed, the frame goes out as not clear (hands_clear false), so it can never be
part of a settled board.

For each settled board the 9 cells of that frame are cropped (on_settle), so save_pictures()
can write one picture per step. on_stable() keeps a copy of the latest frame the stabilizer would
accept, so the extractor's final read can name and crop it after the video has moved on.

With a robot color memory (--color-memory, or robot/outputs/color_memory.json when that exists;
robot/perception/color.py's ColorMemory file), each
settled frame is also named the way the robot names blocks (robot/perception/scene.build_scene):
segment.detect blobs, then ColorNamer.name with the vision model off. Each blob goes to the cell
under its centre at half a block's height, as the robot places blobs (pose.to_sheet, then
perception's mm_to_cell); a cell's read is the name of its one blob, so it does not depend on
perception's own reading of that cell. It counts only when the namer says it came from memory
(last_source "memory"); its fallback hue guess, a blob too big for one block, two blobs in one
cell, or none reads "unknown". The robot's "light blue" is skyblue
(perception/robot_bridge.ROBOT_NAMES). robot_read(frame_id, cell) hands the reads to the extractor.
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
from perception.cells import cell_center_mm, grid_geometry, mm_to_cell
from perception.change import SHEET_PX_PER_MM, change_mask, sheet_size_px, warp_to_sheet
from perception.colors import CALIB_PATH, load_calibration, parse_colors
from perception.grid_lock import load_lock
from perception.perceive import FrameResult, Perceiver
from perception.robot_bridge import ROBOT_NAMES, default_active_colors, set_block_mm
from robot.perception import color as robot_color
from robot.perception import naming as robot_naming
from robot.perception import segment, sheet
from shared.config import CELLS, UNKNOWN
from shared.schemas import GridState

SAMPLE_HZ = 5.0               # frames per second of video run through perception. 5 gave the same steps,
                              # corrections and gate as 10 on expert_take1/2 in about half the time
# Perception counts some things in frames, not seconds, so they span more time at a lower rate:
# CellVote's 5-frame majority, StackMemory's 3 stable frames, RELOCK_FRAMES 5.
FIRST_FRAME_SEARCH_S = 1.0    # first-frame mode: look this far into the video for all 4 tags
PROGRESS_EVERY_S = 5.0
CROP_MARGIN = 0.15            # step pictures: this share of a cell around it on every side
LOCK, EMPTY, COLORS_JSON = "grid_lock.json", "empty_board.png", "colors.json"
HAND_BAND_STACK = 2           # the band stays clear of a stack this many blocks high on any cell
HAND_BAND_CLEAR_MM = 8.0      # ... plus this much (a block off-centre in its cell, its shadow edge)
HAND_BAND_EDGE_MM = 3.0       # and this far inside the paper's edge (warp error at the edge)
HAND_BAND_MAX = 0.010         # share of the band changed above which a frame is not clear: clean frames read
                              # at most 0.007 on expert_take1/2, the faintest hand 0.012 (take 1 frame 611)


def hand_band(pose: Any, block_mm: float) -> np.ndarray:
    """Sheet-warp mask (bool) of the hand band: the paper (HAND_BAND_EDGE_MM inside its edge) minus
    every cell's column HAND_BAND_STACK blocks high as the camera sees it on the table, minus
    HAND_BAND_CLEAR_MM around those columns."""
    w, h = sheet_size_px()
    e = int(round(HAND_BAND_EDGE_MM * SHEET_PX_PER_MM))
    band = np.zeros((h, w), np.uint8)
    band[e:h - e, e:w - e] = 255
    _, _, size = grid_geometry()
    for cell in CELLS:
        cx, cy = cell_center_mm(cell)
        sq = np.array([(cx - size / 2, cy - size / 2), (cx + size / 2, cy - size / 2),
                       (cx + size / 2, cy + size / 2), (cx - size / 2, cy + size / 2)])
        top = pose.to_sheet(pose.to_image(sq, HAND_BAND_STACK * block_mm), 0.0)  # its top, seen on the table
        hull = cv2.convexHull(np.vstack([sq, top]).astype(np.float32)) * SHEET_PX_PER_MM
        cv2.fillPoly(band, [hull.astype(np.int32)], 0)
    k = int(round(HAND_BAND_CLEAR_MM * SHEET_PX_PER_MM)) * 2 + 1
    return cv2.erode(band, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) > 0


# change_mask decides a pixel from pixels at most 14 px away (5x5 blur, 5x5 open, 9x9 close), so running it
# on the band's bounding box grown by this much gives exactly the same band pixels as the whole sheet.
BAND_PAD_PX = 24


def band_change(live_sheet: np.ndarray, reference_sheet: np.ndarray, band: np.ndarray) -> Optional[float]:
    """Share of the band that changed since the empty board (perception's change_mask)."""
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        return None
    h, w = band.shape
    y0, y1 = max(0, ys.min() - BAND_PAD_PX), min(h, ys.max() + 1 + BAND_PAD_PX)
    x0, x1 = max(0, xs.min() - BAND_PAD_PX), min(w, xs.max() + 1 + BAND_PAD_PX)
    if reference_sheet.shape[:2] != live_sheet.shape[:2]:  # change_mask would resize it to the live sheet first
        reference_sheet = cv2.resize(reference_sheet, (live_sheet.shape[1], live_sheet.shape[0]))
    changed = change_mask(live_sheet[y0:y1, x0:x1], reference_sheet[y0:y1, x0:x1]) > 0
    return float(changed[band[y0:y1, x0:x1]].mean())


class VideoSourceError(RuntimeError):
    """The video cannot be turned into frames (file, calibration, or no grid in view)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VideoSource:
    def __init__(self, path: Path, colors: Optional[str], block_mm: Optional[float],
                 calib_dir: Optional[Path] = None, color_memory: Optional[Path] = None,
                 hz: float = SAMPLE_HZ, log: Callable[[str], None] = print):
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
        log(f"[extract] sampling {hz:g} frames per second (every {self.stride} video frame(s))")
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
        self._result: Optional[FrameResult] = None
        self._stable: Optional[tuple[int, np.ndarray]] = None  # (frame id, frame): latest acceptable frame
        self._band: Optional[np.ndarray] = None
        self._band_key: Optional[bytes] = None
        self.band_max = HAND_BAND_MAX
        self.band_log: list[tuple[int, float, float]] = []  # (frame id, t, share of the band changed)
        self.hand_frames = 0          # frames perception called clear that the band check threw out
        self._crops: dict[int, dict[str, np.ndarray]] = {}
        self._robot_reads: dict[int, dict[str, str]] = {}
        self._robot_dist: dict[int, dict[str, dict[str, float]]] = {}  # frame -> cell -> color -> distance
        self._robot_active = robot_naming.ACTIVE_COLORS
        if color_memory is None and robot_color.MEMORY_FILE.is_file():
            color_memory = robot_color.MEMORY_FILE  # the robot's own file, read only
        self.namer = None if color_memory is None else self._robot_namer(Path(color_memory))
        self.clear_limits = (float(robot_color.KNOWN_MAX), float(robot_color.MIN_MARGIN))  # a clear memory read

    def _robot_namer(self, path: Path) -> Any:
        if not path.is_file():
            raise VideoSourceError(f"no color memory file at {path}")
        memory = robot_color.ColorMemory(path)  # read only: learn() and save() are never called
        if not memory.colors:
            raise VideoSourceError(f"{path} has no remembered colors")
        allowed = [name for name in memory.colors if ROBOT_NAMES.get(name, name) in self.active]
        # ColorNamer only names robot ACTIVE_COLORS (red, purple, yellow, teal), so orange and light
        # blue could never be named. Here, in this process only, it may name the colors in play
        # instead (as robot_bridge sets the block size); close() puts the robot's list back.
        robot_naming.ACTIVE_COLORS = allowed
        self.info["color_memory"] = {"path": str(path), "sha256": _sha256(path), "remembered": sorted(memory.colors),
                                     "allowed": allowed}
        self.log(f"[extract] robot color memory {path}: naming {', '.join(allowed)}")
        return robot_naming.ColorNamer(memory=memory, use_model=False, log=lambda m: None)

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
            if started and index + 1 < next_index:  # a frame between samples: move past it without decoding it
                if not self.cap.grab():
                    break
                index += 1
                continue
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
            state = result.state
            change = self._band_change(frame, result)
            if change is not None:
                self.band_log.append((index, t, change))
                if self.band_max is not None and change > self.band_max and state.hands_clear:
                    state = GridState(state.t, state.board, state.arm_clear, False, state.confidence, state.frame_id)
                    self.hand_frames += 1
            self._frame, self._frame_id, self._result = frame, index, result
            if t >= next_progress:
                self.log(f"[extract] {t:5.1f} s  {result.status}")
                next_progress += PROGRESS_EVERY_S
            yield state
        self.info["frames"] = {"perception_not_clear": dict(sorted(reasons.items())),
                               "hand_band_not_clear": self.hand_frames, "hand_band_max": self.band_max}
        if not started:
            raise VideoSourceError(f"no frame in the first {FIRST_FRAME_SEARCH_S:g} s shows all 4 AprilTags "
                                   f"({last_why}). Start the recording on an empty board with all 4 tags in view, "
                                   "or pass --calib <folder>.")

    def band_mask(self) -> Optional[np.ndarray]:
        """The hand band for the current grid lock, recomputed when the lock changes."""
        lock = self.perceiver.lock
        if lock is None:
            return None
        key = np.asarray(lock.H).tobytes()
        if key != self._band_key:
            self._band, self._band_key = hand_band(lock.pose(), self.block_mm), key
        return self._band

    def _band_change(self, frame: np.ndarray, result: FrameResult) -> Optional[float]:
        """Share of the hand band that differs from the empty-board picture, None without a lock."""
        band, ref = self.band_mask(), self.perceiver.reference
        if band is None or ref is None or result.grid != "locked":
            return None
        live = result.analysis.live_sheet if result.analysis is not None and result.analysis.live_sheet is not None             else warp_to_sheet(frame, self.perceiver.lock.H)
        return band_change(live, ref, band)

    def on_stable(self, frame_id: int) -> None:
        """The extractor calls this for each frame the stabilizer would accept: keep a copy."""
        if frame_id == self._frame_id and self._frame is not None:
            self._stable = (frame_id, self._frame.copy())

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
        """For the frame that settled a board (the current frame): crop every cell, and with a
        color memory, name each occupied cell's top block the way the robot does."""
        if self.perceiver.lock is None:
            return
        if settled.frame_id == self._frame_id and self._frame is not None:
            frame = self._frame
        elif self._stable is not None and settled.frame_id == self._stable[0]:
            frame = self._stable[1]  # the final read, taken after the video moved on
        else:
            return
        self._crops[settled.frame_id] = {cell: self._crop(frame, cell) for cell in CELLS}
        if self.namer is not None:
            self._robot_reads[settled.frame_id], self._robot_dist[settled.frame_id] = self._robot_names(frame, settled.board)

    def _robot_names(self, frame: np.ndarray, board: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        pose = self.perceiver.lock.pose()
        tags = {k: v["corners_mm"] for k, v in self.perceiver.localizer.spec["tags"].items()}
        blobs = segment.detect(frame, None, pose=pose, tags=tags)  # as robot/perception/scene.build_scene
        by_cell: dict[str, list[Any]] = {}
        for b in blobs:  # the robot places a blob by its centre at half a block's height (scene.build_scene)
            x, y = pose.to_sheet([b.center_px], self.block_mm / 2.0)[0]
            by_cell.setdefault(mm_to_cell(float(x), float(y)), []).append(b)
        reads, dist = {}, {}
        memory = self.namer.memory
        for cell in (c for c in CELLS if board[c]):
            hits = by_cell.get(cell, [])
            if len(hits) != 1 or hits[0].score < 0:
                reads[cell] = UNKNOWN  # no blob, two blobs, or one blob too big for one block
                continue
            name = self.namer.name(frame, [hits[0].contour], pose, tags)[0]
            reads[cell] = ROBOT_NAMES.get(name, name) if self.namer.last_source == "memory" else UNKNOWN
            # How far the blob is from each remembered color, in the memory's own terms: the signature
            # color_calibrate.py learns from, and the per-color minimum ColorMemory.classify takes.
            sig = robot_color.signature(frame, hits[0].contour)
            dist[cell] = {ROBOT_NAMES.get(n, n): min(robot_color.distance(sig, x) for x in samples)
                          for n, samples in memory.colors.items() if samples and n in robot_naming.ACTIVE_COLORS}
        return reads, dist

    def robot_read(self, frame_id: int, cell: str) -> Optional[str]:
        """The robot-memory name of the top block on cell in that settled frame, None if not read."""
        return self._robot_reads.get(frame_id, {}).get(cell)

    def robot_distance(self, frame_id: int, cell: str) -> Optional[dict[str, float]]:
        """Robot-memory distance from that cell's blob to each color in play, None if not measured."""
        return self._robot_dist.get(frame_id, {}).get(cell)

    def _crop(self, frame: np.ndarray, cell: str) -> np.ndarray:
        _, _, size = grid_geometry()
        cx, cy = cell_center_mm(cell)  # perception's cell naming
        half = (0.5 + CROP_MARGIN) * size
        a, b = cx - half, cy - half
        square = [(a, b), (a + 2 * half, b), (a + 2 * half, b + 2 * half), (a, b + 2 * half)]
        pose = self.perceiver.lock.pose()
        pts = np.vstack([pose.to_image(square, 0.0), pose.to_image(square, self.block_mm)])
        h, w = frame.shape[:2]
        u0, v0 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [w, h])
        u1, v1 = np.clip(np.ceil(pts.max(axis=0)).astype(int), 0, [w, h])
        return frame[v0:v1, u0:u1].copy()

    def crop_jpeg(self, frame_id: int, cell: str) -> Optional[bytes]:
        """That cell's crop from a settled frame, as JPEG (what the vision model is shown), None if not kept."""
        crop = self._crops.get(frame_id, {}).get(cell)
        if crop is None or crop.size == 0:
            return None
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jpg.tobytes() if ok else None

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
        robot_naming.ACTIVE_COLORS = self._robot_active
        self.cap.release()
        shutil.rmtree(self.tmp, ignore_errors=True)
