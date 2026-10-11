"""Publish live camera perception as GridState JSON at about 10 Hz.

Examples:
  python -m perception.calibrate --camera 1           # once per setup: grid lock, empty board, block colors
  python -m perception.publish --camera 1 --block-mm <edge mm> --debug   # E re-takes the empty board
  python -m perception.publish --video videos/demo.mp4
  python -m perception.publish --list-cameras         # which index/backend is the real webcam?

The grid is locked once (perception/grid_lock.py): with the board empty and all 4 AprilTags in
view, the homography and the empty board are saved, and from then on blocks covering tags never
lose the grid. Blocks are found with Taher's robot.perception pipeline (perception/blocks.py):
segment.detect blobs that changed since the empty board, mid-height parallax correction, and
cubefit.fit_cube; the corrected centre names the cell. Colors come from
perception/calib/colors.json, among the colors in play (--colors, default robot ACTIVE_COLORS:
red, purple, yellow, teal); anything else reads "unknown". The block size is --block-mm, or the
robot's CUBE_SIZE_MM with a start-up warning.

Per-cell confidence is min(color margin, cube fit). A corrected centre less than 10 mm inside its
cell caps it at 0.5, so a block on a cell line never passes MIN_CONFIDENCE.

hands_clear is false for a hand or an unknown object on the sheet, two blocks in one cell, or a
grid that is not locked (or moved). A tag hidden by a block is never a hand.

On Linux the camera's exposure and white balance are locked at start-up (--exposure, --wb; see
perception/camera_controls.py).

Only one process can open the webcam, and the robot (robot/api.py) needs it too. So publish owns
the camera and shares it: a reader thread keeps only the newest frame (MJPG 1920x1080, like
robot/perception/camera.py), and a small HTTP server on 127.0.0.1:FRAME_PORT serves it:
  GET /frame   the newest raw frame: body = uint8 bytes, headers X-Shape "h,w,c", X-Timestamp
               (time.time() when captured), X-Frame (a counter). 503 before the first frame.
  GET /health  {"ok", "age_s", "frames", "shape"}
Requests only copy out the latest frame under a lock; they never wait on or slow the capture.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from websockets.asyncio.server import ServerConnection, broadcast, serve

from perception import camera_controls
from perception.cells import cell_center_mm, grid_geometry, tag_corner_cells
from perception.colors import CALIB_PATH, load_calibration, parse_colors
from perception.robot_bridge import default_active_colors, set_block_mm
from perception.perceive import FrameResult, Perceiver
from robot.perception.config import FRAME_H, FRAME_W
from robot.perception.sheet import SheetLocalizer, SheetPose
from shared.config import GRID_WS_PORT, MIN_CONFIDENCE

HZ = 10.0
FRAME_PORT = 8771   # local frame server: robot/api.py reads the camera through it (one process owns the webcam)
MIN_TAGS = 3     # --list-cameras: a camera that sees this many tags is pointed at the sheet
BAND_H = 46      # status band height, px
GOOD, BAD, HAND_BGR, CHANGE_BGR = (80, 220, 80), (60, 60, 255), (255, 0, 255), (255, 255, 0)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[5:7], 16), int(hex_color[3:5], 16), int(hex_color[1:3], 16)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def cell_label(cell: str, stack: list[str], confidence: dict[str, float]) -> str:
    """"B2 L2 [red, teal] 0.93": cell, stack height, colors bottom to top, published confidence
    (a "?" when it is below MIN_CONFIDENCE: the brain will read the cell as unknown)."""
    conf = confidence.get(cell, 0.0)
    unsure = "?" if conf < MIN_CONFIDENCE else ""
    return f"{cell} L{len(stack)} [{', '.join(stack)}]{unsure} {conf:.2f}"


def _polyline(img: np.ndarray, pts_px: np.ndarray, color, thickness: int) -> np.ndarray:
    pts = np.asarray(pts_px).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    return pts


def annotate(frame: np.ndarray, result: FrameResult, perceiver: Perceiver, note: str = "",
             speed: str = "", show_render: bool = False) -> np.ndarray:
    """Camera frame with the grid, changed regions (cyan), each occupied cell's visible top face
    (outlined in its color) with "B2 L2 [red, teal] 0.93" or a red "B2 ? <reason>", hands and
    unknown objects in magenta, and (key R) the rendered board the model believes, painted over
    the view. Status sits in its own band on top,
    the hands line and notes in a band at the bottom, so messages never overlap the picture."""
    out = frame.copy()
    lock = perceiver.lock
    if lock is not None and result.grid != "not_locked":
        pose = lock.pose()
        x0, y0, size = grid_geometry()
        for i in range(4):
            for a, b in (((x0 + i * size, y0), (x0 + i * size, y0 + 3 * size)),
                         ((x0, y0 + i * size), (x0 + 3 * size, y0 + i * size))):
                p, q = pose.to_image([a, b]).astype(int)
                cv2.line(out, tuple(p), tuple(q), (0, 200, 0), 2)
        for cell, (cx, cy) in ((c, cell_center_mm(c)) for c in ("A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3")):
            p = pose.to_image([(cx - 0.42 * size, cy - 0.30 * size)])[0].astype(int)
            _text(out, cell, tuple(p), 0.55, (200, 200, 200))
    a = result.analysis
    if a is not None and lock is not None:
        if show_render:  # the board the model believes, painted over the camera view
            sc, cx0, cy0 = a.to_full
            h_c, w_c = int(round(a.render_bgr.shape[0] / sc)), int(round(a.render_bgr.shape[1] / sc))
            big = cv2.resize(a.render_bgr, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
            region = out[cy0:cy0 + h_c, cx0:cx0 + w_c]
            big = big[:region.shape[0], :region.shape[1]]
            painted = big.any(axis=2)
            region[painted] = (0.35 * region[painted] + 0.65 * big[painted]).astype(np.uint8)
        for c in cv2.findContours(a.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            _polyline(out, a.to_full_px(c), CHANGE_BGR, 1)
        refs = perceiver.refs
        for cell, r in a.cells.items():
            if r.state == "empty":
                continue
            color = _bgr(refs[r.color].hex) if r.color in refs else (60, 60, 255)
            for poly in r.top_px:  # the visible part of the top face
                _polyline(out, poly, color, 3)
            stack, why = result.stacks.get(cell, (None, ""))
            x, y = (int(v) for v in r.center_px)
            if stack is None:  # unknown: say why, in red
                _text(out, f"{cell} ? {why}", (x - 70, y - 16), 0.65, (60, 60, 255))
            else:
                _text(out, cell_label(cell, stack, result.state.confidence), (x - 70, y - 16), 0.8)
        for r in a.others:
            pts = _polyline(out, r.contour_px, HAND_BGR, 4)
            x, y = pts.mean(axis=0).astype(int)
            _text(out, r.kind, (x - 30, y), 0.9, HAND_BGR)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, BAND_H), (20, 20, 20), -1)
    cv2.putText(out, result.status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GOOD if result.ok else BAD, 2, cv2.LINE_AA)
    if speed:  # right end of the status band: frames per second and time per frame
        (tw, _), _ = cv2.getTextSize(speed, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(out, speed, (w - tw - 14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, h - BAND_H), (w, h), (20, 20, 20), -1)
    bottom = f"hands_clear={result.state.hands_clear}" + (f" ({result.reason})" if result.reason else "")
    bottom += f"   frame {result.state.frame_id}   E: re-take empty board   R: rendered board   Q: quit"
    cv2.putText(out, bottom + (f"   {note}" if note else ""), (14, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (230, 230, 230), 1, cv2.LINE_AA)
    return out


def debug_lines(result: FrameResult) -> list[str]:
    """The status, then one line per occupied or unknown cell (this frame's reading, what is
    published after stack memory and the 3-frame vote) and per hand."""
    lines = [result.status + (f"; hands_clear false: {result.reason}" if result.reason else "")]
    a = result.analysis
    if a is None:
        return lines
    for cell, r in a.cells.items():
        stack, why = result.stacks.get(cell, (None, ""))
        if r.state == "empty" and stack == []:
            continue
        frame = (f"L{r.level + 1} {r.color} sides={list(r.sides)}" if r.state == "block"
                 else f"{r.state}{': ' + r.reason if r.reason else ''}")
        published = cell_label(cell, stack, result.state.confidence) if stack else f"{cell} ? {why}"
        lines.append(f"{published:<34} frame: {frame} (conf {r.confidence:.2f})")
    for r in a.others:
        lines.append(f"{r.kind:<34} {r.area_mm2:.0f} mm2: {r.why}")
    return lines


def tag_self_check(pose: Optional[SheetPose] = None, frame_shape: Optional[tuple[int, ...]] = None) -> list[str]:
    """Start-up check of the cell naming: which tag sits at which cell corner (from the sheet
    spec), and, with a grid lock, where those corner cells are on screen."""
    lines = [f"tag {tag} sits at the {cell} corner" for tag, cell in sorted(tag_corner_cells().items())]
    if pose is not None and frame_shape is not None:
        h, w = frame_shape[:2]
        for cell in ("A1", "C1", "A3", "C3"):
            u, v = pose.to_image([cell_center_mm(cell)])[0]
            where = f"{'top' if v < h / 2 else 'bottom'}-{'left' if u < w / 2 else 'right'}"
            lines.append(f"{cell} centre on screen at pixel ({u:.0f}, {v:.0f}), {where} of the image")
    return lines


def lock_camera(args: argparse.Namespace, frames: Iterator[np.ndarray]) -> None:
    """Exposure and white balance lock for a live camera (Linux); see perception/camera_controls."""
    if args.camera is None:
        return
    camera_controls.lock_exposure(camera_controls.device(args.camera), args.exposure, args.wb, grab=lambda: next(frames))


def image_frames(pattern: str) -> Iterator[np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"--images matched no files: {pattern}")
    while True:
        for path in paths:
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError(f"could not read image: {path}")
            yield frame


# Camera indexes are per backend, and virtual cameras (Airtime, NVIDIA Broadcast) take slots too,
# so a camera is an (index, backend) pair. --list-cameras shows which pair sees the grid's tags.
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
LIST_MAX_INDEX = 5
LIST_WARMUP_FRAMES = 10   # the first frames from a just-opened camera are often black


def grab_frame(index: int, backend: str, warmup: int = LIST_WARMUP_FRAMES) -> Optional[np.ndarray]:
    """Open one camera, read a few frames, return the last good one, and always release it."""
    capture = cv2.VideoCapture(index, BACKENDS[backend])
    try:
        if not capture.isOpened():
            return None
        frame = None
        for _ in range(warmup):
            ok, f = capture.read()
            if ok and f is not None:
                frame = f
        return frame
    finally:
        capture.release()


def list_cameras(out_dir: Path, max_index: int = LIST_MAX_INDEX) -> list[dict]:
    """Try every index 0..max_index on every backend: resolution, brightness, and which grid
    tags (0-3) it sees. Saves each frame as cam_<index>_<backend>.png in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    rows = []
    for index in range(max_index + 1):
        for backend in BACKENDS:
            frame = grab_frame(index, backend)
            if frame is None:
                rows.append({"index": index, "backend": backend, "opened": False})
                continue
            path = out_dir / f"cam_{index}_{backend}.png"
            cv2.imwrite(str(path), frame)
            rows.append({
                "index": index, "backend": backend, "opened": True,
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "brightness": round(float(frame.mean()), 1),
                "tags": sorted(localizer.detect_tags(frame)),
                "frame": str(path),
            })
    return rows


def print_camera_table(rows: list[dict]) -> None:
    print(f"{'index':<6}{'backend':<8}{'resolution':<12}{'bright':<8}{'tags':<14}frame")
    for r in rows:
        if not r["opened"]:
            print(f"{r['index']:<6}{r['backend']:<8}{'(no camera)':<12}")
            continue
        tags = ",".join(map(str, r["tags"])) or "-"
        print(f"{r['index']:<6}{r['backend']:<8}{r['resolution']:<12}{r['brightness']:<8}{tags:<14}{r['frame']}")
    grid = [r for r in rows if r["opened"] and len(r["tags"]) >= MIN_TAGS]
    if grid:
        best = max(grid, key=lambda r: len(r["tags"]))
        print(f"sees the grid: --camera {best['index']} --backend {best['backend']}")
    else:
        print(f"no camera sees {MIN_TAGS}+ grid tags: check the camera points at the sheet")


class FrameShare:
    """The newest captured frame, for the local frame server. put() swaps a reference under a
    lock; get() hands it out. Neither waits on the other for long."""

    def __init__(self) -> None:
        self.frame: Optional[np.ndarray] = None
        self.t, self.count = 0.0, 0
        self.lock = threading.Lock()

    def put(self, frame: np.ndarray, t: Optional[float] = None) -> None:
        with self.lock:
            self.frame, self.t = frame, time.time() if t is None else t
            self.count += 1

    def get(self) -> tuple[Optional[np.ndarray], float, int]:
        with self.lock:
            return self.frame, self.t, self.count


def serve_frames(share: FrameShare, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the frame server in a daemon thread (see the module docstring)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            frame, t, n = share.get()
            if path == "/health":
                body = json.dumps({"ok": frame is not None, "age_s": None if frame is None else round(time.time() - t, 3),
                                   "frames": n, "shape": None if frame is None else list(frame.shape)}).encode()
                self._send(200, body, "application/json")
            elif path == "/frame":
                if frame is None:
                    self._send(503, b"no frame yet", "text/plain")
                    return
                self._send(200, np.ascontiguousarray(frame).tobytes(), "application/octet-stream",
                           {"X-Shape": ",".join(map(str, frame.shape)), "X-Dtype": "uint8",
                            "X-Timestamp": repr(t), "X-Frame": str(n)})
            else:
                self._send(404, b"not found", "text/plain")

        def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *args) -> None:
            pass  # quiet: the robot asks for frames all the time

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def live_camera_frames(index: int, backend: Optional[str] = None, share: Optional[FrameShare] = None) -> Iterator[np.ndarray]:
    """The overhead camera, newest frame only. A reader thread keeps grabbing (as
    robot/perception/camera.py does), so a slow perception loop never works on frames the driver
    buffered a while ago, and the frame server always has the newest one. Each new frame is
    yielded once."""
    capture = cv2.VideoCapture(index, BACKENDS[backend]) if backend else cv2.VideoCapture(index)
    if not capture.isOpened():
        raise ValueError(f"could not open camera {index}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    latest: dict = {"frame": None, "n": 0}
    cond = threading.Condition()
    running = [True]

    def reader() -> None:
        while running[0]:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            if share is not None:
                share.put(frame)
            with cond:
                latest["frame"], latest["n"] = frame, latest["n"] + 1
                cond.notify_all()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    seen = 0
    try:
        while True:
            with cond:
                cond.wait_for(lambda: latest["n"] > seen, timeout=5.0)
                if latest["n"] == seen:
                    raise ValueError(f"camera {index} stopped delivering frames")
                frame, seen = latest["frame"], latest["n"]
            yield frame
    finally:
        running[0] = False
        thread.join(timeout=2)
        capture.release()


def capture_frames(source: int | str, loop: bool, backend: Optional[str] = None) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(source, BACKENDS[backend]) if backend else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"could not open frame source: {source}")
    try:
        while True:
            ok, frame = capture.read()
            if ok:
                yield frame
                continue
            if loop:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            return
    finally:
        capture.release()


class Publisher:
    def __init__(self) -> None:
        self.clients: set[ServerConnection] = set()

    async def handler(self, websocket: ServerConnection) -> None:
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)


async def run(args: argparse.Namespace) -> None:
    frames: Iterator[np.ndarray]
    share = FrameShare()
    if args.images:
        frames = image_frames(args.images)
    elif args.video:
        frames = capture_frames(args.video, loop=True)
    else:
        frames = live_camera_frames(args.camera, args.backend, share)
        lock_camera(args, frames)
    if args.frame_port:
        serve_frames(share, args.frame_port)
        print(f"[perception] sharing the newest frame on http://127.0.0.1:{args.frame_port}/frame "
              "(robot/api.py reads the camera through it)", flush=True)
    publisher = Publisher()
    calib = load_calibration(args.calib)
    if calib.calibrated:
        print(f"[perception] block colors from {calib.source} ({calib.meta.get('created', 'undated')})", flush=True)
    else:
        print(f"[perception] NOT CALIBRATED: {args.calib} is missing, using nominal block colors. "
              "Run python -m perception.calibrate first.", flush=True)
    perceiver = Perceiver(calib, log=lambda m: print(m, flush=True), active=args.colors, block_mm=args.block_mm)
    print(f"[perception] colors in play: {', '.join(perceiver.active)} (anything else reads unknown)", flush=True)
    if perceiver.lock is None:
        print("[perception] no grid lock yet: clear the board, then press E in the --debug window "
              "(or run perception.calibrate)", flush=True)
    else:
        print(f"[perception] saved grid lock from {perceiver.lock.created}: waiting for 2 tags to confirm it", flush=True)
    for line in tag_self_check(perceiver.lock.pose() if perceiver.lock else None,
                               (perceiver.lock.frame_size[1], perceiver.lock.frame_size[0]) if perceiver.lock else None):
        print(f"[perception] self-check: {line}", flush=True)
    period = 1.0 / HZ
    note = ""
    show_render = False
    fps, step_ms, last_t = 0.0, 0.0, None   # smoothed over ~10 frames
    async with serve(publisher.handler, args.host, args.port):
        print(f"[perception] publishing GridState on ws://{args.host}:{args.port} at {HZ:g} Hz", flush=True)
        last_report: list[str] = []
        for frame_id, frame in enumerate(frames):
            started = time.monotonic()
            if args.camera is None:
                share.put(frame)  # video or images: share what perception sees (the live camera shares in its reader)
            result = perceiver.step(frame, frame_id)
            took = time.monotonic() - started
            step_ms = took * 1000 if not step_ms else 0.9 * step_ms + 0.1 * took * 1000
            if last_t is not None:
                rate = 1.0 / max(started - last_t, 1e-6)
                fps = rate if not fps else 0.9 * fps + 0.1 * rate
            last_t = started
            if args.debug:
                report = debug_lines(result)
                if report != last_report:  # print on change, not 10 times a second
                    print(f"[perception] frame {frame_id}:", flush=True)
                    for line in report:
                        print("  " + line, flush=True)
                    last_report = report
            # "source", "grid" and "tags" are extra keys GridState.from_dict ignores. brain.server records
            # "source" as provenance; scripts/check.py shows "grid" (not_locked, not_confirmed, locked, lost)
            # and "tags" (tag ids seen this frame).
            broadcast(publisher.clients, json.dumps({**result.state.to_dict(), "source": "perception",
                                                     "grid": result.grid, "tags": result.tags}))
            if args.debug:
                cv2.imshow("Apprentice perception",
                           annotate(frame, result, perceiver, note, f"{fps:.1f} fps   {step_ms:.0f} ms/frame",
                                    show_render))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    show_render = not show_render
                if key in (ord("e"), ord("E")):
                    ok, note = perceiver.capture(frame)
                    print(f"[perception] {'E: ' + note if ok else 'E failed: ' + note}", flush=True)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
    cv2.destroyAllWindows()


def add_block_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-mm", type=float, default=None,
                        help="block edge in mm for perception (default robot/perception/config.py CUBE_SIZE_MM)")
    parser.add_argument("--colors", type=parse_colors, default=default_active_colors(),
                        help="comma-separated colors in play (default robot ACTIVE_COLORS: "
                             + ",".join(default_active_colors()) + "); any other color reads unknown")


def use_block_size(args: argparse.Namespace) -> None:
    """Put the block size in use for robot.perception's segment/cubefit and say where it came from."""
    args.block_mm, line = set_block_mm(args.block_mm)
    print(f"[perception] {line}", flush=True)


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=None,
                        help="capture backend for --camera on Windows (indexes differ per backend)")
    parser.add_argument("--exposure", default="auto",
                        help="Linux: 'auto' (step down until the sheet is not clipped, the default), "
                             "a number (exposure_time_absolute), or 'keep'")
    parser.add_argument("--wb", type=int, default=None,
                        help="Linux: white balance temperature in kelvin (default: auto off at its current value)")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, metavar="INDEX")
    source.add_argument("--video", metavar="PATH")
    source.add_argument("--images", metavar="GLOB")
    source.add_argument("--list-cameras", action="store_true",
                        help="probe camera indexes on each backend, report resolution and grid tags seen, and exit")
    add_camera_args(parser)
    add_block_args(parser)
    parser.add_argument("--snapshot-dir", type=Path, default=Path(tempfile.gettempdir()) / "apprentice_cameras",
                        help="where --list-cameras saves one frame per camera")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=GRID_WS_PORT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--frame-port", type=int, default=FRAME_PORT,
                        help="serve the newest raw frame on 127.0.0.1:PORT for robot/api.py (0 = off)")
    parser.add_argument("--calib", type=Path, default=CALIB_PATH, help="block color calibration from perception.calibrate")
    args = parser.parse_args(argv)
    if args.list_cameras:
        print_camera_table(list_cameras(args.snapshot_dir))
        return
    use_block_size(args)
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, ValueError) as exc:
        if str(exc):
            print(f"[perception] {exc}")


if __name__ == "__main__":
    main()
