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
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from websockets.asyncio.server import ServerConnection, broadcast, serve

from perception import camera_controls
from perception.cells import cell_center_mm, grid_geometry, tag_corner_cells
from perception.blocks import Block
from perception.change import sheet_to_image
from perception.colors import CALIB_PATH, load_calibration, parse_colors
from perception.robot_bridge import default_active_colors, set_block_mm
from perception.perceive import FrameResult, Perceiver
from robot.perception.sheet import SheetLocalizer, SheetPose
from shared.config import GRID_WS_PORT, MIN_CONFIDENCE, UNKNOWN

HZ = 10.0
MIN_TAGS = 3     # --list-cameras: a camera that sees this many tags is pointed at the sheet
BAND_H = 46      # status band height, px
GOOD, BAD, HAND_BGR, CHANGE_BGR = (80, 220, 80), (60, 60, 255), (255, 0, 255), (255, 255, 0)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[5:7], 16), int(hex_color[3:5], 16), int(hex_color[1:3], 16)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def block_label(block: Block, confidence: dict[str, float]) -> str:
    """"B2 teal 0.93", with a "?" when the published confidence is below MIN_CONFIDENCE, so the
    Jetson window says what the brain will see (the brain treats that cell as unknown)."""
    if block.cell is None:
        return f"off-grid {block.color}"
    conf = confidence.get(block.cell, 0.0)
    unsure = "?" if conf < MIN_CONFIDENCE or block.color == UNKNOWN else ""
    return f"{block.cell} {block.color}{unsure} {conf:.2f}"


def _polyline(img: np.ndarray, pts_px: np.ndarray, color, thickness: int) -> np.ndarray:
    pts = np.asarray(pts_px).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    return pts


def annotate(frame: np.ndarray, result: FrameResult, perceiver: Perceiver, note: str = "",
             speed: str = "") -> np.ndarray:
    """Camera frame with the grid, changed regions (cyan), each block's blob (thin white), fitted
    cube outline (its color), corrected centre (dot) and "cell color confidence" label, and hands
    and unknown objects in magenta. Status sits in its own band on top,
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
        contours, _ = cv2.findContours(a.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            _polyline(out, sheet_to_image(c.reshape(-1, 2), lock.H), CHANGE_BGR, 1)
        refs = perceiver.refs
        for b in a.blocks:
            _polyline(out, b.contour_px, (255, 255, 255), 1)
            _polyline(out, b.outline_px, _bgr(refs[b.color].hex) if b.color in refs else (128, 128, 128), 3)
            _polyline(out, b.top_px, (255, 255, 255), 1)
            x, y = (int(v) for v in b.center_px)
            cv2.circle(out, (x, y), 7, (0, 0, 0), -1)
            cv2.circle(out, (x, y), 5, (255, 255, 255), -1)
            stack, why = result.stacks.get(b.cell, (None, ""))
            if b.cell is not None and stack is None:  # unknown: say why, in red
                _text(out, f"{b.cell} ? {why}", (x - 70, y - 16), 0.7, (60, 60, 255))
            else:
                _text(out, stack_label(b, stack, result.state.confidence), (x - 70, y - 16), 0.8)
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
    bottom += f"   frame {result.state.frame_id}   E: re-take empty board   Q: quit"
    cv2.putText(out, bottom + (f"   {note}" if note else ""), (14, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (230, 230, 230), 1, cv2.LINE_AA)
    return out


def stack_label(block: Block, stack: Optional[list[str]], confidence: dict[str, float]) -> str:
    """"B2 L2 [red, teal] 0.93": cell, stack height, colors bottom to top, confidence."""
    if block.cell is None or not stack:
        return block_label(block, confidence)
    conf = confidence.get(block.cell, 0.0)
    unsure = "?" if conf < MIN_CONFIDENCE else ""
    return f"{block.cell} L{len(stack)} [{', '.join(stack)}]{unsure} {conf:.2f}"


def debug_lines(result: FrameResult) -> list[str]:
    """The status, then one line per block (label, the three scores, color distances) and per hand."""
    lines = [result.status + (f"; hands_clear false: {result.reason}" if result.reason else "")]
    a = result.analysis
    if a is None:
        return lines
    for b in sorted(a.blocks, key=lambda b: (b.cell or "~", b.color)):
        stack, why = result.stacks.get(b.cell, (None, ""))
        label = stack_label(b, stack, result.state.confidence) if stack else f"{b.cell} ? {why}"
        fits = "/".join(f"{o:.2f}" for o in b.overlaps)
        sides = " ".join(f"{c or '-'}({n}px)" for c, n in zip(b.sides, b.side_px))
        lines.append(f"{label:<30} centre ({b.center_mm[0]:.0f}, {b.center_mm[1]:.0f}) mm edge={b.edge_mm:.1f}mm "
                     f"fit L1/L2/L3={fits} sides=[{sides}] margin={b.margin_score:.2f} "
                     f"(dE {b.d_nearest:.1f} vs {b.d_second:.1f})")
    for r in a.others:
        lines.append(f"{r.kind:<22} {r.area_mm2:.0f} mm2: {r.why}")
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
    if args.images:
        frames = image_frames(args.images)
    elif args.video:
        frames = capture_frames(args.video, loop=True)
    else:
        frames = capture_frames(args.camera, loop=False, backend=args.backend)
        lock_camera(args, frames)
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
    fps, step_ms, last_t = 0.0, 0.0, None   # smoothed over ~10 frames
    async with serve(publisher.handler, args.host, args.port):
        print(f"[perception] publishing GridState on ws://{args.host}:{args.port} at {HZ:g} Hz", flush=True)
        last_report: list[str] = []
        for frame_id, frame in enumerate(frames):
            started = time.monotonic()
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
            # "source" is an extra key GridState.from_dict ignores; brain.server records it as provenance.
            broadcast(publisher.clients, json.dumps({**result.state.to_dict(), "source": "perception"}))
            if args.debug:
                cv2.imshow("Apprentice perception",
                           annotate(frame, result, perceiver, note, f"{fps:.1f} fps   {step_ms:.0f} ms/frame"))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
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
