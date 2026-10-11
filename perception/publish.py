"""Publish live camera perception as GridState JSON at about 10 Hz.

Examples:
  python -m perception.calibrate --camera 1           # once per lighting setup: measures the blocks
  python -m perception.publish --camera 1 --debug
  python -m perception.publish --video videos/demo.mp4
  python -m perception.publish --images "robot/samples/overhead_*.png" --debug
  python -m perception.publish --list-cameras        # which index/backend is the real webcam?

Blocks are found and named with the calibration in perception/calib/colors.json (see
perception/colors.py): colorful, non-highlight pixels inside the grid are labeled by their
nearest reference color in CIELAB, so two touching blocks of different colors split apart,
and each blob is named by the nearest reference to its median Lab ("unknown" past the cutoff).
Without a calibration file nominal colors are used and the start-up banner says so.

Per-cell confidence is the minimum of three scores in [0, 1] for the block's blob:
  area    1.0 when the blob covers the footprint calibration measured for one block (as a
          fraction of the cell's projected area, with headroom), falling off outside that
          range, so two merged blocks or a sliver score low
  margin  1.0 when the second-nearest reference color is at least MARGIN_FULL delta E farther
          than the nearest, 0.0 when the blob is equally close to two colors (teal vs skyblue)
  fill    fill ratio (blob area / min-area rectangle) rescaled so >= 0.5 maps to 1.0
A block whose centre sits less than EDGE_MARGIN_MM inside its cell gets at most 0.5, so a
block straddling two cells never passes MIN_CONFIDENCE. --debug labels every block with its
cell, color, and confidence, and prints the scores.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
from websockets.asyncio.server import ServerConnection, broadcast, serve

from perception.cells import cell_center_mm, edge_margin_mm, grid_geometry, mm_to_cell, tag_corner_cells
from perception.colors import CALIB_PATH, Calibration, classify, foreground, label_pixels, load_calibration, to_lab
from robot.perception.sheet import SheetLocalizer, SheetPose
from shared.config import CELLS, COLORS, GRID_WS_PORT, MIN_CONFIDENCE, UNKNOWN
from shared.schemas import GridState, empty_board

HZ = 10.0
MIN_TAGS = 3
MIN_CUBE_AREA_PX = 1200
HAND_AREA_FRACTION = 0.08
SPLIT_OF_BLOCK = 0.5       # a color region this fraction of a block's smallest footprint counts as its own block
HAND_OF_MAX_AREA = 2.5     # a colorful blob this many times a block's largest footprint is a hand or arm

# Confidence scoring (see module docstring).
FILL_FULL = 0.5            # fill ratio at or above this scores 1.0
EDGE_MARGIN_MM = 8.0       # block centre must sit this far inside its cell
STRADDLE_CONFIDENCE = 0.5  # confidence cap for a block too close to a cell boundary
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def fill_score(fill_ratio: float) -> float:
    return max(0.0, min(1.0, fill_ratio / FILL_FULL))


def area_score(area_px: float, expected: tuple[float, float]) -> float:
    lo, hi = expected
    if area_px < lo:
        return max(0.0, area_px / lo)
    if area_px > hi:
        return hi / area_px
    return 1.0


def cell_area_px(pose: SheetPose, x_mm: float, y_mm: float) -> float:
    """Pixel area of one cell-sized square on the table, centred at sheet (x_mm, y_mm)."""
    h = grid_geometry()[2] / 2.0
    square = [(x_mm - h, y_mm - h), (x_mm + h, y_mm - h), (x_mm + h, y_mm + h), (x_mm - h, y_mm + h)]
    return float(cv2.contourArea(pose.to_image(square).astype(np.float32)))


def expected_area_px(pose: SheetPose, x_mm: float, y_mm: float, calib: Calibration) -> tuple[float, float]:
    """(min, max) pixel area for one block centred at sheet (x_mm, y_mm): the calibrated
    footprint fractions times this spot's projected cell area."""
    cell_px = cell_area_px(pose, x_mm, y_mm)
    return calib.area_frac[0] * cell_px, calib.area_frac[2] * cell_px


@dataclass
class CubeDetection:
    color: str
    center_px: tuple[float, float]
    contour: np.ndarray
    score: float              # min of the component scores (before the cell-edge rule)
    area_px: float = 0.0
    fill_ratio: float = 0.0
    lab: tuple[float, float, float] = (0.0, 0.0, 0.0)  # median Lab of the blob
    d_nearest: float = 0.0    # delta E to the nearest reference color
    d_second: float = 0.0     # ... and to the second nearest
    fill_score: float = 0.0
    margin_score: float = 0.0
    area_score: float = 1.0   # set once the sheet pose gives an expected area
    cell: Optional[str] = None
    edge_mm: float = 0.0


def detect_all_cubes(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray,
    calib: Calibration,
    min_area_px: int = MIN_CUBE_AREA_PX,
    min_block_px: Optional[float] = None,
) -> tuple[list[CubeDetection], np.ndarray]:
    """Every block-sized colorful blob in the roi, named by its nearest reference color.

    Blobs come from the color-agnostic foreground first. A blob is split by pixel color only
    when two or more colors each cover at least SPLIT_OF_BLOCK of the smallest block footprint
    (min_block_px; touching blocks). Smaller patches of another color (a shaded side, an edge)
    stay part of the block. Pixels far from every color split off only as a whole block."""
    if min_block_px is None:
        min_block_px = 4.0 * min_area_px
    lab = to_lab(frame_bgr)
    fg = foreground(lab, roi_mask, calib.fg_chroma)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, MORPH_KERNEL)  # fills highlight holes and raised letters
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, MORPH_KERNEL)   # drops specks
    labels = label_pixels(lab, fg, calib.refs)
    n, components = cv2.connectedComponents(fg)
    detections: list[CubeDetection] = []
    for i in range(1, n):
        component = components == i
        if np.count_nonzero(component) < min_area_px:
            continue
        counts = np.bincount(labels[component].astype(np.int64) + 1, minlength=len(COLORS) + 2)[1:]
        need = [SPLIT_OF_BLOCK * min_block_px] * len(COLORS) + [min_block_px]  # last: far from every color
        major = [k for k, c in enumerate(counts) if c >= need[k]]
        if len(major) <= 1:
            parts = [component.astype(np.uint8) * 255]
        else:
            parts = [cv2.morphologyEx(((labels == k) & component).astype(np.uint8) * 255, cv2.MORPH_OPEN, MORPH_KERNEL)
                     for k in major]
        for part in parts:
            detections += _blobs(part, lab, fg, calib, min_area_px)
    return detections, fg


def _blobs(mask: np.ndarray, lab: np.ndarray, fg: np.ndarray, calib: Calibration, min_area_px: int) -> list[CubeDetection]:
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if area < min_area_px or moments["m00"] == 0:
            continue
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
        pixels = lab[(blob > 0) & (fg > 0)]
        median = tuple(float(v) for v in np.median(pixels, axis=0)) if len(pixels) else (0.0, 0.0, 0.0)
        color, margin, d1, d2 = classify(median, calib.refs)
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        fill = float(min(1.0, area / max(width * height, 1.0)))
        d = CubeDetection(
            color=color,
            center_px=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
            contour=contour.reshape(-1, 2),
            score=0.0,
            area_px=float(area),
            fill_ratio=fill,
            lab=median,
            d_nearest=d1,
            d_second=d2,
            fill_score=fill_score(fill),
            margin_score=margin,
        )
        d.score = min(d.fill_score, d.margin_score)
        out.append(d)
    return out


def grid_roi_mask(pose: SheetPose, shape: tuple[int, ...]) -> np.ndarray:
    x0, y0, cell_mm = grid_geometry()
    size = 3 * cell_mm
    polygon = pose.to_image([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def large_non_cube_blob(frame_bgr: np.ndarray, grid_mask: np.ndarray, cube_mask: np.ndarray) -> bool:
    """Conservatively flag a large saturated or dark object over the grid."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    foreground_px = np.where(((hsv[:, :, 1] > 45) | (hsv[:, :, 2] < 105)), 255, 0).astype(np.uint8)
    foreground_px &= grid_mask
    foreground_px[cv2.dilate(cube_mask, np.ones((21, 21), np.uint8)) > 0] = 0
    foreground_px = cv2.morphologyEx(foreground_px, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(foreground_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_area = max(float(cv2.countNonZero(grid_mask)), 1.0)
    return any(cv2.contourArea(c) / grid_area >= HAND_AREA_FRACTION for c in contours)


def build_grid_state(
    frame_bgr: np.ndarray,
    localizer: SheetLocalizer,
    last_pose: SheetPose | None,
    frame_id: int,
    timestamp: float | None = None,
    calib: Optional[Calibration] = None,
) -> tuple[GridState, SheetPose | None, list[CubeDetection]]:
    calib = calib or load_calibration()
    found_tags = localizer.detect_tags(frame_bgr)
    current_pose = localizer.locate(frame_bgr) if len(found_tags) >= MIN_TAGS else None
    pose = current_pose or last_pose
    board = empty_board()
    confidence = {cell: 0.0 for cell in CELLS}
    detections: list[CubeDetection] = []
    hands_clear = current_pose is not None

    if pose is not None:
        roi = grid_roi_mask(pose, frame_bgr.shape)
        min_block_px = calib.area_frac[0] * cv2.countNonZero(roi) / 9.0
        blobs, cube_mask = detect_all_cubes(frame_bgr, roi, calib, min_block_px=min_block_px)
        mid_height = calib.side_mm / 2.0
        for detection in blobs:
            x_mm, y_mm = (float(v) for v in pose.to_sheet([detection.center_px], mid_height)[0])
            lo, hi = expected_area_px(pose, x_mm, y_mm, calib)
            if detection.area_px > HAND_OF_MAX_AREA * hi:
                hands_clear = False  # far bigger than any block: a hand, a sleeve, or the arm
                continue
            detection.cell = mm_to_cell(x_mm, y_mm)
            detection.edge_mm = edge_margin_mm(x_mm, y_mm)
            detection.area_score = area_score(detection.area_px, (lo, hi))
            detection.score = min(detection.area_score, detection.margin_score, detection.fill_score)
            detections.append(detection)
        for detection in sorted(detections, key=lambda d: d.score, reverse=True):
            cell = detection.cell
            if cell is None:
                continue
            conf = detection.score
            if detection.edge_mm < EDGE_MARGIN_MM:
                conf = min(conf, STRADDLE_CONFIDENCE)  # may straddle two cells: never a pass
            if board[cell]:
                hands_clear = False
                confidence[cell] = min(confidence[cell], conf)
                continue
            board[cell] = [detection.color]
            confidence[cell] = conf
        if current_pose is not None:
            hands_clear = hands_clear and not large_non_cube_blob(frame_bgr, roi, cube_mask)
            for cell in CELLS:
                if not board[cell]:
                    confidence[cell] = 1.0

    state = GridState(
        t=time.time() if timestamp is None else timestamp,
        board=board,
        arm_clear=True,
        hands_clear=hands_clear,
        confidence=confidence,
        frame_id=frame_id,
    )
    return state, pose, detections


def _bgr(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[5:7], 16), int(hex_color[3:5], 16), int(hex_color[1:3], 16)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def block_label(detection: CubeDetection, state: GridState) -> str:
    """"B2 teal 0.93", with a "?" when the published confidence is below MIN_CONFIDENCE, so the
    Jetson window says what the brain will see (the brain treats that cell as unknown)."""
    if detection.cell is None:
        return f"off-grid {detection.color}"
    conf = state.confidence.get(detection.cell, 0.0)
    unsure = "?" if conf < MIN_CONFIDENCE or detection.color == UNKNOWN else ""
    return f"{detection.cell} {detection.color}{unsure} {conf:.2f}"


def annotate(frame: np.ndarray, pose: SheetPose | None, detections: list[CubeDetection], state: GridState,
             calib: Optional[Calibration] = None) -> np.ndarray:
    out = frame.copy()
    if pose is None:
        _text(out, "GRID NOT FOUND", (30, 55), 1.3, (0, 0, 255))
        return out
    x0, y0, size = grid_geometry()
    for i in range(4):
        vertical = pose.to_image([(x0 + i * size, y0), (x0 + i * size, y0 + 3 * size)])
        horizontal = pose.to_image([(x0, y0 + i * size), (x0 + 3 * size, y0 + i * size)])
        cv2.line(out, tuple(vertical[0].astype(int)), tuple(vertical[1].astype(int)), (0, 255, 0), 2)
        cv2.line(out, tuple(horizontal[0].astype(int)), tuple(horizontal[1].astype(int)), (0, 255, 0), 2)
    for cell in CELLS:
        cx, cy = cell_center_mm(cell)
        corner = pose.to_image([(cx - 0.42 * size, cy - 0.30 * size)])[0].astype(int)
        _text(out, cell, tuple(corner), 0.6, (200, 200, 200))
    refs = (calib or load_calibration()).refs
    for detection in detections:
        hex_color = refs[detection.color].hex if detection.color in refs else "#808080"
        cv2.drawContours(out, [detection.contour.reshape(-1, 1, 2)], -1, _bgr(hex_color), 3)
        cx, cy = (int(v) for v in detection.center_px)
        cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
        _text(out, block_label(detection, state), (cx - 60, cy - 18), 0.8)
    _text(out, f"hands_clear={state.hands_clear} frame={state.frame_id}", (30, 35), 0.8)
    return out


def debug_lines(detections: list[CubeDetection], state: GridState) -> list[str]:
    """One line per block: label, the three component scores, and the color distances."""
    lines = []
    for d in sorted(detections, key=lambda d: (d.cell or "~", d.color)):
        lines.append(
            f"{block_label(d, state):<22} area={d.area_score:.2f} ({d.area_px:.0f}px) "
            f"margin={d.margin_score:.2f} (dE {d.d_nearest:.1f} vs {d.d_second:.1f}) "
            f"fill={d.fill_score:.2f} ({d.fill_ratio:.2f}) edge={d.edge_mm:.1f}mm"
        )
    return lines


def tag_self_check(pose: Optional[SheetPose] = None, frame_shape: Optional[tuple[int, ...]] = None) -> list[str]:
    """Start-up check of the cell naming: which tag sits at which cell corner (from the sheet
    spec), and, once the sheet is in view, where those corner cells are on screen."""
    lines = [f"tag {tag} sits at the {cell} corner" for tag, cell in sorted(tag_corner_cells().items())]
    if pose is not None and frame_shape is not None:
        h, w = frame_shape[:2]
        for cell in ("A1", "C1", "A3", "C3"):
            u, v = pose.to_image([cell_center_mm(cell)])[0]
            where = f"{'top' if v < h / 2 else 'bottom'}-{'left' if u < w / 2 else 'right'}"
            lines.append(f"{cell} centre on screen at pixel ({u:.0f}, {v:.0f}), {where} of the image")
    return lines


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
    frames: Iterable[np.ndarray]
    if args.images:
        frames = image_frames(args.images)
    elif args.video:
        frames = capture_frames(args.video, loop=True)
    else:
        frames = capture_frames(args.camera, loop=False, backend=args.backend)
    publisher = Publisher()
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    calib = load_calibration(args.calib)
    if calib.calibrated:
        print(f"[perception] block colors from {calib.source} ({calib.meta.get('created', 'undated')})", flush=True)
    else:
        print(f"[perception] NOT CALIBRATED: {args.calib} is missing, using nominal block colors. "
              "Run python -m perception.calibrate first.", flush=True)
    for line in tag_self_check():
        print(f"[perception] self-check: {line}", flush=True)
    last_pose: SheetPose | None = None
    checked_screen = False
    period = 1.0 / HZ
    async with serve(publisher.handler, args.host, args.port):
        print(f"[perception] publishing GridState on ws://{args.host}:{args.port} at {HZ:g} Hz", flush=True)
        last_report: list[str] = []
        for frame_id, frame in enumerate(frames):
            started = time.monotonic()
            state, last_pose, detections = build_grid_state(frame, localizer, last_pose, frame_id, calib=calib)
            if last_pose is not None and not checked_screen:
                checked_screen = True  # once the sheet is in view: where the corner cells are on screen
                for line in tag_self_check(last_pose, frame.shape)[4:]:
                    print(f"[perception] self-check: {line}", flush=True)
            if args.debug:
                report = debug_lines(detections, state)
                if report != last_report:  # print on change, not 10 times a second
                    print(f"[perception] frame {frame_id}: {len(detections)} cube(s)", flush=True)
                    for line in report:
                        print("  " + line, flush=True)
                    last_report = report
            # "source" is an extra key GridState.from_dict ignores; brain.server records it as provenance.
            broadcast(publisher.clients, json.dumps({**state.to_dict(), "source": "perception"}))
            if args.debug:
                cv2.imshow("Apprentice perception", annotate(frame, last_pose, detections, state, calib))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
    cv2.destroyAllWindows()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, metavar="INDEX")
    source.add_argument("--video", metavar="PATH")
    source.add_argument("--images", metavar="GLOB")
    source.add_argument("--list-cameras", action="store_true",
                        help="probe camera indexes on each backend, report resolution and grid tags seen, and exit")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=None,
                        help="capture backend for --camera (indexes differ per backend)")
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
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, ValueError) as exc:
        if str(exc):
            print(f"[perception] {exc}")


if __name__ == "__main__":
    main()
