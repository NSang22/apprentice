"""Publish live camera perception as GridState JSON at about 10 Hz.

Examples:
  python -m perception.publish --camera 1 --debug
  python -m perception.publish --video videos/demo.mp4
  python -m perception.publish --images "robot/samples/overhead_*.png" --debug

Per-cell confidence is the minimum of three scores in [0, 1] for the cube's blob:
  area  1.0 when the blob area fits one cube at this spot (the 48 mm cube projected through
        the sheet pose: from 0.6x its top face up to 1.15x its full silhouette), falling off
        as the ratio outside that range, so two merged cubes or a sliver score low
  hue   1.0 when the blob's median hue sits at least HUE_MARGIN inside its color band,
        lower near a band edge, and at most 0.5 if that hue also falls in another color's band
  fill  fill ratio (blob area / min-area rectangle) rescaled so >= 0.5 maps to 1.0; cubes
        seen at an angle show their sides and fill only 0.58-0.73, which is fine
A cube whose centre sits less than EDGE_MARGIN_MM inside its cell gets at most 0.5, so a
cube straddling two cells never passes MIN_CONFIDENCE. --debug prints the scores per cube.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
from websockets.asyncio.server import ServerConnection, broadcast, serve

from perception.cells import DEFAULT_SHEET_SPEC, edge_margin_mm, mm_to_cell
from robot.perception import detect_color
from robot.perception.config import CUBE_SIZE_MM
from robot.perception.sheet import SheetLocalizer, SheetPose
from shared.config import CELLS, GRID_WS_PORT
from shared.schemas import GridState, empty_board

HZ = 10.0
MIN_TAGS = 3
MIN_CUBE_AREA_PX = 1200
HAND_AREA_FRACTION = 0.08
GREEN_RANGES = [((36, 55, 45), (90, 255, 255))]

# Confidence scoring (see module docstring).
AREA_MIN_OF_TOP = 0.6      # smallest believable blob: this fraction of the cube's top face
AREA_MAX_OF_HULL = 1.15    # largest: this multiple of the cube's full projected silhouette
HUE_MARGIN = 4.0           # OpenCV hue units (0-179) inside the band edge for a full hue score
FILL_FULL = 0.5            # fill ratio at or above this scores 1.0
EDGE_MARGIN_MM = 8.0       # cube centre must sit this far inside its cell
STRADDLE_CONFIDENCE = 0.5  # confidence cap for a cube too close to a cell boundary


def hue_bands() -> dict[str, list[tuple[float, float]]]:
    """Color -> hue intervals (OpenCV 0-179), from the same ranges the masks use."""
    bands = {name.removesuffix("_cube"): [(float(lo[0]), float(hi[0])) for lo, hi in ranges]
             for name, ranges in detect_color.HSV_RANGES.items()}
    bands["green"] = [(float(lo[0]), float(hi[0])) for lo, hi in GREEN_RANGES]
    return bands


def merge_wrapped(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    """One circular interval (start, length) for a color's hue ranges. Red's 170-179 and
    0-8 join across the wrap into start 170, length 18."""
    if len(intervals) == 1:
        lo, hi = intervals[0]
        return lo, hi - lo
    (lo_a, hi_a), (lo_b, hi_b) = sorted(intervals)[:1] + sorted(intervals)[-1:]
    if lo_a <= 0.0 and hi_b >= 179.0:
        return lo_b, (180.0 - lo_b) + hi_a
    return lo_a, hi_b - lo_a


def band_depth(hue: float, band: tuple[float, float]) -> float:
    """How far (hue units) a hue sits inside a circular band; negative when outside."""
    start, length = band
    offset = (hue - start) % 180.0
    if offset <= length:
        return min(offset, length - offset)
    return -min(offset - length, 180.0 - offset)


def circular_median_hue(hues: np.ndarray, center: float) -> float:
    """Median hue measured around `center`, so a red blob straddling 0/179 stays red."""
    delta = (np.asarray(hues, np.float64) - center + 90.0) % 180.0 - 90.0
    if delta.size == 0:
        return float(center)
    return float((center + np.median(delta)) % 180.0)


def hue_score(color: str, median_hue: float, bands: Optional[dict] = None) -> float:
    bands = bands or hue_bands()
    score = max(0.0, min(1.0, band_depth(median_hue, merge_wrapped(bands[color])) / HUE_MARGIN))
    for other, ranges in bands.items():
        if other != color and band_depth(median_hue, merge_wrapped(ranges)) >= 0.0:
            score = min(score, 0.5)  # this hue could be either color
    return score


def fill_score(fill_ratio: float) -> float:
    return max(0.0, min(1.0, fill_ratio / FILL_FULL))


def area_score(area_px: float, expected: tuple[float, float]) -> float:
    lo, hi = expected
    if area_px < lo:
        return max(0.0, area_px / lo)
    if area_px > hi:
        return hi / area_px
    return 1.0


def expected_area_px(pose: SheetPose, x_mm: float, y_mm: float,
                     cube_mm: float = CUBE_SIZE_MM or 48.0) -> tuple[float, float]:
    """(min, max) pixel area for one cube centred at sheet (x_mm, y_mm): its top face times
    AREA_MIN_OF_TOP, and the convex hull of its 8 projected corners (the silhouette a pinhole
    camera sees, sides included) times AREA_MAX_OF_HULL."""
    h = cube_mm / 2.0
    square = [(x_mm - h, y_mm - h), (x_mm + h, y_mm - h), (x_mm + h, y_mm + h), (x_mm - h, y_mm + h)]
    top = pose.to_image(square, cube_mm).astype(np.float32)
    bottom = pose.to_image(square, 0.0).astype(np.float32)
    hull = cv2.convexHull(np.vstack([top, bottom]))
    return AREA_MIN_OF_TOP * float(cv2.contourArea(top)), AREA_MAX_OF_HULL * float(cv2.contourArea(hull))


@dataclass
class CubeDetection:
    color: str
    center_px: tuple[float, float]
    contour: np.ndarray
    score: float              # min of the component scores (before the cell-edge rule)
    area_px: float = 0.0
    fill_ratio: float = 0.0
    median_hue: float = 0.0
    fill_score: float = 0.0
    hue_score: float = 0.0
    area_score: float = 1.0   # set once the sheet pose gives an expected area
    cell: Optional[str] = None
    edge_mm: float = 0.0


def _mask_for_color(frame_bgr: np.ndarray, color: str, roi_mask: np.ndarray) -> np.ndarray:
    if color != "green":
        return detect_color.color_mask(frame_bgr, f"{color}_cube", roi_mask)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in GREEN_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask &= roi_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_all_cubes(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray,
    min_area_px: int = MIN_CUBE_AREA_PX,
) -> tuple[list[CubeDetection], np.ndarray]:
    """Return every qualifying color blob, rather than one blob per color."""
    detections: list[CubeDetection] = []
    combined = np.zeros(frame_bgr.shape[:2], np.uint8)
    hue = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    bands = hue_bands()
    for color in ("red", "blue", "yellow", "green"):
        mask = _mask_for_color(frame_bgr, color, roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        start, length = merge_wrapped(bands[color])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            (_, _), (width, height), _ = cv2.minAreaRect(contour)
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
            median = circular_median_hue(hue[(blob > 0) & (mask > 0)], start + length / 2.0)
            fill = float(min(1.0, area / max(width * height, 1.0)))
            d = CubeDetection(
                color=color,
                center_px=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                contour=contour.reshape(-1, 2),
                score=0.0,
                area_px=float(area),
                fill_ratio=fill,
                median_hue=median,
                fill_score=fill_score(fill),
                hue_score=hue_score(color, median, bands),
            )
            d.score = min(d.fill_score, d.hue_score)
            detections.append(d)
            combined |= blob
    return detections, combined


def grid_roi_mask(pose: SheetPose, shape: tuple[int, ...]) -> np.ndarray:
    spec = json.loads(DEFAULT_SHEET_SPEC.read_text(encoding="utf-8"))["grid"]
    x0, y0 = spec["origin_mm"]
    size = 3 * spec["cell_mm"]
    polygon = pose.to_image([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def large_non_cube_blob(frame_bgr: np.ndarray, grid_mask: np.ndarray, cube_mask: np.ndarray) -> bool:
    """Conservatively flag a large saturated or dark object over the grid."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    foreground = np.where(((hsv[:, :, 1] > 45) | (hsv[:, :, 2] < 105)), 255, 0).astype(np.uint8)
    foreground &= grid_mask
    foreground[cv2.dilate(cube_mask, np.ones((21, 21), np.uint8)) > 0] = 0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_area = max(float(cv2.countNonZero(grid_mask)), 1.0)
    return any(cv2.contourArea(c) / grid_area >= HAND_AREA_FRACTION for c in contours)


def build_grid_state(
    frame_bgr: np.ndarray,
    localizer: SheetLocalizer,
    last_pose: SheetPose | None,
    frame_id: int,
    timestamp: float | None = None,
) -> tuple[GridState, SheetPose | None, list[CubeDetection]]:
    found_tags = localizer.detect_tags(frame_bgr)
    current_pose = localizer.locate(frame_bgr) if len(found_tags) >= MIN_TAGS else None
    pose = current_pose or last_pose
    board = empty_board()
    confidence = {cell: 0.0 for cell in CELLS}
    detections: list[CubeDetection] = []
    hands_clear = current_pose is not None

    if pose is not None:
        roi = grid_roi_mask(pose, frame_bgr.shape)
        detections, cube_mask = detect_all_cubes(frame_bgr, roi)
        mid_height = (CUBE_SIZE_MM or 0.0) / 2.0
        for detection in detections:
            x_mm, y_mm = (float(v) for v in pose.to_sheet([detection.center_px], mid_height)[0])
            detection.cell = mm_to_cell(x_mm, y_mm)
            detection.edge_mm = edge_margin_mm(x_mm, y_mm)
            detection.area_score = area_score(detection.area_px, expected_area_px(pose, x_mm, y_mm))
            detection.score = min(detection.area_score, detection.hue_score, detection.fill_score)
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


def annotate(frame: np.ndarray, pose: SheetPose | None, detections: list[CubeDetection], state: GridState) -> np.ndarray:
    out = frame.copy()
    if pose is None:
        cv2.putText(out, "GRID NOT FOUND", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        return out
    spec = json.loads(DEFAULT_SHEET_SPEC.read_text(encoding="utf-8"))["grid"]
    x0, y0, size = *spec["origin_mm"], spec["cell_mm"]
    for i in range(4):
        vertical = pose.to_image([(x0 + i * size, y0), (x0 + i * size, y0 + 3 * size)])
        horizontal = pose.to_image([(x0, y0 + i * size), (x0 + 3 * size, y0 + i * size)])
        cv2.line(out, tuple(vertical[0].astype(int)), tuple(vertical[1].astype(int)), (0, 255, 0), 2)
        cv2.line(out, tuple(horizontal[0].astype(int)), tuple(horizontal[1].astype(int)), (0, 255, 0), 2)
    for col in range(3):
        for row in range(3):
            cell = f"{'ABC'[col]}{row + 1}"
            point = pose.to_image([(x0 + (col + 0.5) * size, y0 + (row + 0.5) * size)])[0].astype(int)
            cv2.putText(out, cell, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
            cv2.putText(out, cell, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    colors = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 220, 255), "green": (0, 180, 0)}
    for detection in detections:
        cv2.drawContours(out, [detection.contour.reshape(-1, 1, 2)], -1, colors[detection.color], 3)
        cv2.drawMarker(out, tuple(np.asarray(detection.center_px, int)), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    status = f"hands_clear={state.hands_clear} frame={state.frame_id}"
    cv2.putText(out, status, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, status, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def debug_lines(detections: list[CubeDetection], state: GridState) -> list[str]:
    """One line per cube: cell, color, the three component scores, and the published confidence."""
    lines = []
    for d in sorted(detections, key=lambda d: (d.cell or "~", d.color)):
        final = state.confidence.get(d.cell, 0.0) if d.cell else 0.0
        lines.append(
            f"{d.cell or 'off-grid':<8} {d.color:<6} area={d.area_score:.2f} ({d.area_px:.0f}px) "
            f"hue={d.hue_score:.2f} (H {d.median_hue:.1f}) fill={d.fill_score:.2f} ({d.fill_ratio:.2f}) "
            f"edge={d.edge_mm:.1f}mm final={final:.2f}"
        )
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


def capture_frames(source: int | str, loop: bool) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(source)
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
        frames = capture_frames(args.camera, loop=False)
    publisher = Publisher()
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    last_pose: SheetPose | None = None
    period = 1.0 / HZ
    async with serve(publisher.handler, args.host, args.port):
        print(f"[perception] publishing GridState on ws://{args.host}:{args.port} at {HZ:g} Hz", flush=True)
        last_report: list[str] = []
        for frame_id, frame in enumerate(frames):
            started = time.monotonic()
            state, last_pose, detections = build_grid_state(frame, localizer, last_pose, frame_id)
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
                cv2.imshow("Apprentice perception", annotate(frame, last_pose, detections, state))
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
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=GRID_WS_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, ValueError) as exc:
        if str(exc):
            print(f"[perception] {exc}")


if __name__ == "__main__":
    main()
