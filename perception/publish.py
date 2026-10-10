"""Publish live camera perception as GridState JSON at about 10 Hz.

Examples:
  python -m perception.publish --camera 1 --debug
  python -m perception.publish --video videos/demo.mp4
  python -m perception.publish --images "robot/samples/overhead_*.png" --debug
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

from perception.cells import DEFAULT_SHEET_SPEC, mm_to_cell
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


@dataclass
class CubeDetection:
    color: str
    center_px: tuple[float, float]
    contour: np.ndarray
    score: float


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
    for color in ("red", "blue", "yellow", "green"):
        mask = _mask_for_color(frame_bgr, color, roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            (_, _), (width, height), _ = cv2.minAreaRect(contour)
            detections.append(CubeDetection(
                color=color,
                center_px=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                contour=contour.reshape(-1, 2),
                score=float(min(1.0, area / max(width * height, 1.0))),
            ))
            cv2.drawContours(combined, [contour], -1, 255, thickness=cv2.FILLED)
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
        for detection in sorted(detections, key=lambda d: d.score, reverse=True):
            x_mm, y_mm = pose.to_sheet([detection.center_px], mid_height)[0]
            cell = mm_to_cell(float(x_mm), float(y_mm))
            if cell is None:
                continue
            if board[cell]:
                hands_clear = False
                confidence[cell] = min(confidence[cell], detection.score)
                continue
            board[cell] = [detection.color]
            confidence[cell] = detection.score
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
        for frame_id, frame in enumerate(frames):
            started = time.monotonic()
            state, last_pose, detections = build_grid_state(frame, localizer, last_pose, frame_id)
            broadcast(publisher.clients, json.dumps(state.to_dict()))
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
