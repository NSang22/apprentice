"""Interactive camera setup and calibration tool for the grid station.

Run this on the prototyping laptop (and again at the demo table) to:
  - find the right camera index and resolution
  - lock exposure / gain / brightness so cube colors stay stable
  - detect the four corner AprilTags (36h11, ids 0-3) and verify grid alignment
  - sample cube colors in each cell to seed color classification thresholds
  - save everything to perception/calib/camera_calib.json

This is a standalone setup tool, not the live perception pipeline. It has no
opinion on `shared/` contracts yet (that module doesn't exist in this repo
yet); once `shared/config.py` lands, the constants below (cell names, colors)
should move there instead of living here.

Usage:
    python -m perception.setup_gui --camera-index 1
    python -m perception.setup_gui --camera-index 1 --backend msmf   # if dshow doesn't take
    python -m perception.setup_gui --camera-index 1 --print-scale 0.975  # if the printed
        # 2.00in ruler mark on the PDF actually measures 1.95in with a real ruler

On startup this probes whether exposure/gain/brightness actually respond on
your camera+backend (many webcams silently ignore them) and prints the
result, so you know up front which controls are real instead of guessing.

GRID GEOMETRY: rather than assuming the corner AprilTags sit exactly at the
grid's outer corners (they don't: see GEOMETRY_IN below), this tool uses the
exact physical layout measured from Apprentice_AprilTag_Grid.pdf (tag
centers and the grid boundary, in inches) via PyMuPDF. The homography maps
each detected tag's pixel position to that tag's *known* physical position,
so the grid overlay lines up with the printed grid even when the printed
grid lines themselves are too washed out (overexposed) for the camera to see.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parent / "calib"
CALIB_PATH = CALIB_DIR / "camera_calib.json"

# README section 3: 0 front-left, 1 front-right, 2 back-right, 3 back-left ("front" = operator side).
CORNER_TAG_IDS = (0, 1, 2, 3)
CORNER_LABELS = ("front-left", "front-right", "back-right", "back-left")
GRID_COLS = ("A", "B", "C")
GRID_ROWS = (1, 2, 3)
CUBE_COLORS = ("red", "blue", "yellow", "green")

# Measured directly from Apprentice_AprilTag_Grid.pdf with PyMuPDF (inches, page
# top-left origin, x right, y down), at the PDF's intended 100% print scale.
# Tag centers do NOT coincide with grid corners: each tag's near edge is flush
# with the grid's edge along one axis and sits a fixed 0.2in gap off it along
# the other (tag is 1.25in square).
TAG_CENTERS_IN = {
    0: (1.5, 1.075),
    1: (7.0, 1.075),
    2: (7.0, 9.475),
    3: (1.5, 9.475),
}
TAG_SIZE_IN = 1.25
GRID_BBOX_IN = (0.875, 1.9, 7.625, 8.65)  # x0, y0, x1, y1
CELL_SIZE_IN = 2.25

DETECT_MAX_DIM = 480  # tag detection runs on a frame downscaled to this, for speed
PPI = 60.0  # pixels per inch for the warped top-down view

# Windows extended key codes from cv2.waitKeyEx(). Arrow keys don't fit in a byte,
# so waitKey()&0xFF can't see them.
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 2490368, 2621440, 2424832, 2555904

# Settings the Tab key cycles through; arrow Up/Down adjusts whichever is active.
ADJUSTABLE = ("exposure", "gain", "brightness")
STEP = {"exposure": 1.0, "gain": 4.0, "brightness": 4.0}
CAM_PROP = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
}
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


@dataclass
class Geometry:
    """Pixel-space layout of the warped top-down view, derived from the PDF's
    physical measurements (scaled by print_scale, to compensate for a printer
    that didn't print at exactly 100%)."""

    warp_w: int
    warp_h: int
    tag_centers_px: dict[int, tuple[float, float]]
    grid_corners_px: list[tuple[float, float]]  # front-left, front-right, back-right, back-left
    grid_bbox_px: tuple[float, float, float, float]  # x0, y0, x1, y1
    cell_size_px: float


def compute_geometry(print_scale: float = 1.0, ppi: float = PPI) -> Geometry:
    s = print_scale
    tag_centers_in = {k: (x * s, y * s) for k, (x, y) in TAG_CENTERS_IN.items()}
    gx0, gy0, gx1, gy1 = (v * s for v in GRID_BBOX_IN)
    tag_size_in = TAG_SIZE_IN * s
    cell_size_in = CELL_SIZE_IN * s

    xs = [gx0, gx1] + [c[0] - tag_size_in / 2 for c in tag_centers_in.values()] + [
        c[0] + tag_size_in / 2 for c in tag_centers_in.values()
    ]
    ys = [gy0, gy1] + [c[1] - tag_size_in / 2 for c in tag_centers_in.values()] + [
        c[1] + tag_size_in / 2 for c in tag_centers_in.values()
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    def to_px(pt: tuple[float, float]) -> tuple[float, float]:
        return ((pt[0] - x_min) * ppi, (pt[1] - y_min) * ppi)

    tag_centers_px = {k: to_px(v) for k, v in tag_centers_in.items()}
    grid_corners_px = [to_px((gx0, gy0)), to_px((gx1, gy0)), to_px((gx1, gy1)), to_px((gx0, gy1))]
    gx0_px, gy0_px = to_px((gx0, gy0))
    gx1_px, gy1_px = to_px((gx1, gy1))

    return Geometry(
        warp_w=int(round((x_max - x_min) * ppi)),
        warp_h=int(round((y_max - y_min) * ppi)),
        tag_centers_px=tag_centers_px,
        grid_corners_px=grid_corners_px,
        grid_bbox_px=(gx0_px, gy0_px, gx1_px, gy1_px),
        cell_size_px=cell_size_in * ppi,
    )


@dataclass
class ExposureState:
    auto: bool = True
    exposure: float = -6.0
    gain: float = 0.0
    brightness: float = 0.0
    active: str = "exposure"

    def value(self, name: str) -> float:
        return getattr(self, name)

    def adjust(self, name: str, delta: float) -> None:
        setattr(self, name, self.value(name) + delta)


@dataclass
class ColorSamples:
    samples: dict = field(default_factory=lambda: {c: [] for c in CUBE_COLORS})
    active: str = "red"
    last: tuple[str, str, tuple[float, float, float]] | None = None  # (cell, color, hsv)
    last_t: float = 0.0


def open_camera(index: int, width: int, height: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep only the newest frame so the preview doesn't fall behind real time.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe_supported(cap: cv2.VideoCapture) -> dict[str, bool]:
    """Empirically check which of exposure/gain/brightness this camera+backend
    actually honors. Many webcams accept cv2's .set() call (sometimes even
    returning True) but silently ignore it: this checks the readback, not the
    return value, and restores the original value afterward."""
    results: dict[str, bool] = {}
    for name, prop in CAM_PROP.items():
        orig = cap.get(prop)
        cap.set(prop, orig - 4.0)
        time.sleep(0.03)
        after = cap.get(prop)
        results[name] = abs(after - orig) > 1e-3
        cap.set(prop, orig)
    return results


def apply_exposure(cap: cv2.VideoCapture, state: ExposureState) -> None:
    # DirectShow convention (varies by driver): 0.75 = auto, 0.25 = manual.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if state.auto else 0.25)
    if not state.auto:
        cap.set(cv2.CAP_PROP_EXPOSURE, state.exposure)
    cap.set(cv2.CAP_PROP_GAIN, state.gain)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, state.brightness)


def detect_corner_tags(
    gray: np.ndarray, detector: "cv2.aruco.ArucoDetector"
) -> dict[int, np.ndarray]:
    scale = min(1.0, DETECT_MAX_DIM / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    corners, ids, _ = detector.detectMarkers(small)
    found: dict[int, np.ndarray] = {}
    if ids is not None:
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            if int(tag_id) in CORNER_TAG_IDS:
                found[int(tag_id)] = tag_corners.reshape(4, 2).mean(axis=0) / scale
    return found


def compute_point_homography(
    src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]]
) -> np.ndarray | None:
    if len(src_points) != 4 or len(dst_points) != 4:
        return None
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst)
    return homography


def build_homography(tag_centers: dict[int, np.ndarray], geom: Geometry) -> np.ndarray | None:
    if not all(tid in tag_centers for tid in CORNER_TAG_IDS):
        return None
    src = [tuple(tag_centers[tid]) for tid in CORNER_TAG_IDS]
    dst = [geom.tag_centers_px[tid] for tid in CORNER_TAG_IDS]
    return compute_point_homography(src, dst)


def project_grid_onto_frame(frame: np.ndarray, homography: np.ndarray, geom: Geometry) -> None:
    """Draw the 3x3 grid lines directly onto the raw camera view, in
    perspective, by projecting them with the inverse homography. This is the
    direct way to check calibration: the drawn lines should sit exactly on
    top of the grid lines printed on the paper (even if those printed lines
    are too washed out for the camera to see on their own)."""
    gx0, gy0, gx1, gy1 = geom.grid_bbox_px
    step = geom.cell_size_px
    segments = []
    for i in range(4):
        x = gx0 + i * step
        segments.append(((x, gy0), (x, gy1)))
    for i in range(4):
        y = gy0 + i * step
        segments.append(((gx0, y), (gx1, y)))
    pts = np.array(segments, dtype=np.float32).reshape(-1, 1, 2)
    inv = np.linalg.inv(homography)
    projected = cv2.perspectiveTransform(pts, inv).reshape(-1, 2, 2)
    for (x1, y1), (x2, y2) in projected:
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)


def draw_grid_overlay(warped: np.ndarray, geom: Geometry) -> np.ndarray:
    out = warped.copy()
    gx0, gy0, gx1, gy1 = (int(v) for v in geom.grid_bbox_px)
    step = geom.cell_size_px
    for i in range(1, 3):
        x = int(gx0 + i * step)
        y = int(gy0 + i * step)
        cv2.line(out, (x, gy0), (x, gy1), (0, 255, 0), 1)
        cv2.line(out, (gx0, y), (gx1, y), (0, 255, 0), 1)
    cv2.rectangle(out, (gx0, gy0), (gx1, gy1), (0, 255, 0), 1)
    for col_idx, col in enumerate(GRID_COLS):
        for row_idx, row in enumerate(GRID_ROWS):
            cx = int(gx0 + (col_idx + 0.5) * step)
            cy = int(gy0 + (row_idx + 0.5) * step)
            cv2.putText(out, f"{col}{row}", (cx - 18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def cell_at(px: float, py: float, geom: Geometry) -> str | None:
    gx0, gy0, gx1, gy1 = geom.grid_bbox_px
    if not (gx0 <= px < gx1 and gy0 <= py < gy1):
        return None
    step = geom.cell_size_px
    col_idx = min(2, int((px - gx0) // step))
    row_idx = min(2, int((py - gy0) // step))
    return f"{GRID_COLS[col_idx]}{GRID_ROWS[row_idx]}"


def mean_hsv_at(warped_bgr: np.ndarray, cell: str, geom: Geometry, sample_frac: float = 0.3) -> tuple[float, float, float]:
    gx0, gy0, _, _ = geom.grid_bbox_px
    step = geom.cell_size_px
    col_idx = GRID_COLS.index(cell[0])
    row_idx = GRID_ROWS.index(int(cell[1]))
    cx = gx0 + (col_idx + 0.5) * step
    cy = gy0 + (row_idx + 0.5) * step
    half = max(1, int(step * sample_frac / 2))
    cx, cy = int(cx), int(cy)
    patch = warped_bgr[cy - half : cy + half, cx - half : cx + half]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean = hsv.mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def save_calibration(
    path: Path,
    camera_index: int,
    width: int,
    height: int,
    print_scale: float,
    homography: np.ndarray | None,
    exposure: ExposureState,
    color_samples: ColorSamples,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "camera_index": camera_index,
        "width": width,
        "height": height,
        "print_scale": print_scale,
        "homography": homography.tolist() if homography is not None else None,
        "exposure": {
            "auto": exposure.auto,
            "exposure": exposure.exposure,
            "gain": exposure.gain,
            "brightness": exposure.brightness,
        },
        "color_samples_hsv_mean": {
            color: (np.mean(vals, axis=0).tolist() if vals else None)
            for color, vals in color_samples.samples.items()
        },
        "saved_at": time.time(),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved calibration to {path}")


def print_controls() -> None:
    print(
        "\n".join(
            [
                "Apprentice camera setup",
                "  q / ESC     quit",
                "  a           toggle auto exposure",
                "  Tab         cycle which setting Up/Down adjusts (exposure / gain / brightness)",
                "  Up / Down   adjust the active setting",
                "  e           open the camera driver's native settings dialog",
                "              (use this if Up/Down has no visible effect; many UVC webcams",
                "               only expose exposure through the driver, not through OpenCV)",
                "  1-4         select color to sample (red, blue, yellow, green)",
                "  click       (in the 'warped' window) sample that cell as the active color",
                "  s           save calibration to perception/calib/camera_calib.json",
                "  c           calibrate corners manually: click the 4 physical grid corners",
                "              on the 'live' window, in order front-left, front-right,",
                "              back-right, back-left (same order as the printed corner tags)",
                "  r           reset manual corners and go back to using the AprilTags",
                "",
                "Tape the grid down, set tags 0-3 at the corners (0 front-left, 1 front-right,",
                "2 back-right, 3 back-left, front = operator side). Yellow lines drawn on the",
                "'live' window are the grid projected from the known PDF geometry, not from",
                "seeing the printed lines: if they don't sit on the real lines, either the",
                "print wasn't at 100% scale (use --print-scale) or the tags moved. Press 'c'",
                "and click the real corners by hand to override directly. Lock exposure under",
                "the actual demo lighting before sampling colors.",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="dshow")
    parser.add_argument(
        "--print-scale", type=float, default=1.0,
        help="actual/intended ratio of the printed 2.00in ruler mark; measure it with a ruler",
    )
    args = parser.parse_args()

    geom = compute_geometry(args.print_scale)
    print(
        f"Grid geometry: {CELL_SIZE_IN * args.print_scale:.3f}in cells, "
        f"warp canvas {geom.warp_w}x{geom.warp_h}px (print_scale={args.print_scale})"
    )

    cap = open_camera(args.camera_index, args.width, args.height, BACKENDS[args.backend])
    exposure = ExposureState()
    apply_exposure(cap, exposure)

    supported = probe_supported(cap)
    print(f"Capability check (backend={args.backend}, camera {args.camera_index}):")
    for name in ADJUSTABLE:
        print(f"  {name:10s} {'OK' if supported[name] else 'NOT controllable from software'}")
    cycle_list = [n for n in ADJUSTABLE if supported[n]] or list(ADJUSTABLE)
    if not any(supported.values()):
        print(
            "None of exposure/gain/brightness respond on this camera+backend.\n"
            "Try: --backend msmf, the 'e' native driver dialog, or a different camera.\n"
            "If the image is overexposed, moving the camera/light is the reliable fix here."
        )
    exposure.active = cycle_list[0]

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    color_samples = ColorSamples()
    state = {
        "tag_homography": None,
        "manual_homography": None,
        "manual_corners": [],
        "warped_bgr": None,
    }

    def on_mouse_warped(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and state["warped_bgr"] is not None:
            cell = cell_at(x, y, geom)
            if cell:
                hsv = mean_hsv_at(state["warped_bgr"], cell, geom)
                color_samples.samples[color_samples.active].append(hsv)
                color_samples.last = (cell, color_samples.active, hsv)
                color_samples.last_t = time.time()
                rounded = tuple(round(v, 1) for v in hsv)
                print(f"Sampled {cell} as {color_samples.active}: HSV={rounded}")

    def on_mouse_live(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(state["manual_corners"]) < 4:
            state["manual_corners"].append((x, y))
            n = len(state["manual_corners"])
            print(f"Picked corner {n}/4 ({CORNER_LABELS[n - 1]}) at ({x}, {y})")
            if n == 4:
                state["manual_homography"] = compute_point_homography(
                    state["manual_corners"], geom.grid_corners_px
                )
                print("Manual calibration set. Press 'r' to go back to AprilTag detection.")
            else:
                print(f"Click corner {n + 1}/4: {CORNER_LABELS[n]}")

    cv2.namedWindow("live")
    cv2.namedWindow("warped")
    cv2.setMouseCallback("live", on_mouse_live)
    cv2.setMouseCallback("warped", on_mouse_warped)
    print_controls()

    fps_t0 = time.time()
    fps_frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag_centers = detect_corner_tags(gray, detector)
            tag_homography = build_homography(tag_centers, geom)
            if tag_homography is not None:
                state["tag_homography"] = tag_homography

            active_homography = (
                state["manual_homography"] if state["manual_homography"] is not None else state["tag_homography"]
            )
            picking = 0 < len(state["manual_corners"]) < 4

            live = frame.copy()
            for tag_id, center in tag_centers.items():
                point = tuple(center.astype(int))
                cv2.circle(live, point, 6, (0, 0, 255), -1)
                cv2.putText(
                    live, str(tag_id), (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            for i, (cx, cy) in enumerate(state["manual_corners"]):
                cv2.circle(live, (cx, cy), 8, (255, 0, 255), -1)
                cv2.putText(live, CORNER_LABELS[i], (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if active_homography is not None:
                project_grid_onto_frame(live, active_homography, geom)

            missing = sorted(set(CORNER_TAG_IDS) - set(tag_centers))
            if state["manual_homography"] is not None:
                banner, banner_color = "GRID LOCKED (manual corners)", (0, 200, 0)
            elif picking:
                n = len(state["manual_corners"])
                banner = f"CALIBRATING... click corner {n + 1}/4: {CORNER_LABELS[n]}"
                banner_color = (255, 0, 255)
            elif tag_homography is not None:
                banner, banner_color = "GRID LOCKED (tags)", (0, 200, 0)
            else:
                banner = f"SEARCHING... missing tags {missing}  (or press 'c' to calibrate by hand)"
                banner_color = (0, 140, 255)
            cv2.rectangle(live, (0, 0), (live.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(live, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)

            active = exposure.active
            readback = cap.get(CAM_PROP[active])
            active_ok = supported[active]
            adjust_line = (
                f"adjusting: {active.upper()} (set {exposure.value(active):.0f}, camera reports {readback:.2f})"
                if active_ok
                else f"adjusting: {active.upper()}  -- NOT SUPPORTED by this camera, press 'e' for driver dialog"
            )
            hud = [
                (f"camera {args.camera_index}  {frame.shape[1]}x{frame.shape[0]}  "
                 f"fps {fps:.1f}  backend {args.backend}", (255, 255, 0)),
                (f"auto-exposure: {exposure.auto}   {adjust_line}", (255, 255, 0) if active_ok else (0, 0, 255)),
                (f"gain: {exposure.gain:.0f} [{'OK' if supported['gain'] else 'unsupported'}]   "
                 f"brightness: {exposure.brightness:.0f} [{'OK' if supported['brightness'] else 'unsupported'}]",
                 (255, 255, 0)),
                (f"sampling color: {color_samples.active}  (click a cell in 'warped' to sample, s to save)",
                 (255, 255, 0)),
            ]
            for i, (line, color) in enumerate(hud):
                cv2.putText(live, line, (10, 56 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            if color_samples.last is not None and time.time() - color_samples.last_t < 2.0:
                cell, color, hsv = color_samples.last
                msg = f"sampled {cell} -> {color}  HSV={tuple(round(v, 1) for v in hsv)}"
                cv2.putText(live, msg, (10, live.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("live", live)

            if active_homography is not None:
                warped = cv2.warpPerspective(frame, active_homography, (geom.warp_w, geom.warp_h))
                state["warped_bgr"] = warped
                cv2.imshow("warped", draw_grid_overlay(warped, geom))

            fps_frames += 1
            if time.time() - fps_t0 >= 0.5:
                fps = fps_frames / (time.time() - fps_t0)
                fps_frames = 0
                fps_t0 = time.time()

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                exposure.auto = not exposure.auto
                apply_exposure(cap, exposure)
            elif key == 9:  # Tab
                idx = cycle_list.index(exposure.active) if exposure.active in cycle_list else -1
                exposure.active = cycle_list[(idx + 1) % len(cycle_list)]
                tag = "" if supported[exposure.active] else " (NOT SUPPORTED on this camera)"
                print(f"Now adjusting: {exposure.active}{tag}")
            elif key == KEY_UP:
                exposure.adjust(exposure.active, STEP[exposure.active])
                apply_exposure(cap, exposure)
            elif key == KEY_DOWN:
                exposure.adjust(exposure.active, -STEP[exposure.active])
                apply_exposure(cap, exposure)
            elif key == ord("e"):
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
            elif key == ord("c"):
                state["manual_corners"] = []
                state["manual_homography"] = None
                print(f"Corner picking started. Click corner 1/4: {CORNER_LABELS[0]} (on 'live').")
            elif key == ord("r"):
                state["manual_corners"] = []
                state["manual_homography"] = None
                print("Reset to AprilTag detection.")
            elif ord("1") <= key <= ord("4"):
                color_samples.active = CUBE_COLORS[key - ord("1")]
                print(f"Now sampling: {color_samples.active}")
            elif key == ord("s"):
                save_calibration(
                    CALIB_PATH, args.camera_index, args.width, args.height, args.print_scale,
                    active_homography, exposure, color_samples,
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
