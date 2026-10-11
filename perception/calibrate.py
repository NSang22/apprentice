"""Set up perception on the real sheet, under the real light: grid lock, empty board, block colors.

    python -m perception.calibrate --camera 1                    # Jetson / Linux: also locks exposure
    python -m perception.calibrate --camera 1 --exposure 120     # a fixed exposure from an earlier run

Three steps, prompted in the window:
  1. Clear the board and press SPACE. All 4 AprilTags must be visible. Saves the grid lock
     (perception/calib/grid_lock.json) and the empty board (perception/calib/empty_board.png).
  2. For each color, put one block in the middle of B2, the centre cell (so it never touches a
     tag), and press SPACE. The block is found by what changed since the empty board; a sample
     is rejected unless the whole block sits inside B2. Over SAMPLE_FRAMES frames the tool keeps
     the middle of the block (the top face), drops highlights (silicone sheen: very bright,
     nearly colorless pixels), and records the median CIELAB color and its spread, plus the
     block's footprint as a fraction of a cell.
  3. Saves perception/calib/colors.json (perception.publish reads it at start-up, the UI reads
     its hex colors) and prints how far apart the colors are.

Keys: SPACE capture or sample, B back one step, Q quit without saving colors.
Rerun it whenever the lighting, the camera, or the sheet changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from perception.cells import cell_center_mm, grid_geometry
from perception.change import SHEET_PX_PER_MM, change_mask, sheet_to_image, warp_to_sheet
from perception.colors import (CALIB_PATH, ColorRef, distances, lab_to_hex, load_calibration, sample_stats,
                               save_calibration, to_lab)
from perception.perceive import Perceiver
from perception.publish import _text, add_camera_args, capture_frames, image_frames, lock_camera
from shared.config import COLORS

CELL = "B2"                  # every color is sampled in the centre cell, clear of the tags
SAMPLE_FRAMES = 8
INSIDE_ALLOWANCE_MM = 4.0    # the block's outline may lean this far past B2's lines (its sides)
TOP_FACE_SHRINK = 0.2        # erode the block by this fraction of its width to keep the top face
MIN_SEPARATION = 10.0        # delta E: closer than this, two colors will be confused


def cell_box_mm(cell: str, grow_mm: float = 0.0) -> tuple[float, float, float, float]:
    cx, cy = cell_center_mm(cell)
    h = grid_geometry()[2] / 2.0 + grow_mm
    return cx - h, cy - h, cx + h, cy + h


def find_block(live_sheet: np.ndarray, ref_sheet: np.ndarray, cell: str = CELL) -> tuple[Optional[np.ndarray], str]:
    """(mask of the block in sheet-warp pixels, "ok") or (None, why not). The block is the
    changed region overlapping the cell most, and must lie inside the cell (plus allowance)."""
    mask = change_mask(live_sheet, ref_sheet)
    n, labels = cv2.connectedComponents(mask)
    x0, y0, x1, y1 = (v * SHEET_PX_PER_MM for v in cell_box_mm(cell))
    best, best_overlap = 0, 0
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        overlap = int(np.count_nonzero((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)))
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    if not best:
        return None, f"no block on {cell}"
    blob = np.where(labels == best, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(blob)
    bx0, by0, bx1, by1 = (v * SHEET_PX_PER_MM for v in cell_box_mm(cell, INSIDE_ALLOWANCE_MM))
    if xs.min() < bx0 or xs.max() >= bx1 or ys.min() < by0 or ys.max() >= by1:
        return None, f"the block is not fully inside {cell}: move it to the middle of the square"
    if any(np.count_nonzero(labels == i) > 0.3 * len(xs) for i in range(1, n) if i != best):
        return None, "something else is on the sheet: only this one block, hands away"
    return blob, "ok"


def top_face(blob: np.ndarray) -> np.ndarray:
    width = np.sqrt(np.count_nonzero(blob))
    k = max(3, int(width * TOP_FACE_SHRINK) | 1)
    core = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return core if np.count_nonzero(core) > 50 else blob


def sample(frames: Iterator[np.ndarray], perceiver: Perceiver) -> tuple[np.ndarray, float]:
    """Top-face Lab pixels over SAMPLE_FRAMES frames and the median footprint (fraction of a cell)."""
    pixels, fracs, why = [], [], "no frames"
    cell_px = (grid_geometry()[2] * SHEET_PX_PER_MM) ** 2
    for _ in range(SAMPLE_FRAMES * 3):
        if len(fracs) == SAMPLE_FRAMES:
            break
        live = warp_to_sheet(next(frames), perceiver.lock.H)
        blob, why = find_block(live, perceiver.reference)
        if blob is None:
            continue
        fracs.append(np.count_nonzero(blob) / cell_px)
        core = to_lab(live)[top_face(blob) > 0]
        pixels.append(core[:: max(1, len(core) // 4000)])  # at most ~4000 pixels a frame
    if len(fracs) < SAMPLE_FRAMES // 2:
        raise ValueError(why)
    return np.concatenate(pixels), float(np.median(fracs))


def overlay(frame: np.ndarray, perceiver: Perceiver, title: str, note: str) -> np.ndarray:
    out = frame.copy()
    lock = perceiver.lock
    if lock is not None and perceiver.reference is not None:
        x0, y0, x1, y1 = cell_box_mm(CELL)
        box = lock.pose().to_image([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).astype(np.int32)
        cv2.polylines(out, [box], True, (0, 255, 255), 4)
        live = warp_to_sheet(frame, lock.H)
        blob, _ = find_block(live, perceiver.reference)
        if blob is not None:
            for c in cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                cv2.polylines(out, [sheet_to_image(c.reshape(-1, 2), lock.H).astype(np.int32)], True, (255, 255, 255), 2)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 120), (20, 20, 20), -1)
    _text(out, title, (20, 45), 1.1)
    _text(out, "SPACE: go   B: back one step   Q: quit", (20, 90), 0.75)
    if note:
        cv2.rectangle(out, (0, h - 50), (w, h), (20, 20, 20), -1)
        _text(out, note, (20, h - 17), 0.75, (0, 255, 255))
    return out


def separation_report(refs: dict[str, ColorRef]) -> list[str]:
    """Each color's nearest other color, closest pairs first, flagged when too close."""
    pairs, seen = [], set()
    for name, ref in refs.items():
        d, other = distances(ref.lab, {k: v for k, v in refs.items() if k != name})[0]
        key = tuple(sorted((name, other)))
        if key not in seen:
            seen.add(key)
            pairs.append((d, name, other))
    lines = []
    for d, a, b in sorted(pairs):
        limit = max(MIN_SEPARATION, refs[a].spread + refs[b].spread)
        flag = "  TOO CLOSE: check the lighting, or recalibrate" if d < limit else ""
        lines.append(f"{a:>8} vs {b:<8} dE {d:5.1f}  (spreads {refs[a].spread:.1f} + {refs[b].spread:.1f}){flag}")
    return lines


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, metavar="INDEX")
    source.add_argument("--images", metavar="GLOB", help="offline: cycle through still frames")
    add_camera_args(parser)
    parser.add_argument("--out", type=Path, default=CALIB_PATH)
    args = parser.parse_args(argv)

    frames = image_frames(args.images) if args.images else capture_frames(args.camera, loop=False, backend=args.backend)
    if args.camera is not None:
        lock_camera(args, frames)
    perceiver = Perceiver(load_calibration(args.out))
    window = "Apprentice calibration"
    results: dict[str, tuple[ColorRef, float]] = {}
    step, note = -1, ""          # -1: empty board; 0..5: COLORS[step]
    while step < len(COLORS):
        frame = next(frames)
        title = ("Clear the board (all 4 tags visible), then press SPACE" if step < 0
                 else f"Put {COLORS[step].upper()} in the middle of {CELL}, then press SPACE")
        cv2.imshow(window, overlay(frame, perceiver, title, note))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("[calibrate] quit, colors not saved")
            cv2.destroyAllWindows()
            return
        if key in (ord("b"), ord("B")) and step > -1:
            step, note = step - 1, "went back one step"
            continue
        if key != ord(" "):
            continue
        if step < 0:
            ok, note = perceiver.capture(frame)
            print(f"[calibrate] empty board: {note}")
            step += 1 if ok else 0
            continue
        color = COLORS[step]
        try:
            pixels, frac = sample(frames, perceiver)
            lab, spread, used = sample_stats(pixels)
        except (ValueError, StopIteration) as e:
            note = f"{color}: {e}"
            print(f"[calibrate] {note}")
            continue
        ref = ColorRef(color, lab, spread, lab_to_hex(lab))
        results[color] = (ref, frac)
        print(f"[calibrate] {color:<8} Lab ({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}) {ref.hex} spread {spread:.1f}, "
              f"footprint {frac:.2f} of a cell, {1.0 - used / max(len(pixels), 1):.0%} highlight pixels dropped")
        step, note = step + 1, f"{color} done: take it off"
    cv2.destroyAllWindows()

    refs = {c: r for c, (r, _) in results.items()}
    data = save_calibration(args.out, refs, [f for _, f in results.values()], grid_geometry()[2],
                            meta={"camera": args.camera, "exposure": args.exposure, "wb": args.wb, "cell": CELL})
    print(f"[calibrate] wrote {args.out}: block side about {data['block']['side_mm']} mm, "
          f"footprint {data['block']['area_frac']} of a cell, foreground chroma {data['fg_chroma']}")
    for line in separation_report(refs):
        print(f"[calibrate] {line}")


if __name__ == "__main__":
    main()
