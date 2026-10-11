"""Set up perception on the real sheet, under the real light: grid lock, empty board, block colors.

    python -m perception.calibrate --camera 1 --block-mm <edge mm>    # Jetson / Linux: also locks exposure
    python -m perception.calibrate --camera 1 --exposure 120          # a fixed exposure from an earlier run

Three steps, prompted in the window:
  1. Clear the board and press SPACE. All 4 AprilTags must be visible. Saves the grid lock
     (perception/calib/grid_lock.json) and the empty board (perception/calib/empty_board.png).
  2. For each color in play (--colors, default robot ACTIVE_COLORS: red, purple, yellow, teal),
     put one block in the middle of B2, the centre cell, and press SPACE. The block is found the
     way perception.publish finds it (perception/blocks.py: changed since the empty board,
     segment.detect, mid-height correction, cubefit). The sample counts once its corrected centre
     sits at least 10 mm inside B2; the block may overhang the lines (it is about a cell wide).
     Over SAMPLE_FRAMES frames the tool reads only the central part of the fitted top face, drops
     highlights (silicone sheen), and records the median CIELAB color and its spread.
  3. Saves perception/calib/colors.json (perception.publish reads it at start-up, the UI reads
     its hex colors) and prints how far apart the colors are.

Keys: SPACE capture or sample, B back one step, Q quit without saving colors.
Rerun it whenever the lighting, the camera, or the sheet changes.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from perception.blocks import EDGE_MARGIN_MM, CHANGED_SHARE, changed_in_image, measure, sheet_roi, top_face_lab
from perception.cells import cell_center_mm, edge_margin_mm, grid_geometry, mm_to_cell
from perception.change import change_mask, warp_to_sheet
from perception.colors import (CALIB_PATH, ColorRef, default_calibration, distances, lab_to_hex, sample_stats,
                               save_calibration)
from perception.perceive import Perceiver
from perception.publish import _text, add_block_args, add_camera_args, capture_frames, image_frames, lock_camera, use_block_size
from robot.perception import segment

CELL = "B2"                  # every color is sampled in the centre cell, clear of the tags
SAMPLE_FRAMES = 8
MIN_SEPARATION = 10.0        # delta E: closer than this, two colors will be confused


def find_block(frame: np.ndarray, perceiver: Perceiver, cell: str = CELL):
    """((top-face polygon px, fitted outline px, footprint as a fraction of a cell), "ok") for the
    block in `cell`, or (None, why not)."""
    pose = perceiver.lock.pose()
    changed = changed_in_image(change_mask(warp_to_sheet(frame, pose.H), perceiver.reference), pose.H, frame.shape)
    best, why = None, f"no block on {cell}"
    for d in segment.detect(frame, None, roi_mask=sheet_roi(pose, frame.shape), pose=pose):
        blob = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(blob, [d.contour.astype(np.int32)], 255)
        area = np.count_nonzero(blob)
        if not area or np.count_nonzero(changed[blob > 0]) < CHANGED_SHARE * area:
            continue
        xy, _, _, outline, top = measure(frame, pose, d.contour, d.center_px, perceiver.block_mm)
        if mm_to_cell(*xy) != cell:
            why = "something else is on the sheet: only this one block, hands away"
            continue
        if edge_margin_mm(*xy) < EDGE_MARGIN_MM:
            why = f"the block's centre is too close to a line of {cell}: move it to the middle"
            continue
        cx, cy = cell_center_mm(cell)
        h = grid_geometry()[2] / 2
        cell_px = cv2.contourArea(pose.to_image([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h),
                                                 (cx - h, cy + h)]).astype(np.float32))
        best = (top, outline, area / max(cell_px, 1.0))
    return (best, "ok") if best else (None, why)


def sample(frames: Iterator[np.ndarray], perceiver: Perceiver) -> tuple[np.ndarray, float]:
    """Central top-face Lab pixels over SAMPLE_FRAMES frames and the median footprint (fraction of a cell)."""
    pixels, fracs, why = [], [], "no frames"
    for _ in range(SAMPLE_FRAMES * 3):
        if len(fracs) == SAMPLE_FRAMES:
            break
        frame = next(frames)
        found, why = find_block(frame, perceiver)
        if found is None:
            continue
        top, _, frac = found
        px = top_face_lab(frame, top)
        if len(px):
            pixels.append(px[:: max(1, len(px) // 4000)])  # at most ~4000 pixels a frame
            fracs.append(frac)
    if len(fracs) < SAMPLE_FRAMES // 2:
        raise ValueError(why)
    return np.concatenate(pixels), float(np.median(fracs))


def overlay(frame: np.ndarray, perceiver: Perceiver, title: str, note: str) -> np.ndarray:
    out = frame.copy()
    if perceiver.lock is not None and perceiver.reference is not None:
        try:
            pose = perceiver.lock.pose()
            cx, cy = cell_center_mm(CELL)
            h = grid_geometry()[2] / 2
            box = pose.to_image([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
            cv2.polylines(out, [box.astype(np.int32)], True, (0, 255, 255), 4)
            found, _ = find_block(frame, perceiver)
            if found is not None:
                top, outline, _ = found
                cv2.polylines(out, [outline.astype(np.int32)], True, (255, 255, 255), 2)
                cv2.polylines(out, [top.astype(np.int32)], True, (0, 255, 0), 2)
        except Exception as e:  # never let the preview kill a calibration run: show it instead
            traceback.print_exc()
            note = f"preview error: {type(e).__name__}: {e}"
    h_img, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 120), (20, 20, 20), -1)
    _text(out, title, (20, 45), 1.1)
    _text(out, "SPACE: go   B: back one step   Q: quit", (20, 90), 0.75)
    if note:
        cv2.rectangle(out, (0, h_img - 50), (w, h_img), (20, 20, 20), -1)
        _text(out, note, (20, h_img - 17), 0.75, (0, 255, 255))
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
    add_block_args(parser)
    parser.add_argument("--out", type=Path, default=CALIB_PATH)
    args = parser.parse_args(argv)
    use_block_size(args)
    colors = args.colors
    print(f"[calibrate] colors to sample, in order: {', '.join(colors)}")

    frames = image_frames(args.images) if args.images else capture_frames(args.camera, loop=False, backend=args.backend)
    if args.camera is not None:
        lock_camera(args, frames)
    # Nominal references only stand in until this run measures the real ones.
    perceiver = Perceiver(default_calibration(), active=colors, block_mm=args.block_mm)
    window = "Apprentice calibration"
    results: dict[str, tuple[ColorRef, float]] = {}
    step, note = -1, ""          # -1: empty board; 0..: colors[step]
    while step < len(colors):
        frame = next(frames)
        title = ("Clear the board (all 4 tags visible), then press SPACE" if step < 0
                 else f"Put {colors[step].upper()} in the middle of {CELL}, then press SPACE")
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
        color = colors[step]
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
    save_calibration(args.out, refs, [f for _, f in results.values()], grid_geometry()[2],
                            meta={"camera": args.camera, "exposure": args.exposure, "wb": args.wb, "cell": CELL,
                                  "colors": colors, "block_mm": args.block_mm})
    print(f"[calibrate] wrote {args.out} ({', '.join(refs)}), block size in use {args.block_mm:g} mm")
    for line in separation_report(refs):
        print(f"[calibrate] {line}")


if __name__ == "__main__":
    main()
