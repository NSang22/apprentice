"""Measure the real blocks' colors and size on the real sheet, under the real light.

    python -m perception.calibrate --camera 1                 # the overhead camera
    python -m perception.calibrate --camera 1 --backend dshow # if indexes differ per backend

With the grid in view, the window asks for one block at a time ("Put RED on A1, then press
SPACE"). Put that block in the middle of the outlined cell, keep hands out of the picture, and
press SPACE. The tool watches SAMPLE_FRAMES frames, finds the block's blob in that cell, keeps
the middle of it (the top face; the sides look darker), drops highlights (silicone sheen:
very bright, nearly colorless pixels), and records the median CIELAB color and its spread.
Take each block off again before the next one: touching blocks merge into one blob.

Keys: SPACE sample, B back one color, Q quit without saving.

Writes perception/calib/colors.json (perception.publish reads it at start-up, the UI reads its
hex colors), then prints how far apart the colors are. Rerun it whenever the lighting changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from perception.cells import cell_center_mm, grid_geometry
from perception.colors import (CALIB_PATH, ColorRef, distances, foreground, lab_to_hex, sample_stats,
                               save_calibration, specular, to_lab)
from perception.publish import (MIN_TAGS, MORPH_KERNEL, _text, capture_frames, cell_area_px, grid_roi_mask,
                                image_frames)
from robot.perception.sheet import SheetLocalizer, SheetPose
from shared.config import COLORS

# One block per color, in this order and these cells. Look-alike pairs (teal/skyblue,
# orange/red/yellow) are never neighbours, so a block's blob can't run into its look-alike.
SEQUENCE = [("red", "A1"), ("teal", "B1"), ("yellow", "C1"), ("purple", "A2"), ("orange", "B2"), ("skyblue", "C2")]
SAMPLE_FRAMES = 8
CAL_FG_CHROMA = 12.0     # before any reference exists: anything this colorful may be a block
MIN_BLOB_FRAC = 0.15     # a blob smaller than this fraction of the cell is not a block
CELL_GROW = 1.3          # read only inside the cell grown by this factor (a block's sides lean out)
TOP_FACE_SHRINK = 0.2    # erode the blob by this fraction of its width to keep the top face
MIN_SEPARATION = 10.0    # delta E: closer than this, two colors will be confused


def cell_polygon_px(pose: SheetPose, cell: str, grow: float = 1.0) -> np.ndarray:
    cx, cy = cell_center_mm(cell)
    h = grid_geometry()[2] / 2.0 * grow
    square = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    return pose.to_image(square).astype(np.int32)


def find_block(lab: np.ndarray, pose: SheetPose, cell: str, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    """uint8 mask of the colorful blob that overlaps the cell most, or None."""
    fg = foreground(lab, grid_roi_mask(pose, shape), CAL_FG_CHROMA)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, MORPH_KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, MORPH_KERNEL)
    cell_mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(cell_mask, [cell_polygon_px(pose, cell)], 255)
    near = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(near, [cell_polygon_px(pose, cell, CELL_GROW)], 255)
    fg &= near
    n, labels = cv2.connectedComponents(fg)
    best, best_overlap = None, 0
    for i in range(1, n):
        overlap = int(np.count_nonzero((labels == i) & (cell_mask > 0)))
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    if best is None:
        return None
    blob = np.where(labels == best, 255, 0).astype(np.uint8)
    cx, cy = cell_center_mm(cell)
    if np.count_nonzero(blob) < MIN_BLOB_FRAC * cell_area_px(pose, cx, cy):
        return None
    return blob


def top_face(blob: np.ndarray) -> np.ndarray:
    width = np.sqrt(np.count_nonzero(blob))
    k = max(3, int(width * TOP_FACE_SHRINK) | 1)
    core = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return core if np.count_nonzero(core) > 50 else blob


def sample(frames: Iterator[np.ndarray], localizer: SheetLocalizer, cell: str) -> tuple[np.ndarray, float]:
    """Top-face Lab pixels over SAMPLE_FRAMES frames, and the median footprint fraction."""
    pixels, fracs = [], []
    for _ in range(SAMPLE_FRAMES * 3):
        if len(fracs) == SAMPLE_FRAMES:
            break
        frame = next(frames)
        pose = localizer.locate(frame) if len(localizer.detect_tags(frame)) >= MIN_TAGS else None
        if pose is None:
            continue
        lab = to_lab(frame)
        blob = find_block(lab, pose, cell, frame.shape)
        if blob is None:
            continue
        cx, cy = cell_center_mm(cell)
        fracs.append(np.count_nonzero(blob) / cell_area_px(pose, cx, cy))
        core = lab[top_face(blob) > 0]
        pixels.append(core[:: max(1, len(core) // 4000)])  # at most ~4000 pixels a frame
    if not fracs:
        raise ValueError(f"no block found on {cell}: is it in the middle of the outlined cell, with 3+ tags in view?")
    return np.concatenate(pixels), float(np.median(fracs))


def overlay(frame: np.ndarray, pose: Optional[SheetPose], color: str, cell: str, note: str) -> np.ndarray:
    out = frame.copy()
    if pose is None:
        _text(out, "GRID NOT FOUND: get 3+ tags in view", (30, 60), 1.1, (0, 0, 255))
    else:
        cv2.polylines(out, [cell_polygon_px(pose, cell)], True, (0, 255, 255), 4)
        blob = find_block(to_lab(frame), pose, cell, frame.shape)
        if blob is not None:
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
            out[(specular(to_lab(frame)) & (blob > 0))] = (255, 0, 255)  # highlights left out, shown magenta
    _text(out, f"Put {color.upper()} on {cell}, then press SPACE", (30, 60), 1.3)
    _text(out, "Only this one block on the sheet: take the last one off first", (30, 185), 0.8)
    _text(out, "B: back one color   Q: quit without saving", (30, 105), 0.8)
    if note:
        _text(out, note, (30, 145), 0.8, (0, 255, 255))
    return out


def separation_report(refs: dict[str, ColorRef]) -> list[str]:
    """Each color's nearest other color, closest pairs first, flagged when too close."""
    lines, seen = [], set()
    pairs = []
    for name, ref in refs.items():
        others = {k: v for k, v in refs.items() if k != name}
        d, other = distances(ref.lab, others)[0]
        key = tuple(sorted((name, other)))
        if key not in seen:
            seen.add(key)
            pairs.append((d, name, other))
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
    parser.add_argument("--backend", choices=["dshow", "msmf"], default=None)
    parser.add_argument("--out", type=Path, default=CALIB_PATH)
    args = parser.parse_args(argv)
    assert sorted(c for c, _ in SEQUENCE) == sorted(COLORS), "SEQUENCE must cover every color once"

    frames = image_frames(args.images) if args.images else capture_frames(args.camera, loop=False, backend=args.backend)
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    window = "Apprentice calibration"
    results: dict[str, tuple[ColorRef, float]] = {}
    i, note = 0, ""
    while i < len(SEQUENCE):
        color, cell = SEQUENCE[i]
        frame = next(frames)
        pose = localizer.locate(frame) if len(localizer.detect_tags(frame)) >= MIN_TAGS else None
        cv2.imshow(window, overlay(frame, pose, color, cell, note))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("[calibrate] quit, nothing saved")
            cv2.destroyAllWindows()
            return
        if key in (ord("b"), ord("B")) and i > 0:
            i, note = i - 1, "went back one color"
            continue
        if key != ord(" "):
            continue
        try:
            pixels, frac = sample(frames, localizer, cell)
            lab, spread, used = sample_stats(pixels)
        except (ValueError, StopIteration) as e:
            note = str(e)
            print(f"[calibrate] {color} on {cell}: {e}")
            continue
        highlight = 1.0 - used / max(len(pixels), 1)
        ref = ColorRef(color, lab, spread, lab_to_hex(lab))
        results[color] = (ref, frac)
        print(f"[calibrate] {color:<8} on {cell}: Lab ({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}) {ref.hex} "
              f"spread {spread:.1f}, footprint {frac:.2f} of the cell, {highlight:.0%} highlight pixels dropped")
        i, note = i + 1, f"{color} done"
    cv2.destroyAllWindows()

    refs = {c: r for c, (r, _) in results.items()}
    data = save_calibration(args.out, refs, [f for _, f in results.values()], grid_geometry()[2],
                            meta={"camera": args.camera, "backend": args.backend,
                                  "cells": {c: cell for c, cell in SEQUENCE}})
    print(f"[calibrate] wrote {args.out}: block side about {data['block']['side_mm']} mm, "
          f"footprint {data['block']['area_frac']} of a cell, foreground chroma {data['fg_chroma']}")
    for line in separation_report(refs):
        print(f"[calibrate] {line}")


if __name__ == "__main__":
    main()
