"""Teach the camera this setup's block colours by position. About 30 s, no internet.

Put each colour on its spot (only the colours you have), then run:

    python -m robot.scripts.calibration.color_calibrate
    python -m robot.scripts.calibration.color_calibrate --layout "A1=red,B2=dark green"
    python -m robot.scripts.calibration.color_calibrate --image robot/outputs/snapshots/X_raw.png

Default spots keep blocks apart (blocks pushed together into neighbouring cells merged into one
blob, 17:15). Row A is nearest the arm. For the current kit (config.BLOCK_COLORS, in spot order):

    A1 orange       .          A3 yellow
          .     B2 purple          .
    C1 teal      C2 red      C3 light blue

Centre each block in its cell. Six blocks on a 3x3 grid can't all avoid neighbours, so C2 (the
only spot with neighbours) gets a colour unlike all of them: similar colours that touch merge and
can't be learned (teal next to light blue, 17:24).

Any grid cell (A1-C3) can be used in --layout. One photo; each block is remembered as its spot's
colour for this lighting, and naming then matches against these instead of guessing. Colours not
found this time keep their previous calibration, so one misplaced block doesn't wipe the rest.
"""

import argparse
import json
import time

import cv2
import numpy as np

from robot.perception import color
from robot.perception.camera import Camera
from robot.perception.config import BLOCK_COLORS, ROOT, SHEET_SPEC
from robot.perception.naming import normalize
from robot.perception.scene import annotate, build_scene
from robot.perception.sheet import SheetLocalizer

SPOT_ORDER = ["A1", "A3", "B2", "C1", "C2", "C3"]  # checkerboard + C2; only C2 has neighbours
SPOT_RADIUS_MM = 28.0  # a block counts for a spot if its centre is within this (half a cell)
WARN_HUE_DEG = 45.0    # a block this far from its colour's usual hue may be on the wrong spot


def spot_centers(grid):
    o, s = grid["origin_mm"], grid["cell_mm"]
    return {f"{'ABC'[r]}{c + 1}": (o[0] + (c + 0.5) * s, o[1] + (r + 0.5) * s) for r in range(3) for c in range(3)}


DEFAULT_LAYOUT = dict(zip(SPOT_ORDER, BLOCK_COLORS))


def nearest_spot(xy, spots):
    name = min(spots, key=lambda k: np.hypot(xy[0] - spots[k][0], xy[1] - spots[k][1]))
    d = float(np.hypot(xy[0] - spots[name][0], xy[1] - spots[name][1]))
    return (name, d) if d <= SPOT_RADIUS_MM else (None, d)


def parse_layout(text, spots):
    layout = {}
    for item in text.split(","):
        spot, _, name = item.partition("=")
        spot, name = spot.strip().upper(), normalize(name.strip().strip('"'))
        if spot not in spots or not name:
            raise SystemExit(f"bad layout entry {item!r}; use e.g. A1=red (cells A1-C3)")
        layout[spot] = name
    return layout


def print_layout(layout):
    w = 16
    cell = lambda s: f"{s} {layout[s]}" if s in layout else "."
    for r in "ABC":
        print("   " + "".join(f"{cell(f'{r}{c}'):<{w}}" for c in "123"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", help='e.g. "A1=red,B2=dark green" (replaces the default layout)')
    ap.add_argument("--image", help="use a saved raw frame instead of the camera")
    args = ap.parse_args()
    grid = json.loads(open(SHEET_SPEC).read())["grid"]
    spots = spot_centers(grid)
    layout = parse_layout(args.layout, spots) if args.layout else DEFAULT_LAYOUT

    print("Layout (row A is nearest the arm; centre each block in its cell):")
    print_layout(layout)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"can't read {args.image}")
    else:
        with Camera() as cam:
            frame = cam.read()

    snaps = ROOT / "outputs" / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.image:
        cv2.imwrite(str(snaps / f"{stamp}_color_calibration_raw.png"), frame)  # kept even if this run fails

    scene, pose, dets = build_scene(frame, SheetLocalizer(), naming="off")
    if scene is None:
        raise SystemExit("Sheet not found: need at least 2 tags visible.")
    where = [nearest_spot(o["center_mm"], spots) for o in scene["objects"]]
    print(f"\nFound {len(dets)} block(s) on the sheet: " + ", ".join(
        f"{s or 'no spot'} ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f})" for (s, _), o in zip(where, scene["objects"])))

    found, problems = {}, []
    for d, o, (spot, dist) in zip(dets, scene["objects"], where):
        at = f"({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm"
        if o["score"] < 0:
            problems.append(f"blocks touching near {spot or at}; spread them out")
        elif spot is None:
            problems.append(f"a block at {at} isn't on any spot ({dist:.0f} mm from the nearest)")
        elif spot in found:
            problems.append(f"two blocks on {spot}")
        else:
            found[spot] = d
    for s in found:
        if s not in layout:
            problems.append(f"a block on {s}, but the layout has no colour for {s}")
    if problems:
        print("\nProblems:\n  " + "\n  ".join(problems))
    taught = {s: layout[s] for s in found if s in layout}
    if not taught:
        raise SystemExit("No blocks found on layout spots; nothing learned.")

    memory = color.ColorMemory()
    kept = sorted(n for n in memory.colors if n not in taught.values())
    print("\nLearned:")
    for spot, name in sorted(taught.items()):
        sig = color.signature(frame, found[spot].contour)
        memory.colors[name] = []  # this lighting's sample replaces the colour's old ones
        memory.learn(name, sig)
        note = ""
        if name in color.PRIOR and sig["chroma"] >= color.LOW_CHROMA:
            off = abs((sig["hue"] - color.PRIOR[name] + 180) % 360 - 180)
            if off > WARN_HUE_DEG:
                note = f"   <- unusual for {name} ({off:.0f} deg from typical); is the right block on {spot}?"
        print(f"  {spot}  {name:12s} hue {sig['hue']:5.1f}  chroma {sig['chroma']:4.1f}  L {sig['L']:4.1f}{note}")
    names = sorted(memory.colors)
    for i, a in enumerate(names):  # colours too similar to tell apart reliably in this lighting
        for b in names[i + 1:]:
            d = color.distance(memory.colors[a][0], memory.colors[b][0])
            if d < 2 * color.MIN_MARGIN:
                print(f"  note: {a} and {b} look very similar here (distance {d:.0f}); they may get confused")
    memory.save()
    missing = [s for s in layout if s not in found]
    if missing:
        print(f"\nNot found (skipped): {', '.join(f'{s} {layout[s]}' for s in missing)}")
    if kept:
        print(f"Kept from the previous calibration: {', '.join(kept)} (place those blocks and re-run to refresh them)")

    out = snaps / f"{stamp}_color_calibration.jpg"
    for d, (spot, _) in zip(dets, where):
        d.name = taught.get(spot, "?").replace(" ", "_") + "_cube"
    img = annotate(frame, scene, pose, dets)
    for s, xy in spots.items():  # mark the spots so a misplaced block is easy to see
        if s in layout:
            u, v = pose.to_image([xy], 0.0)[0].astype(int)
            cv2.circle(img, (u, v), 8, (0, 255, 0), 2)
            cv2.putText(img, s, (u + 10, v + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 0), 2)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nSaved {len(taught)} colour(s) to {color.MEMORY_FILE.relative_to(ROOT.parent)}; photo: {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
