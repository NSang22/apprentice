"""Measure where the overhead camera sits and pin it in perception/config.py (CAMERA_XYZ_MM).

    python -m robot.scripts.calibration.pin_camera

Clear the sheet first: no blocks, arm parked off to the side, all 4 corner tags in view.
It takes the median camera position over many 4-tag frames, because any single frame's
estimate can be far off for a tilted camera. Run auto_calibrate afterwards.
"""

import re
import time

import cv2
import numpy as np

from robot.perception.camera import Camera
from robot.perception.config import ROOT
from robot.perception.sheet import SheetLocalizer

FRAMES = 30
MAX_SPREAD_MM = 15.0  # wider than this between frames: something moved or tags are misread
CONFIG = ROOT / "perception" / "config.py"


def main():
    loc, cam = SheetLocalizer(), Camera()
    estimates = []
    try:
        for _ in range(FRAMES * 3):
            if len(estimates) == FRAMES:
                break
            frame = cam.read()
            found = loc.detect_tags(frame)
            if len(found) < len(loc.tags):
                time.sleep(0.1)
                continue
            ids = sorted(found)
            img = np.concatenate([found[i] for i in ids])
            obj = np.concatenate([loc.tags[i] for i in ids])
            H, _ = cv2.findHomography(img, obj)
            c = loc._camera_centre(H, img, obj, frame.shape)
            if c is not None:
                estimates.append(c)
    finally:
        cam.close()
    if len(estimates) < FRAMES // 2:
        raise SystemExit(f"Only {len(estimates)} frames showed all {len(loc.tags)} tags. Clear the sheet "
                         "(blocks, arm, hands) and try again. Config NOT changed.")
    e = np.array(estimates)
    med = np.median(e, axis=0)
    spread = np.percentile(e, 90, axis=0) - np.percentile(e, 10, axis=0)
    print(f"{len(e)} frames. Camera at x {med[0]:.1f}, y {med[1]:.1f}, height {med[2]:.1f} mm "
          f"(10-90% spread x {spread[0]:.1f}, y {spread[1]:.1f}, z {spread[2]:.1f} mm)")
    if spread.max() > MAX_SPREAD_MM:
        raise SystemExit(f"Spread over {MAX_SPREAD_MM:.0f} mm: the estimate is not steady. Config NOT changed.")
    value = f"({med[0]:.1f}, {med[1]:.1f}, {med[2]:.1f})"
    text = CONFIG.read_text()
    new, n = re.subn(r"^(CAMERA_XYZ_MM: [^=]+= ).*$", rf"\g<1>{value}  # pin_camera, {time.strftime('%Y-%m-%d %H:%M')}",
                     text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"Couldn't find CAMERA_XYZ_MM in {CONFIG}; set it by hand to {value}.")
    CONFIG.write_text(new)
    print(f"Pinned CAMERA_XYZ_MM = {value} in robot/perception/config.py. Now run auto_calibrate.")


if __name__ == "__main__":
    main()
