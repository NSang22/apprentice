"""Grab one overhead frame (or load an image), detect cubes, print the scene in sheet mm.

    python -m robot.scripts.tests.scene_snapshot                     # live camera
    python -m robot.scripts.tests.scene_snapshot robot/samples/overhead_02.png
Saves an annotated image to robot/outputs/scene_latest.png.
"""

import json
import sys

import cv2

from robot.perception.camera import Camera
from robot.perception.config import ROOT
from robot.perception.scene import annotate, build_scene
from robot.perception.sheet import SheetLocalizer


def main():
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            raise SystemExit(f"Can't read {sys.argv[1]}")
    else:
        with Camera() as cam:
            frame = cam.read()

    scene, pose, detections = build_scene(frame, SheetLocalizer())
    out_path = ROOT / "outputs" / "scene_latest.png"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), annotate(frame, scene, pose, detections))
    print(json.dumps(scene, indent=2) if scene else "Sheet not found: need at least 2 tags visible.")
    print("Annotated image:", out_path)


if __name__ == "__main__":
    main()
