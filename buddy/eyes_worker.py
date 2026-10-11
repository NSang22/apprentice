"""Robot vision for Buddy: what the arm's own perception sees on the grid.

Run by buddy/table.py (RobotEyes) as a long-lived helper with cwd and PYTHONPATH set to the
repo's main checkout, so every import below is the code robot.api runs, with its calibration,
its colour memory (robot/outputs/color_memory.json) and its Gemini key:

  segmentation      robot.perception.segment (via scene.build_scene)
  colour names      robot.perception.naming: colour memory first, Gemini when unsure, hue guess last
  sheet mm -> cell  perception.cells.mm_to_cell (the team's one cell-naming source)

Protocol: one line "scan" on stdin -> one line "@@{json}" on stdout. Lines without "@@" are the
robot code's own log output and are only logged. At start it prints one "@@{json}" with the
cell map: for every cell name, the robot cell whose centre (robot.executor.cell_center, what
robot.api moves to) is the same point on the sheet as perception's centre for that name.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
import urllib.request

import numpy as np

from perception.cells import cell_center_mm, edge_margin_mm, mm_to_cell
from robot.executor import cell_center as robot_cell_center
from robot.perception import gemini_namer, naming, scene as scene_mod
from robot.perception.sheet import SheetLocalizer
from shared.config import CELLS

FRAME_URL = "http://127.0.0.1:8771/frame"   # perception.publish's frame server (it owns the camera)
TO_SHARED = {"light blue": "skyblue"}       # robot colour words -> shared/config.py COLORS


def say(obj: dict) -> None:
    print("@@" + json.dumps(obj), flush=True)


def frame() -> np.ndarray:
    with urllib.request.urlopen(FRAME_URL, timeout=3) as r:
        shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
        return np.frombuffer(r.read(), np.uint8).reshape(shape).copy()


def cell_map() -> dict:
    out = {}
    for cell in CELLS:
        target = np.array(cell_center_mm(cell))
        best = min(CELLS, key=lambda r: float(np.linalg.norm(robot_cell_center(r) - target)))
        out[cell] = {"robot": best, "error_mm": round(float(np.linalg.norm(robot_cell_center(best) - target)), 2)}
    return out


def colour_of(object_name: str) -> str:
    """'light_blue_cube_2' -> 'skyblue'."""
    words = naming.normalize(re.sub(r"_cube(_\d+)?$", "", object_name))
    return TO_SHARED.get(words, words)


def scan(localizer: SheetLocalizer) -> dict:
    t0 = time.perf_counter()
    img = frame()
    t1 = time.perf_counter()
    notes: list[str] = []   # what the colour namer says this scan (Gemini asked, Gemini failed, guesses)
    namer = scene_mod._namer("auto")
    namer.log = lambda m: (notes.append(m.strip()), print(m, flush=True))
    scene, _, _ = scene_mod.build_scene(img, localizer, naming="auto")
    if scene is None:
        return {"ok": False, "error": "I can't see the sheet's tags"}
    blocks = []
    for o in scene["objects"]:
        x, y = o["center_mm"]
        cell = mm_to_cell(x, y)
        blocks.append({"cell": cell, "color": colour_of(o["name"]), "center_mm": [x, y],
                       "edge_mm": round(edge_margin_mm(x, y), 1) if cell else None, "score": o["score"]})
    t2 = time.perf_counter()
    source = getattr(namer, "last_source", None)
    if source == "model":
        notes.insert(0, "traditional vision was unsure of a colour, so Gemini was asked")
    elif source == "prior":
        notes.insert(0, "traditional vision was unsure of a colour and Gemini did not answer: hue guess used")
    return {"ok": True, "blocks": blocks, "naming": source, "notes": notes,
            "ms": {"frame": round((t1 - t0) * 1000), "vision": round((t2 - t1) * 1000)}}


def main() -> None:
    localizer = SheetLocalizer()
    say({"ready": True, "cell_map": cell_map(), "gemini": gemini_namer.available()})
    for line in sys.stdin:
        if line.strip() != "scan":
            continue
        try:
            say(scan(localizer))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            say({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
