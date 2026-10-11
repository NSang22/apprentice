"""Robot vision for Buddy: what the arm's own perception sees on the grid.

Run by buddy/table.py (RobotEyes) as a long-lived helper with cwd and PYTHONPATH set to the
repo's main checkout, so every import below is the code robot.api runs, with its calibration,
its colour memory (robot/outputs/color_memory.json) and its OpenAI key:

  segmentation      robot.perception.segment (via scene.build_scene)
  colour names      robot.perception.naming: colour memory first, the vision model (OpenAI) when unsure, hue guess last
  sheet mm -> cell  perception.cells.mm_to_cell (the team's one cell-naming source)

Protocol: one command line on stdin -> one line "@@{json}" on stdout. Lines without "@@" are the
robot code's own log output and are only logged. At start it prints one "@@{json}" with the
cell map: for every cell name, the robot cell whose centre (robot.executor.cell_center, what
robot.api moves to) is the same point on the sheet as perception's centre for that name.

  scan               the grid, as above
  bank               the vision model looks at the robot's cube bank (the green box, robot config
                     BANK_ZONE_MM) and lists the block colors in it. robot.api names bank blocks
                     with the camera's sheet calibration, which can misread them on the green box;
                     this is the second opinion Buddy asks before telling a child a color is missing.
  narration <path>   speech to text (OPENAI_TRANSCRIBE_MODEL, gpt-4o-transcribe), then the vision
                     model lists the placements spoken in it ("red on B2"). Advice only: the grid
                     decides (AGENTS.md rule 2).
All model calls go through robot/perception/openai_vision.py (OPENAI_API_KEY, OPENAI_MODEL, default
gpt-5.6-sol), the same as the extractor's color fallback, and never print the key.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

from perception.cells import cell_center_mm, edge_margin_mm, mm_to_cell
from robot.executor import cell_center as robot_cell_center
from robot.perception import naming, scene as scene_mod

try:
    from robot.perception import openai_vision, vision_namer
except ImportError:
    # The main checkout predates the OpenAI switch: load this branch's client from next to this file,
    # and point the robot's colour namer at it, in this helper process only (no Gemini call is made).
    import importlib.util

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    _here = Path(__file__).resolve().parent.parent / "robot" / "perception"
    openai_vision = _load("robot.perception.openai_vision", _here / "openai_vision.py")
    vision_namer = _load("robot.perception.vision_namer", _here / "vision_namer.py")
    if hasattr(naming, "gemini_namer"):
        naming.gemini_namer = vision_namer
from robot.perception.sheet import SheetLocalizer
from shared.config import CELLS, COLORS

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
    notes: list[str] = []   # what the colour namer says this scan (model asked, model failed, guesses)
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
        notes.insert(0, "traditional vision was unsure of a colour, so the vision model was asked")
    elif source == "prior":
        notes.insert(0, "traditional vision was unsure of a colour and the vision model did not answer: hue guess used")
    return {"ok": True, "blocks": blocks, "naming": source, "notes": notes,
            "ms": {"frame": round((t1 - t0) * 1000), "vision": round((t2 - t1) * 1000)}}


BANK_PAD_MM = 25.0
BANK_PROMPT = (
    "This photo, taken from above, shows a green box that holds toy blocks for a robot arm. List the "
    f"color of every block inside the green box, one entry per block, using only these words: {', '.join(COLORS)}. "
    "Ignore blocks outside the box, the box itself, shadows and glare. If the box is empty, return an empty list. "
    'Reply with JSON only, like {"blocks": ["red", "teal"]}.'
)
NARRATION_PROMPT = (
    "This is the transcript of someone (often a child, maybe with an adult) building with toy blocks "
    "on a 3 by 3 grid while talking. Grid squares are named A1 to C3: the letter is the row (A on the "
    "robot's side, C nearest the builder), the number is the column (1 to 3, left to right). Transcribe "
    "what is said, then list every block placement or removal the speaker describes, with the time in "
    f"seconds when it is said. Colors must be one of: {', '.join(COLORS)}, or \"unknown\". Use a cell only "
    "if it is said or clearly described, else null. Reply with JSON only: "
    '{"placements": [{"action": "place", "color": "red", "cell": "B2", "text": "red one in the middle"}]}'
)


def bank(localizer: SheetLocalizer) -> dict:
    import cv2
    from robot.perception.config import BANK_ZONE_MM
    if BANK_ZONE_MM is None:
        return {"ok": False, "error": "no cube bank in robot/perception/config.py"}
    img = frame()
    pose = localizer.locate(img)
    if pose is None:
        return {"ok": False, "error": "I can't see the sheet's tags"}
    x0, y0, x1, y1 = BANK_ZONE_MM
    p = BANK_PAD_MM
    pts = pose.to_image([(x0 - p, y0 - p), (x1 + p, y0 - p), (x1 + p, y1 + p), (x0 - p, y1 + p)])
    h, w = img.shape[:2]
    bx0, by0 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
    bx1, by1 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
    if bx1 - bx0 < 20 or by1 - by0 < 20:
        return {"ok": False, "error": "the bank is out of the camera's view"}
    ok, jpg = cv2.imencode(".jpg", img[by0:by1, bx0:bx1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    r = openai_vision.ask_json(BANK_PROMPT, [jpg.tobytes()])
    colors = [TO_SHARED.get(c, c) for c in (r["answer"].get("blocks") or []) if isinstance(c, str)]
    return {"ok": True, "colors": [c for c in colors if c in COLORS], "seconds": r["seconds"], "model": r["model"]}


def narration(path: str) -> dict:
    audio = Path(path)
    if not audio.is_file() or audio.stat().st_size < 2000:
        return {"ok": False, "error": "no audio"}
    heard = openai_vision.transcribe(audio, prompt="Toy blocks on a 3 by 3 grid, squares A1 to C3. "
                                                   "Colors: red, orange, yellow, teal, sky blue, purple.")
    if not heard["text"]:
        return {"ok": True, "transcript": "", "placements": [], "seconds": heard["seconds"], "model": heard["model"]}
    r = openai_vision.ask_json(NARRATION_PROMPT + "\n\nThe transcript:\n" + heard["text"])
    a = r["answer"] if isinstance(r["answer"], dict) else {}
    return {"ok": True, "transcript": heard["text"], "placements": a.get("placements") or [],
            "seconds": round(heard["seconds"] + r["seconds"], 2), "model": f"{heard['model']} + {r['model']}"}


def main() -> None:
    # The games use only the colors in play (robot ACTIVE_COLORS). The robot's namer also knows its
    # extra colors (sky blue, orange) and once named a teal block "light blue", which then disagreed
    # with the camera; in this helper process only, it chooses among the colors in play.
    from robot.perception.config import ACTIVE_COLORS
    naming.NAMEABLE = list(ACTIVE_COLORS)
    localizer = SheetLocalizer()
    say({"ready": True, "cell_map": cell_map(), "model": openai_vision.model() if openai_vision.available() else None})
    for line in sys.stdin:
        cmd, _, arg = line.strip().partition(" ")
        if cmd not in ("scan", "bank", "narration"):
            continue
        try:
            say(scan(localizer) if cmd == "scan" else bank(localizer) if cmd == "bank" else narration(arg))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            say({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
