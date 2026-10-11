"""Robot HTTP API on 127.0.0.1:8770 (README section 5.3), on top of robot/executor.py.

    python -m robot.api                      # in the lerobot venv, after perception.publish is running
    python -m robot.api --lesson lessons/sample-flat-v1.json

The Jetson has one overhead webcam and perception.publish owns it, so this process reads frames
from publish's local frame server (FrameClientCamera, http://127.0.0.1:8771/frame) instead of
opening the camera. Frames older than 0.5 s are refused.

One command at a time; each call blocks until the motion has finished and returns {"ok": true}
or {"ok": false, "error": ...}. A second command while one runs gets {"ok": false, "error":
"busy ..."}. SkillError, TaskError, camera trouble, anything: reported, never a crash.
  POST /place {color, cell, level}  level 0: an executor "cell" step; level >= 1: "on_top" of the
                                    block now on top of that cell. The block comes from the supply:
                                    blocks of that color off the grid first, then blocks in a cell
                                    where the lesson does not want that color (never a block that
                                    already sits right in the lesson, --lesson).
  POST /pick {cell, level}          the top block of that cell goes to a free supply spot below the
                                    grid (on the sheet, so the camera can still see it as supply).
  POST /hover {cell}                jaws above the cell centre at travel height, then hold.
  POST /home                        park beside the sheet, out of the camera's view of the grid.
  GET  /status                      {busy, at_home, last_error}; at_home only when parked and no
                                    part of the arm hides the grid from the camera (skills.hides_area).
Cells: letter = row, number = column, as printed on the sheet (executor.cell_center).
Colors: the shared names; "skyblue" is the executor's "light blue".

Block naming: the executor addresses blocks as "<color>_cube", which only reaches the first block
of a color, and the vision-model namer gives each color to one block only. The kit has two of
each, so ApiExecutor names blocks itself, from perception's calibrated colors (no network): for
each command, the block it moves (and the block it stacks on) gets the plain name, the other
block of that color "<color>_cube_2". No change to executor.py or skills.py.
Every command and result is printed and appended to robot/outputs/api_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from perception.cells import mm_to_cell
from perception.colors import CALIB_PATH, active_refs, classify, load_calibration, sample_stats, to_lab
from perception.robot_bridge import default_active_colors
from robot.arm import Arm
from robot.correction import Correction
from robot.executor import Executor, TaskError, cell_center, grid_box
from robot.perception.config import CUBE_SIZE_MM, ROOT
from robot.perception.naming import object_name
from robot.perception.scene import build_scene
from robot.perception.sheet import SheetLocalizer
from robot.session import ArmSession
from robot.skills import SkillError, Skills, hides_area
from robot.workspace import Workspace

API_HOST, API_PORT = "127.0.0.1", 8770
FRAME_URL = "http://127.0.0.1:8771/frame"
MAX_FRAME_AGE_S = 0.5
SUPPLY_SPOTS = [(80.0, 250.0), (136.0, 250.0)]  # sheet mm: below the grid, between tags 3 and 2
MIN_GAP_MM = 8.0
LOG_PATH = ROOT / "outputs" / "api_log.jsonl"
DEFAULT_LESSON = ROOT.parent / "lessons" / "sample-flat-v1.json"
TO_EXECUTOR = {"skyblue": "light blue"}           # shared color -> executor color
FROM_EXECUTOR = {v: k for k, v in TO_EXECUTOR.items()}


# ---------------------------------------------------------------- camera


class FrameClientCamera:
    """robot/perception/camera.Camera's interface (read, latest, close, with), with frames from
    perception.publish's frame server instead of the webcam."""

    def __init__(self, url: str = FRAME_URL, request_timeout_s: float = 2.0):
        self.url, self.request_timeout_s = url, request_timeout_s

    def _fetch(self) -> tuple[np.ndarray, float]:
        with urllib.request.urlopen(self.url, timeout=self.request_timeout_s) as r:
            shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
            t = float(r.headers["X-Timestamp"])
            data = r.read()
        return np.frombuffer(data, np.uint8).reshape(shape).copy(), t

    def read(self, max_age_s: float = MAX_FRAME_AGE_S, timeout_s: float = 3.0) -> np.ndarray:
        """Newest frame captured after this call started and at most max_age_s old (as Camera.read)."""
        asked = time.time()
        deadline, why = asked + timeout_s, "frames too old"
        while time.time() < deadline:
            try:
                frame, t = self._fetch()
            except (OSError, ValueError, KeyError) as e:
                why = f"{type(e).__name__}: {e}"
                time.sleep(0.05)
                continue
            if t > asked and time.time() - t <= max_age_s:
                return frame
            time.sleep(0.03)
        raise RuntimeError(f"no fresh camera frame from {self.url} ({why}); is perception.publish running?")

    def latest(self) -> Optional[np.ndarray]:
        try:
            frame, t = self._fetch()
        except (OSError, ValueError, KeyError):
            return None
        return frame if time.time() - t <= MAX_FRAME_AGE_S else None

    def close(self) -> None:
        pass  # nothing to release: publish owns the camera

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- executor naming


class ApiExecutor(Executor):
    """Executor whose blocks are named by perception's calibrated colors (never the vision model),
    with `focus` deciding which block of a color gets the plain "<color>_cube" name."""

    def __init__(self, *args, refs: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.refs = refs
        self.focus: dict[str, list] = {}  # executor color -> sheet points; the nearest block is "<color>_cube"

    def observe(self):
        frame = self.cam.read()
        scene, _, detections = build_scene(frame, self.loc, naming="off")
        if scene is None:
            raise TaskError("the camera can't see the sheet (need 2+ tags)")
        self.camera_xyz = scene["sheet"]["camera_xyz_mm"]
        blocks = []
        for d, o in zip(detections, scene["objects"]):
            if o["score"] < 0:
                continue  # merged blobs can't be grasped (as Executor.observe)
            blocks.append({**o, "colour": self._colour(frame, d.contour)})
        return self._name(blocks)

    def _colour(self, frame: np.ndarray, contour: np.ndarray) -> Optional[str]:
        """Executor color name of a blob, from the central part of it, or None."""
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.asarray(contour, np.int32).reshape(-1, 2)], 255)
        k = max(3, int(np.sqrt(cv2.countNonZero(mask)) * 0.25) | 1)
        core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        x, y, w, h = cv2.boundingRect(core if cv2.countNonZero(core) else mask)
        sel = (core if cv2.countNonZero(core) else mask)[y:y + h, x:x + w] > 0
        try:
            median, _, _ = sample_stats(to_lab(frame[y:y + h, x:x + w])[sel])
        except ValueError:
            return None
        colour, _, _, _ = classify(median, self.refs)
        return TO_EXECUTOR.get(colour, colour) if colour in self.refs else None

    def _name(self, blocks: list[dict]) -> dict:
        objs, by_colour = {}, {}
        for b in blocks:
            by_colour.setdefault(b["colour"], []).append(b)
        for colour, group in by_colour.items():
            if colour is None:
                for i, b in enumerate(group, 1):
                    objs[f"unknown_cube_{i}"] = {**b, "name": f"unknown_cube_{i}"}
                continue
            base = object_name(colour)
            points = self.focus.get(colour) or ([self.last_xy[base]] if base in self.last_xy else [])

            def near(b):
                xy = np.asarray(b["center_mm"], float)
                return min((float(np.hypot(*(xy - np.asarray(p, float)))) for p in points), default=0.0), tuple(xy)

            for i, b in enumerate(sorted(group, key=near)):
                name = base if i == 0 else f"{base}_{i + 1}"
                objs[name] = {**b, "name": name}
        return objs


# ---------------------------------------------------------------- commands


class RobotAPI:
    def __init__(self, ex: ApiExecutor, sk: Skills, arm: Arm, ws: Workspace, lesson_final: dict, log=print):
        self.ex, self.sk, self.arm, self.ws, self.lesson_final, self.log = ex, sk, arm, ws, lesson_final, log
        self.lock = threading.Lock()
        self.busy, self.at_home, self.last_error = False, False, None

    # -- bookkeeping
    def _record(self, entry: dict) -> None:
        entry = {"t": time.time(), **entry}
        self.log(f"[robot.api] {json.dumps(entry)}")
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def run(self, cmd: str, body: dict, fn) -> dict:
        if not self.lock.acquire(blocking=False):
            return {"ok": False, "error": "busy with another command"}
        self.busy, self.at_home = True, False
        started = time.time()
        self._record({"cmd": cmd, **body})
        try:
            fn(**body)
            result = {"ok": True}
        except (TaskError, SkillError, RuntimeError, ValueError) as e:
            result = {"ok": False, "error": str(e)}
        except Exception as e:  # never crash the server
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.at_home = self._clear_of_grid()
            self.busy = False
            self.lock.release()
        self.last_error = result.get("error")
        self._record({"cmd": cmd, "result": result, "seconds": round(time.time() - started, 1),
                      "at_home": self.at_home})
        return result

    def status(self) -> dict:
        return {"busy": self.busy, "at_home": self.at_home and not self.busy, "last_error": self.last_error}

    def _clear_of_grid(self) -> bool:
        """Parked: the jaws at rest beside the sheet and no part of the arm over the grid as the
        camera sees it."""
        if self.ex.camera_xyz is None or self.sk.holding is not None:
            return False
        try:
            return not hides_area(self.ws, self.arm.read(), self.ex.camera_xyz, grid_box())
        except Exception:
            return False

    # -- helpers
    @staticmethod
    def _cell(cell: str) -> str:
        cell = str(cell).strip().upper()
        cell_center(cell)  # raises TaskError for anything but A1-C3
        return cell

    @staticmethod
    def _colour(color: str) -> str:
        c = str(color).strip().lower()
        if c in FROM_EXECUTOR:
            c = FROM_EXECUTOR[c]
        return TO_EXECUTOR.get(c, c)

    def _cell_of(self, name: str, objs: dict) -> Optional[str]:
        return mm_to_cell(*self.ex.pos(name, objs))

    def _top_of_cell(self, cell: str, objs: dict) -> Optional[str]:
        here = [n for n in objs if self._cell_of(n, objs) == cell and self.ex.on_top_of(n) is None]
        return max(here, key=self.ex.base) if here else None

    def _level(self, name: str) -> int:
        return int(round(self.ex.base(name) / CUBE_SIZE_MM))

    def _in_lesson_place(self, name: str, objs: dict, colour: str) -> bool:
        cell = self._cell_of(name, objs)
        want = self.lesson_final.get(cell) or []
        level = self._level(name)
        return cell is not None and level < len(want) and TO_EXECUTOR.get(want[level], want[level]) == colour

    def _supply(self, colour: str, objs: dict, target) -> str:
        cands = [n for n, o in objs.items() if o.get("colour") == colour and self.ex.on_top_of(n) is None
                 and not self._in_lesson_place(n, objs, colour)]
        if not cands:
            raise TaskError(f"no {FROM_EXECUTOR.get(colour, colour)} block in the supply "
                            "(off the grid, or in a cell the lesson doesn't want it)")
        return min(cands, key=lambda n: (self._cell_of(n, objs) is not None,
                                         float(np.hypot(*(self.ex.pos(n, objs) - np.asarray(target, float))))))

    def _free_supply_spot(self, mover: str, objs: dict):
        others = [self.ex.pos(n, objs) for n in objs if n != mover]
        half = CUBE_SIZE_MM
        for spot in SUPPLY_SPOTS:
            if all(max(abs(spot[0] - o[0]), abs(spot[1] - o[1])) - half >= MIN_GAP_MM for o in others):
                return list(spot)
        return None

    # -- the four motions
    def place(self, color: str, cell: str, level: int = 0) -> None:
        colour, cell = self._colour(color), self._cell(cell)
        self.ex.focus = {}
        objs = self.ex.observe()
        if level == 0:
            target = cell_center(cell)
            goal = {"type": "cell", "cell": cell}
        else:
            top = self._top_of_cell(cell, objs)
            if top is None:
                raise TaskError(f"nothing on {cell} to stack on")
            if self._level(top) != level - 1:
                raise TaskError(f"the top of {cell} is at level {self._level(top)}, so a block goes on level "
                                f"{self._level(top) + 1}, not {level}")
            ref_colour = objs[top]["colour"]
            if ref_colour is None:
                raise TaskError(f"can't tell the color of the block on top of {cell}")
            if ref_colour == colour:
                raise TaskError("stacking a block on one of the same color isn't supported (the executor "
                                "names both the same)")
            target = self.ex.pos(top, objs)
            self.ex.focus[ref_colour] = [target]
            goal = {"type": "on_top", "reference": ref_colour}
        source = self._supply(colour, objs, target)
        self.ex.focus[colour] = [self.ex.pos(source, objs), target]
        self.ex.run([{"object": colour, "goal": goal}])

    def pick(self, cell: str, level: int = 0) -> None:
        cell = self._cell(cell)
        self.ex.focus = {}
        objs = self.ex.observe()
        top = self._top_of_cell(cell, objs)
        if top is None:
            raise TaskError(f"no block on {cell}")
        if self._level(top) != level:
            raise TaskError(f"the top block of {cell} is at level {self._level(top)}, not {level}")
        colour = objs[top]["colour"]
        if colour is None:
            raise TaskError(f"can't tell the color of the block on {cell}, so it can't be addressed")
        spot = self._free_supply_spot(top, objs)
        if spot is None:
            raise TaskError("no free supply spot below the grid to put it")
        self.ex.focus = {colour: [self.ex.pos(top, objs), spot]}
        self.ex.run([{"object": colour, "goal": {"type": "spot", "xy": spot}}])

    def hover(self, cell: str) -> None:
        xy = cell_center(self._cell(cell))
        self.sk.go(xy, self.sk.travel_h(), roll=0.0)

    def home(self) -> None:
        self.sk.park(camera_xyz=self.ex.camera_xyz, keep_clear=grid_box())


# ---------------------------------------------------------------- HTTP


class PlaceBody(BaseModel):
    color: str
    cell: str
    level: int = 0


class PickBody(BaseModel):
    cell: str
    level: int = 0


class HoverBody(BaseModel):
    cell: str


def _fields(body: BaseModel) -> dict:
    return body.model_dump() if hasattr(body, "model_dump") else body.dict()  # pydantic 2 or 1


def create_app(api: RobotAPI) -> FastAPI:
    app = FastAPI(title="Apprentice robot API")

    @app.post("/place")
    def place(body: PlaceBody):
        return api.run("place", _fields(body), api.place)

    @app.post("/pick")
    def pick(body: PickBody):
        return api.run("pick", _fields(body), api.pick)

    @app.post("/hover")
    def hover(body: HoverBody):
        return api.run("hover", _fields(body), api.hover)

    @app.post("/home")
    def home():
        return api.run("home", {}, api.home)

    @app.get("/status")
    def status():
        return api.status()

    return app


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=API_PORT)
    ap.add_argument("--frame-url", default=FRAME_URL)
    ap.add_argument("--lesson", type=Path, default=DEFAULT_LESSON,
                    help="lesson JSON: blocks already in their lesson cell are never taken as supply")
    ap.add_argument("--calib", type=Path, default=CALIB_PATH, help="perception's block colors (colors.json)")
    args = ap.parse_args(argv)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    cam = FrameClientCamera(args.frame_url)
    cam.read(timeout_s=10.0)  # fail now, before the arm moves, if perception.publish isn't serving frames
    lesson_final = json.loads(args.lesson.read_text(encoding="utf-8")).get("final_board", {}) if args.lesson.exists() else {}
    refs = active_refs(load_calibration(args.calib), default_active_colors())
    ws, loc, arm = Workspace(), SheetLocalizer(), Arm()
    sk = Skills(arm, ws, log=lambda m: log(f"  {m}"), correct=Correction())
    with ArmSession(arm, sk, log=log):
        ex = ApiExecutor(sk, cam, loc, log=lambda m: log(f"  {m}"), refs=refs)
        api = RobotAPI(ex, sk, arm, ws, lesson_final, log=log)
        try:  # learn where the camera is, then park beside the sheet out of its view
            ex.observe()
            sk.park(camera_xyz=ex.camera_xyz, keep_clear=grid_box())
        except (TaskError, SkillError, RuntimeError) as e:
            log(f"[robot.api] start-up look-and-park failed ({e}); /home will retry")
        api.at_home = api._clear_of_grid()
        log(f"[robot.api] serving on http://{API_HOST}:{args.port} (at_home={api.at_home}, "
            f"colors {', '.join(refs)}, lesson {args.lesson.name if args.lesson.exists() else 'none'})")
        uvicorn.run(create_app(api), host=API_HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
