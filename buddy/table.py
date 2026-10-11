"""The real table for Buddy: the camera's board, the robot's own eyes, and Pip's arm.

Nothing here opens the camera or the servo bus. perception.publish owns the webcam (and shares
frames on :8771), robot.api owns the arm (:8770); Buddy is one more client of each.

Two readings of the board, kept in sync:
  TableBoard  perception's GridState stream (ws :8765) through brain.stabilizer, unchanged. It
              answers "are hands off?" and "has the board settled?" ten times a second.
  RobotEyes   the arm's own vision (buddy/eyes_worker.py, run in the main checkout): robot
              segmentation, colour memory, the vision model (OpenAI) when unsure. It decides what the child built,
              so the board Buddy grades is the board the arm acts on.
Each scan records both, and where they disagree (buddy/session.py logs it, the phone shows it).

Cell names: perception/cells.py and robot/api.py both name cells as printed on the sheet (letter
= row from the arm's edge, number = column). Pip never assumes that: the cell map pairs each
name with the robot cell whose centre is the same point on the sheet (eyes_worker.cell_map, or
local_cell_map here), so a future convention change cannot send the arm to the wrong square.

Safety: every arm move waits until the camera saw hands clear in a fresh frame, runs one at a
time, and only points (hover) or takes a block off (pick). Pip announces each move on the phone.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from websockets.asyncio.client import connect

from brain.robot_client import HttpRobotClient
from brain.stabilizer import Stabilizer
from shared.config import CELLS, GRID_WS_PORT, REPO_ROOT, ROBOT_HTTP_PORT, UNKNOWN
from shared.schemas import GridState

FRESH_S = 1.0             # a frame older than this does not count as "the camera sees it"
HANDS_WAIT_S = 20.0       # how long an arm move waits for hands to leave the board
HOVER_HOLD_S = 3.0        # Pip points this long, then goes home
EDGE_MM = 8.0             # a block centre closer than this to a cell line reads "unknown" (nudge it)
CELL_MATCH_MM = 1.0       # a cell-map pair must name the same sheet point to within this
EYES_START_S = 90.0       # the robot-vision helper's first import of torch/lerobot is slow
EYES_SCAN_S = 45.0        # one scan, including a vision-model call when the colour memory is unsure
WORKER = Path(__file__).resolve().parent / "eyes_worker.py"


def main_checkout() -> Path:
    """The repo's main checkout (first entry of git worktree list). The camera, the arm and the
    robot's vision run from there: the gitignored calibration, colour memory and API keys live in
    that folder only, not in other worktrees such as apprentice-buddy."""
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True,
                             timeout=5, cwd=str(REPO_ROOT)).stdout
        return Path(next(l[len("worktree "):] for l in out.splitlines() if l.startswith("worktree ")))
    except Exception:
        return REPO_ROOT


def local_cell_map() -> dict[str, str]:
    """name -> robot cell name, by matching centres on the sheet (this checkout's code)."""
    import numpy as np
    from perception.cells import cell_center_mm
    from robot.perception.sheet import cell_center as robot_cell_center

    out = {}
    for cell in CELLS:
        target = np.array(cell_center_mm(cell))
        best = min(CELLS, key=lambda r: float(np.linalg.norm(np.array(robot_cell_center(r)) - target)))
        if float(np.linalg.norm(np.array(robot_cell_center(best)) - target)) > CELL_MATCH_MM:
            raise RuntimeError(f"no robot cell at perception's {cell}: the two grids disagree")
        out[cell] = best
    return out


FRAME_URL = "http://127.0.0.1:8771/frame"   # perception.publish's frame server (it owns the camera)


def camera_jpeg(width: int = 640, quality: int = 70) -> Optional[bytes]:
    """The camera's newest frame as a small JPEG (the app's "Pip's eyes" view), or None when
    perception is not sharing frames."""
    import urllib.request
    import cv2
    import numpy as np
    try:
        with urllib.request.urlopen(FRAME_URL, timeout=2) as r:
            shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
            img = np.frombuffer(r.read(), np.uint8).reshape(shape)
    except Exception:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def missing_from_bank(need: list[str], seen: list[str]) -> list[str]:
    """Colors still needed after matching what the vision model saw in the bank, counting repeats."""
    left = list(seen)
    out = []
    for c in need:
        if c in left:
            left.remove(c)
        else:
            out.append(c)
    return out


def blocks_to_board(blocks: list[dict], edge_mm: float = EDGE_MM) -> dict[str, list[str]]:
    """The robot's blocks -> a board. Off the grid (the supply spots) is ignored; a block near a
    cell line reads "unknown" so the game asks for a nudge instead of guessing its square."""
    board: dict[str, list[str]] = {c: [] for c in CELLS}
    for b in blocks:
        cell = b.get("cell")
        if cell not in board:
            continue
        near_line = b.get("edge_mm") is not None and b["edge_mm"] < edge_mm
        board[cell].append(UNKNOWN if near_line else b["color"])
    return board


def board_diff(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list]:
    """Cells where two boards disagree: {cell: [what a has, what b has]} (bottom block, or None)."""
    out = {}
    for cell in CELLS:
        x = a.get(cell)[0] if a.get(cell) else None
        y = b.get(cell)[0] if b.get(cell) else None
        if x != y:
            out[cell] = [x, y]
    return out


class TableBoard:
    """Latest camera frame plus a stabilized board, fed by perception's GridState stream."""

    def __init__(self, url: str = f"ws://127.0.0.1:{GRID_WS_PORT}"):
        self.url = url
        self.lock = threading.Lock()
        self.stabilizer = Stabilizer()
        self.connected = False
        self.last_t = 0.0                    # when the last frame arrived (time.time)
        self.hands_clear = False
        self.latest: Optional[dict] = None   # raw board of the last frame
        self.stable: Optional[dict] = None   # last board the stabilizer emitted
        self.stable_t = 0.0
        self.recording: Optional[list[dict]] = None   # Teach the Robot: every GridState while recording

    def record_start(self) -> None:
        with self.lock:
            self.recording = []

    def record_stop(self) -> list[dict]:
        with self.lock:
            frames, self.recording = self.recording or [], None
            return frames

    async def run(self) -> None:
        while True:
            try:
                async with connect(self.url, open_timeout=2) as ws:
                    self.connected = True
                    async for msg in ws:
                        self._feed(msg)
            except Exception:
                pass
            self.connected = False
            await asyncio.sleep(1.0)

    def _feed(self, msg) -> None:
        try:
            raw = json.loads(msg)
            gs = GridState.from_dict(raw)
        except Exception:
            return
        with self.lock:
            if self.recording is not None:
                self.recording.append(raw)
            self.last_t = time.time()
            self.hands_clear, self.latest = gs.hands_clear, gs.board
            board = self.stabilizer.feed(gs)
            if board is not None:
                self.stable, self.stable_t = board, time.time()

    def fresh(self) -> bool:
        return time.time() - self.last_t < FRESH_S

    def ready(self) -> bool:
        return self.connected and self.fresh()

    def hands_away(self) -> bool:
        return self.fresh() and self.hands_clear

    def begin_scan(self) -> float:
        """Forget the settled board; the next one emitted settled after this moment."""
        with self.lock:
            self.stabilizer.reset()
            self.stable = None
            return time.time()

    def scanned(self, since: float) -> Optional[dict]:
        with self.lock:
            return self.stable if self.stable is not None and self.stable_t >= since else None

    def status(self) -> dict:
        return {"connected": self.connected, "fresh": self.fresh(), "hands_clear": self.hands_away()}


class RobotEyes:
    """Client for buddy/eyes_worker.py, started once in the main checkout and kept warm."""

    def __init__(self, repo: Optional[Path] = None):
        self.repo = repo or main_checkout()
        self.proc: Optional[subprocess.Popen] = None
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self.info: dict = {}                 # the worker's start-up report (cell map, model)
        self.log: list[str] = []             # the robot code's own output, newest last
        self.error: Optional[str] = None

    def _reader(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                self.lines.put(line[2:])
            else:
                self.log = (self.log + [line])[-200:]

    def _reply(self, timeout_s: float) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                return json.loads(self.lines.get(timeout=0.5))
            except queue.Empty:
                if self.proc is None or self.proc.poll() is not None:
                    raise RuntimeError("the robot vision helper stopped (see the robot log lines)")
        raise RuntimeError(f"no answer from the robot vision helper in {timeout_s:.0f} s")

    def _ensure(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        env = {**os.environ, "PYTHONPATH": str(self.repo), "PYTHONUNBUFFERED": "1"}
        self.lines = queue.Queue()
        self.proc = subprocess.Popen([sys.executable, str(WORKER)], cwd=str(self.repo), env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     bufsize=1, start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.info = self._reply(EYES_START_S)

    def warm(self) -> None:
        """Start the helper in the background (its imports take a while) so the first scan is quick."""
        def job():
            try:
                with self.lock:
                    self._ensure()
                self.error = None
            except Exception as e:
                self.error = str(e)
        threading.Thread(target=job, daemon=True).start()

    def _ask(self, command: str, timeout_s: float) -> dict:
        with self.lock:
            try:
                self._ensure()
                self.proc.stdin.write(command + "\n")
                self.proc.stdin.flush()
                return self._reply(timeout_s)
            except Exception as e:
                self.error = str(e)
                return {"ok": False, "error": str(e)}

    def scan(self) -> dict:
        """{ok, blocks: [{cell, color, center_mm, edge_mm, score}], naming} or {ok: False, error}."""
        return self._ask("scan", EYES_SCAN_S)

    def bank(self) -> dict:
        """The vision model's list of the block colors in the robot's cube bank: {ok, colors: [...]}."""
        return self._ask("bank", EYES_SCAN_S)

    def narration(self, audio: Path) -> dict:
        """The recording's narration as text (OpenAI speech to text) and the placements said in it."""
        return self._ask(f"narration {Path(audio).resolve()}", 90.0)

    def cell_map(self) -> dict[str, str]:
        m = self.info.get("cell_map") or {}
        if len(m) == len(CELLS) and all(v.get("error_mm", 99) <= CELL_MATCH_MM for v in m.values()):
            return {c: v["robot"] for c, v in m.items()}
        return local_cell_map()

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


class Pip:
    """Pip's arm moves, one at a time on a worker thread, so the phone never waits on a motion."""

    def __init__(self, board: TableBoard, eyes: Optional[RobotEyes] = None,
                 base_url: str = f"http://127.0.0.1:{ROBOT_HTTP_PORT}"):
        self.board = board
        self.eyes = eyes
        self.client = HttpRobotClient(base_url)
        self.busy = False
        self.step = ""                       # what Pip is doing, for the phone
        self.error: Optional[str] = None
        self.left_over: list[str] = []       # cells Pip could not clear (the child takes those off)

    def to_robot_cell(self, cell: str) -> str:
        """The robot's name for the square the camera calls `cell` (same point on the sheet)."""
        mapping = self.eyes.cell_map() if self.eyes else local_cell_map()
        return mapping[cell]

    def reachable(self) -> bool:
        reply = self.client.status()  # {busy, at_home, last_error} on success, {"ok": False, ...} on failure
        return reply.get("ok") is not False and "busy" in reply

    def _wait_hands(self, should_stop: Callable[[], bool]) -> bool:
        deadline = time.time() + HANDS_WAIT_S
        while time.time() < deadline and not should_stop():
            if self.board.hands_away():
                return True
            self.step = "hands"
            time.sleep(0.1)
        return False

    def _run(self, job: Callable[[], None]) -> None:
        def wrapped():
            try:
                job()
            except Exception as e:  # never let a robot problem crash the game
                self.error = str(e)
            finally:
                try:
                    self.client.home()
                except Exception:
                    pass
                self.busy, self.step = False, "done"

        self.busy, self.error, self.step, self.left_over = True, None, "start", []
        threading.Thread(target=wrapped, daemon=True).start()

    def point(self, cell: str, should_stop: Callable[[], bool] = lambda: False) -> None:
        def job():
            if not self._wait_hands(should_stop):
                self.error = "hands"
                return
            self.step = "moving"                    # hover blocks until the jaws are over the square
            reply = self.client.hover(self.to_robot_cell(cell))
            if not reply.get("ok"):
                self.error = reply.get("error", "hover failed")
                return
            self.step = "pointing"                  # arrived: hold so the child sees it, then home
            time.sleep(HOVER_HOLD_S)
            self.step = "homing"
        self._run(job)

    def place(self, cell: str, color: str, should_stop: Callable[[], bool] = lambda: False, level: int = 0) -> None:
        """Pick a block of `color` from the supply beside the grid and put it on `cell` at `level`
        (Take Turns: always 0; Teach the Robot rebuilds: 1 stacks on the block already there)."""
        def job():
            if not self._wait_hands(should_stop):
                self.error = "hands"
                return
            self.step = "placing"
            reply = self.client.place(color, self.to_robot_cell(cell), level)
            if not reply.get("ok"):
                self.error = reply.get("error", "place failed")
                return
            self.step = "homing"
        self._run(job)

    def remove(self, cells: list[str], should_stop: Callable[[], bool] = lambda: False) -> None:
        def job():
            for i, cell in enumerate(cells):
                if not self._wait_hands(should_stop):
                    self.error, self.left_over = "hands", cells[i:]
                    return
                self.step = "removing"
                reply = self.client.pick(self.to_robot_cell(cell), 0)
                if not reply.get("ok"):
                    self.error, self.left_over = reply.get("error", "pick failed"), cells[i:]
                    return
            self.step = "homing"
        self._run(job)
