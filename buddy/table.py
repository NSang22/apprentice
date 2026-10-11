"""The real table for Buddy: the camera's board (perception ws :8765) and Pip's arm (robot API :8770).

Neither the camera nor the arm is opened here. scripts/jetson_up.sh starts perception.publish
(which owns the webcam) and robot.api (which owns the servo bus); Buddy is one more client of
each, like brain.server.

Board: every GridState goes through brain.stabilizer, unchanged. scan() resets it, so the board
it returns settled after the scan began: the child's final build, never one from before time ran
out. Hands over the grid keep frames out of the stabilizer, so a scan waits for "hands up".

Arm: robot/api.py names cells letter = row, number = column (robot/README.md, decision 1);
shared/config.py and perception name them letter = column, number = row. Both index the same
sheet frame (x along tag 0 -> 1, y along tag 0 -> 3), so the names are transposed: shared C1 is
robot A3. to_robot_cell() is the only place Buddy crosses that boundary.

Safety: every arm move waits until the camera saw hands clear in a fresh frame, runs one at a
time, and only points (hover) or takes a block off (pick). Pip announces each move on the phone.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Callable, Optional

from websockets.asyncio.client import connect

from brain.robot_client import HttpRobotClient
from brain.stabilizer import Stabilizer
from shared.config import GRID_WS_PORT, ROBOT_HTTP_PORT
from shared.schemas import GridState

FRESH_S = 1.0             # a frame older than this does not count as "the camera sees it"
HANDS_WAIT_S = 20.0       # how long an arm move waits for hands to leave the board
HOVER_HOLD_S = 3.0        # Pip points this long, then goes home


def to_robot_cell(cell: str) -> str:
    """shared/perception cell name -> robot/api.py cell name (a transpose, see module docstring)."""
    col, row = "ABC".index(cell[0]), int(cell[1])
    return "ABC"[row - 1] + str(col + 1)


class TableBoard:
    """Latest camera frame plus a stabilized board, fed by perception's GridState stream."""

    def __init__(self, url: str = f"ws://127.0.0.1:{GRID_WS_PORT}"):
        self.url = url
        self.lock = threading.Lock()
        self.stabilizer = Stabilizer()
        self.connected = False
        self.last_t = 0.0                    # when the last frame arrived (time.time)
        self.hands_clear = False
        self.arm_clear = False
        self.latest: Optional[dict] = None   # raw board of the last frame
        self.stable: Optional[dict] = None   # last board the stabilizer emitted
        self.stable_t = 0.0

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
            gs = GridState.from_dict(json.loads(msg))
        except Exception:
            return
        with self.lock:
            self.last_t = time.time()
            self.hands_clear, self.arm_clear, self.latest = gs.hands_clear, gs.arm_clear, gs.board
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


class Pip:
    """Pip's arm moves, one at a time on a worker thread, so the phone never waits on a motion."""

    def __init__(self, board: TableBoard, base_url: str = f"http://127.0.0.1:{ROBOT_HTTP_PORT}"):
        self.board = board
        self.client = HttpRobotClient(base_url)
        self.busy = False
        self.step = ""                       # what Pip is doing, for the phone
        self.error: Optional[str] = None
        self.left_over: list[str] = []       # cells Pip could not clear (the child takes those off)

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
            reply = self.client.hover(to_robot_cell(cell))
            if not reply.get("ok"):
                self.error = reply.get("error", "hover failed")
                return
            self.step = "pointing"                  # arrived: hold so the child sees it, then home
            time.sleep(HOVER_HOLD_S)
            self.step = "homing"
        self._run(job)

    def remove(self, cells: list[str], should_stop: Callable[[], bool] = lambda: False) -> None:
        def job():
            for i, cell in enumerate(cells):
                if not self._wait_hands(should_stop):
                    self.error, self.left_over = "hands", cells[i:]
                    return
                self.step = "removing"
                reply = self.client.pick(to_robot_cell(cell), 0)
                if not reply.get("ok"):
                    self.error, self.left_over = reply.get("error", "pick failed"), cells[i:]
                    return
            self.step = "homing"
        self._run(job)
