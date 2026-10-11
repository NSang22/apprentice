"""Fake robot: the README 5.3 API as a Python class that edits a shared fake board.

Brain tests and `server --robot fake --grid sim` use it in place of the arm. Same
interface and reply shapes as HttpRobotClient. Failure injection, keyed on the 1-based
count of place() calls:

  wrong_cell_on={2}  the 2nd place lands on the nearest other cell instead
  fail_on={3}        the 3rd place returns {"ok": false} and moves nothing

The arm leaves home on place, pick, and hover, and only returns on home(), like the real
arm. snapshot() gives (board copy, at_home) for a sim feed running on another thread.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Optional

from brain.checker import distance
from brain.robot_client import RobotReply, fail
from shared.config import CELLS, COLORS, MAX_LEVEL
from shared.schemas import Board, copy_board, empty_board


class FakeRobot:
    def __init__(
        self,
        board: Optional[Board] = None,
        wrong_cell_on: Iterable[int] = (),
        fail_on: Iterable[int] = (),
        motion_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.board = board if board is not None else empty_board()
        self.wrong_cell_on = set(wrong_cell_on)
        self.fail_on = set(fail_on)
        self.motion_s = motion_s
        self.sleep = sleep
        self.at_home = True
        self.busy = False
        self.last_error: Optional[str] = None
        self.place_count = 0
        self.calls: list[tuple] = []  # every call with its args, for tests
        self._lock = threading.Lock()

    # ------------------------------------------------------------ API

    def place(self, color: str, cell: str, level: int) -> RobotReply:
        self.calls.append(("place", color, cell, level))
        self.place_count += 1
        n = self.place_count
        if color not in COLORS or cell not in CELLS:
            return self._error(f"bad place target {color} {cell}")
        if n in self.fail_on:
            return self._error(f"injected failure on place {n}")
        if level != len(self.board[cell]):
            return self._error(f"{cell} has height {len(self.board[cell])}, cannot place at level {level}")
        if n in self.wrong_cell_on:
            cell = self._nearest_open_cell(cell)
        self._move(lambda: self.board[cell].append(color))
        return self._ok()

    def pick(self, cell: str, level: int) -> RobotReply:
        self.calls.append(("pick", cell, level))
        if cell not in CELLS:
            return self._error(f"bad pick cell {cell}")
        if level != len(self.board[cell]) - 1:
            return self._error(f"no cube on top of {cell} at level {level}")
        self._move(lambda: self.board[cell].pop())
        return self._ok()

    def hover(self, cell: str) -> RobotReply:
        self.calls.append(("hover", cell))
        if cell not in CELLS:
            return self._error(f"bad hover cell {cell}")
        self._move(lambda: None)
        return self._ok()

    def home(self) -> RobotReply:
        self.calls.append(("home",))
        self._move(lambda: None, end_home=True)
        return self._ok()

    def status(self) -> RobotReply:
        return {"ok": True, "busy": self.busy, "at_home": self.at_home, "last_error": self.last_error}

    # ------------------------------------------------------------ helpers

    def clear_board(self) -> None:
        """Take every cube off (stands in for a person clearing the table on reset)."""
        with self._lock:
            for stack in self.board.values():
                stack.clear()

    def snapshot(self) -> tuple[Board, bool]:
        with self._lock:
            return copy_board(self.board), self.at_home

    def _move(self, edit: Callable[[], object], end_home: bool = False) -> None:
        with self._lock:
            self.busy, self.at_home = True, False
        if self.motion_s:
            self.sleep(self.motion_s)
        with self._lock:
            edit()
            self.busy, self.at_home = False, end_home

    def _nearest_open_cell(self, cell: str) -> str:
        others = [c for c in CELLS if c != cell and len(self.board[c]) <= MAX_LEVEL]
        return min(others, key=lambda c: (distance(c, cell), len(self.board[c]), CELLS.index(c)))

    def _ok(self) -> RobotReply:
        self.last_error = None
        return {"ok": True}

    def _error(self, error: str) -> RobotReply:
        self.last_error = error
        return fail(error)
