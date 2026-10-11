"""One look at the table: hands up, the camera sees a settled board, then the robot's eyes read it.

The same steps buddy/session.py (Picture Recall) runs inline, as a reusable piece for the other
games. poll() never blocks: the robot's eyes run on a worker thread (a vision-model call can take a few
seconds), so the phone keeps getting fresh state the whole time.

Statuses from poll():
  hands     the camera sees hands (or an arm) over the grid
  settling  waiting for a board that settled after start()
  looking   the robot's eyes are reading the board
  nudge     a block sits on a cell line ("cell" says where); the scan restarts by itself
  stuck     nothing settled for STUCK_S: offer "Scan again"
  done      "board" is ready (the robot's reading, or the camera's if the robot's eyes failed)
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from buddy.recall import has_unknown
from buddy.table import RobotEyes, TableBoard, blocks_to_board, board_diff

STUCK_S = 15.0


class Scanner:
    def __init__(self, table: TableBoard, eyes: RobotEyes, clock=time.time):
        self.table, self.eyes, self.clock = table, eyes, clock
        self.lock = threading.Lock()
        self.active = False
        self.last_sync: Optional[dict] = None
        self._reset()

    def _reset(self) -> None:
        self.since = 0.0
        self.started = 0.0
        self.camera: Optional[dict] = None
        self.busy = False
        self.result: Optional[dict] = None

    def start(self) -> None:
        with self.lock:
            self._reset()
            self.active = True
            self.since, self.started = self.table.begin_scan(), self.clock()

    def stop(self) -> None:
        with self.lock:
            self.active = False

    def _look(self) -> None:
        self.busy = True

        def job():
            try:
                result = self.eyes.scan()
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            with self.lock:
                self.result, self.busy = result, False
        threading.Thread(target=job, daemon=True).start()

    def poll(self) -> dict:
        with self.lock:
            if not self.active:
                return {"status": "idle"}
            if self.result is not None:
                res, camera = self.result, self.camera
                if res.get("ok"):
                    board, source = blocks_to_board(res["blocks"]), "robot"
                else:
                    board, source = camera, "camera"
                diffs = board_diff(board, camera) if source == "robot" else {}
                self.last_sync = {"t": round(self.clock(), 1), "source": source, "agree": not diffs, "diffs": diffs,
                                  "naming": res.get("naming"), "error": None if source == "robot" else res.get("error"),
                                  "robot": board if source == "robot" else None, "camera": camera}
                if has_unknown(board):
                    cell = next(c for c, s in board.items() if "unknown" in s)
                    self._reset()
                    self.since, self.started = self.table.begin_scan(), self.clock()
                    return {"status": "nudge", "cell": cell}
                self.active = False
                return {"status": "done", "board": board, "sync": self.last_sync, "blocks": res.get("blocks")}
            if self.busy:
                return {"status": "looking"}
            board = self.table.scanned(self.since)
            if board is not None:
                self.camera = board
                self._look()
                return {"status": "looking"}
            if not self.table.hands_away():
                return {"status": "hands"}
            if self.clock() - self.started > STUCK_S:
                return {"status": "stuck"}
            return {"status": "settling"}
