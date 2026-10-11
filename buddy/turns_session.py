"""The Take Turns game loop (rules in buddy/turns.py): Pip and the child build a pattern together.

Phases:
  home         nothing running
  clear        (table) waiting for an empty board before the next pattern
  pip_turn     Pip places a block (the arm on the table; it just appears in practice)
  pip_check    (table) looking at the board: did Pip's block land on its square?
  need_supply  Pip could not place (no block of that color below the grid, or hands stayed on)
  child_turn   the child places a block, then taps "Your turn, Pip!"
  child_check  (table) looking at the board after the child's tap
  pointing     (table) Pip points at where a misplaced block goes
  round_done   the pattern is built: stars, then the next pattern
  over         the lesson summary

Pip goes first, then turns alternate one block each. When Pip misses its square the child is
asked to move Pip's block, and only that turn's own block counts against them if it is wrong.
Stars per round come from the child's slips only: 3 with none, 2 with one, 1 otherwise.

Board source as in Picture Recall: "table" (camera + the robot's own eyes, buddy/scanner.py) or
"practice" (the phone is the board, Pip's blocks appear after a short pause). Every turn is
appended to data/buddy/sessions/<id>.jsonl.
"""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Optional

from buddy import recall, turns
from buddy.scanner import Scanner
from buddy.session import SESSIONS_DIR, Profile, default_colors, empty_board
from buddy.table import Pip, RobotEyes, TableBoard

PRACTICE_PIP_S = 2.0        # practice: Pip "thinks" this long before its block appears


class TurnsSession:
    def __init__(self, table: TableBoard, eyes: RobotEyes, pip: Pip, profile: Profile,
                 clock=time.time, rng: Optional[random.Random] = None):
        self.table, self.eyes, self.pip, self.profile = table, eyes, pip, profile
        self.clock = clock
        self.rng = rng or random.Random()
        self.lock = threading.RLock()
        self.colors = default_colors()
        self.scanner = Scanner(table, eyes, clock)
        self._reset()

    def _reset(self) -> None:
        self.phase = "home"
        self.session_id: Optional[str] = None
        self.round = 0
        self.target: dict[str, str] = {}
        self.board = empty_board()           # practice: the phone's board
        self.seen = empty_board()            # table: the last board Pip's eyes read
        self.until = 0.0
        self.pip_move: Optional[tuple[str, str]] = None   # (cell, color) Pip is placing
        self.arm_used = False
        self.right_before = 0
        self.glow: Optional[str] = None      # the square the phone highlights
        self.mistakes = 0                    # the child's slips this round
        self.pip_misses = 0
        self.helped_pip = 0                  # times the child fixed Pip's block (whole lesson)
        self.stars: list[int] = []
        self.xp = 0
        self.gained = 0
        self.scan_stuck = False
        self.rounds: list[dict] = []
        self.mood, self.say = "happy", ""

    @property
    def source(self) -> str:
        return self.profile.data["source"]

    def current(self) -> dict:
        return self.board if self.source == "practice" else self.seen

    # ------------------------------------------------------------ commands

    def start(self) -> None:
        with self.lock:
            if self.source == "table" and not self.table.ready():
                raise ValueError("I can't see the table yet. Turn on the table (⚙), or switch to Practice.")
            self._reset()
            if self.source == "table":
                self.eyes.warm()
            self.session_id = time.strftime("t-%Y%m%d-%H%M%S")
            self._log("session_start", {"game": "turns", "source": self.source, "colors": self.colors,
                                        "profile": self.profile.data.get("name", "")})
            self._next_round()

    def set_cell(self, cell: str, color: Optional[str]) -> None:
        with self.lock:
            if self.phase != "child_turn" or self.source != "practice" or cell not in self.board:
                return
            if color is not None and color not in self.colors:
                return
            self.board[cell] = [color] if color else []

    def your_turn_pip(self) -> None:
        """The child's "Your turn, Pip!" tap: look at the board and judge the turn."""
        with self.lock:
            if self.phase != "child_turn":
                return
            if self.source == "practice":
                self._judge_child(self.board)
            else:
                self.phase, self.scan_stuck = "child_check", False
                self.scanner.start()
                self.mood, self.say = "look", "Hands up! Let me look."

    def retry(self) -> None:
        with self.lock:
            if self.phase == "need_supply":
                self._pip_turn()
            elif self.phase in ("pip_check", "child_check") and self.scan_stuck:
                self.scan_stuck = False
                self.scanner.start()

    def next(self) -> None:
        with self.lock:
            if self.phase != "round_done":
                return
            if self.round >= len(turns.BLOCKS_PER_ROUND):
                self._finish()
            else:
                self._next_round()

    def quit(self) -> None:
        with self.lock:
            self.scanner.stop()
            if self.phase not in ("home", "over") and self.rounds:
                self._bank()
            if self.session_id and self.phase not in ("home", "over"):
                self._log("session_end", {"outcome": "quit", **self.summary()})
            self._reset()

    # ------------------------------------------------------------ the loop

    def _next_round(self) -> None:
        self.round += 1
        self.target = recall.make_pattern(turns.BLOCKS_PER_ROUND[self.round - 1], self.colors, self.rng)
        self.board, self.seen = empty_board(), empty_board()
        self.mistakes, self.right_before, self.glow = 0, 0, None
        if self.source == "table":
            stable = self.table.scanned(0.0)
            if stable is None or recall.block_count(stable) > 0:
                self.phase = "clear"
                self.mood, self.say = "happy", "Clear the board, and we'll build the next one together!"
                return
        self._pip_turn(first=True)

    def _pip_turn(self, first: bool = False) -> None:
        move = turns.pip_next(self.target, self.current())
        if move is None:
            self._round_done()
            return
        cell, color = move
        self.pip_move, self.glow, self.phase = move, cell, "pip_turn"
        self.mood = "happy"
        self.say = ("I'll go first! " if first else "My turn! ") + f"I'll put a {color} block here."
        self.arm_used = self.source == "table" and self.pip.reachable()
        if self.arm_used:
            self.pip.place(cell, color, should_stop=lambda: self.phase != "pip_turn")
        elif self.source == "table":  # the arm is off: the child places Pip's block for it
            self.right_before = len(turns.look(self.target, self.seen).right) + 1
            self._child_turn(f"My arm is resting. Can you put my {color} block on the glowing square, "
                             "then add yours and tap Your turn, Pip?")
            return
        self.until = self.clock() + PRACTICE_PIP_S

    def _child_turn(self, say: Optional[str] = None) -> None:
        self.phase = "child_turn"
        self.mood = "encourage"
        self.say = say or "Your turn! Add one block from the picture, then tap Your turn, Pip!"

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            if self.phase == "clear":
                stable = self.table.scanned(0.0)
                if stable is not None and recall.block_count(stable) == 0 and self.table.hands_away():
                    self._pip_turn(first=True)
            elif self.phase == "pip_turn":
                if not self.arm_used:
                    if now >= self.until:  # practice: Pip's block appears
                        cell, color = self.pip_move
                        self.board[cell] = [color]
                        self._log("turn", {"round": self.round, "who": "pip", "cell": cell, "color": color, "landed": True})
                        self._after_pip(self.board, landed=True)
                elif not self.pip.busy:
                    if self.pip.error:
                        cell, color = self.pip_move
                        self._log("turn", {"round": self.round, "who": "pip", "cell": cell, "color": color,
                                           "error": self.pip.error})
                        self.phase, self.mood = "need_supply", "think"
                        self.say = ("Hands away from the board, please, so I can move! Then tap Try again."
                                    if self.pip.error == "hands" else
                                    f"I need a {color} block below the grid to pick up. Put one there and tap Try again!")
                    else:
                        self.phase, self.scan_stuck = "pip_check", False
                        self.scanner.start()
                        self.mood, self.say = "look", "Let me check where my block landed..."
            elif self.phase in ("pip_check", "child_check"):
                st = self.scanner.poll()
                status = st["status"]
                if status == "done":
                    self.seen = st["board"]
                    if self.phase == "pip_check":
                        cell, color = self.pip_move
                        landed = turns.pip_landed(self.target, self.seen, cell, color)
                        self._log("turn", {"round": self.round, "who": "pip", "cell": cell, "color": color,
                                           "landed": landed, "sync": st.get("sync")})
                        self._after_pip(self.seen, landed)
                    else:
                        self._judge_child(self.seen, sync=st.get("sync"))
                elif status == "hands":
                    self.mood, self.say = "look", "Hands up! I need to see the board."
                elif status == "nudge":
                    self.mood, self.say = "think", f"I can't tell the block on {st['cell']}. Nudge it to the middle of its square!"
                elif status == "looking":
                    self.mood, self.say = "look", "Pip is looking closely..."
                elif status == "stuck" and not self.scan_stuck:
                    self.scan_stuck = True
                    self.mood, self.say = "think", "Hmm, I can't see the board well. Tap Scan again!"
            elif self.phase == "pointing" and not self.pip.busy:
                self.phase = "child_turn"

    def _after_pip(self, board: dict, landed: bool) -> None:
        seen = turns.look(self.target, board)
        cell, color = self.pip_move
        if landed:
            self.right_before, self.glow = len(seen.right), None
            if seen.done:
                self._round_done()
            else:
                self._child_turn()
            return
        self.pip_misses += 1
        self.right_before, self.glow = len(seen.right) + 1, cell   # the child moves Pip's block too
        self._child_turn(f"Oops, I missed! Can you move my {color} block to the glowing square? "
                         "Then add yours and tap Your turn, Pip!")
        self.mood = "think"

    def _judge_child(self, board: dict, sync: Optional[dict] = None) -> None:
        r = turns.child_turn(self.target, board, self.right_before)
        seen: turns.Look = r["look"]
        self._log("turn", {"round": self.round, "who": "child", "result": r["result"], "right": len(seen.right),
                           "wrong": seen.wrong, "sync": sync})
        if r["result"] == "ok":
            if self.glow and self.pip_move and self.glow == self.pip_move[0]:
                self.helped_pip += 1        # they moved Pip's missed block for it
            self.right_before, self.glow = len(seen.right), None
            if seen.done:
                self._round_done()
            else:
                self._pip_turn()
            return
        if r["result"] == "none":
            self._child_turn("Put one block from the picture on the board first, then tap Your turn, Pip!")
            return
        self.mistakes += 1
        w, point = r["wrong"], r["point"]
        self.glow = point
        line = turns.wrong_line(w, point)
        if point and self.source == "table" and self.pip.reachable():
            self.phase, self.mood = "pointing", "look"
            self.say = line + " Look where I'm pointing!"
            self.pip.point(point, should_stop=lambda: self.phase != "pointing")
            return
        self._child_turn(line + " Then tap Your turn, Pip!")
        self.mood = "think"

    def _round_done(self) -> None:
        stars = turns.stars_for(self.mistakes)
        self.gained = 5 * stars + len(self.target)
        self.xp += self.gained
        self.stars.append(stars)
        entry = {"round": self.round, "blocks": len(self.target), "pattern": self.target, "mistakes": self.mistakes,
                 "pip_misses": self.pip_misses, "stars": stars, "xp": self.gained}
        self.rounds.append(entry)
        self._log("round", entry)
        self.phase, self.glow, self.mood = "round_done", None, "cheer"
        name = self.profile.data.get("name") or "friend"
        self.say = f"We built it together, {name}!" if self.mistakes == 0 else "We did it! Great teamwork!"

    def _bank(self) -> None:
        self.profile.data["xp"] += self.xp
        self.profile.data["stars"] = self.profile.data.get("stars", 0) + sum(self.stars)
        self.profile.played_today()
        self.profile.save()

    def _finish(self) -> None:
        self.phase = "over"
        self.profile.data["lessons"] += 1
        self._bank()
        self._log("session_end", {"outcome": "complete", **self.summary()})
        self.mood, self.say = "cheer", f"What a team! Thanks for building with me, {self.profile.data.get('name') or 'friend'}!"

    def summary(self) -> dict:
        return {"rounds": len(self.rounds), "stars": sum(self.stars), "max_stars": 3 * len(self.rounds),
                "mistakes": sum(r["mistakes"] for r in self.rounds), "pip_misses": self.pip_misses,
                "helped_pip": self.helped_pip, "xp": self.xp}

    def _log(self, kind: str, data: dict) -> None:
        if not self.session_id:
            return
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"t": round(self.clock(), 3), "session": self.session_id, "kind": kind, "data": data})
        with open(SESSIONS_DIR / f"{self.session_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------ view

    def state(self) -> dict:
        with self.lock:
            self.tick()
            if self.source == "practice":
                board = self.board
            elif self.phase in ("clear",):
                board = self.table.latest or empty_board()
            else:
                board = self.seen
            seen = turns.look(self.target, board) if self.target else None
            p = self.profile.data
            return {
                "phase": self.phase,
                "source": self.source,
                "colors": self.colors,
                "profile": {k: p[k] for k in ("name", "avatar", "xp", "stars")},
                "round": self.round,
                "rounds": len(turns.BLOCKS_PER_ROUND),
                "target": self.target,
                "board": board,
                "right": seen.right if seen else [],
                "glow": self.glow,
                "turn": "pip" if self.phase in ("pip_turn", "pip_check", "need_supply") else "child",
                "pip": {"busy": self.pip.busy, "step": self.pip.step, "arm": self.arm_used},
                "table": self.table.status(),
                "scan_stuck": self.scan_stuck,
                "stars": self.stars,
                "round_stars": self.stars[-1] if self.phase == "round_done" and self.stars else 0,
                "gained": self.gained,
                "mood": self.mood,
                "say": self.say,
                "sync": self.scanner.last_sync,
                "summary": self.summary() if self.phase == "over" else None,
            }
