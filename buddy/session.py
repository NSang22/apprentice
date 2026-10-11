"""The Picture Recall game loop: phases, timers, Pip's help, the child's profile, and a log.

Phases:
  home     the game picker, nothing running
  clear    (table) waiting for an empty board before the next pattern
  show     the pattern is visible for LOOK_S[difficulty] seconds
  build    the pattern is hidden; the child builds until the timer runs out
  scan     (table) "Hands up!": waiting for a board that settled after time ran out
  choice   not right on the first try: the child picks Hint or Fix
  helping  Pip points at one square (hint) or takes the wrong blocks off (fix)
  result   right, or the second try is over; shows the answer, then the next round
  over     the lesson summary

After a hint or a fix the child gets RETRY_S seconds on the same pattern, then the round ends
either way. Stars: 3 right first time, 2 right after help, 1 for trying.

Board source: "table" (perception's board + Pip's arm, buddy/table.py) or "practice" (the phone
is the board and Pip's help is shown on screen). If the arm is not running, help falls back to
the screen: the hint is highlighted and the child takes the wrong blocks off.

Time-based moves happen in tick(), which every state read and command calls. Each round is
appended to data/buddy/sessions/<id>.jsonl, never rewritten; the profile (name, avatar, XP, best
memory, days played, lesson settings) is data/buddy/profile.json.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Optional

from buddy import recall
from buddy.table import Pip, TableBoard
from shared.config import CELLS, REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "buddy"
PROFILE_PATH = DATA_DIR / "profile.json"
SESSIONS_DIR = DATA_DIR / "sessions"

AVATARS = ("teal", "purple", "red", "yellow")
NAME_MAX = 16
SOURCES = ("table", "practice")
SCAN_STUCK_S = 15.0         # no settled board this long after "hands up": offer Scan again
PRACTICE_HINT_S = 3.0       # practice mode: how long Pip "points"
PRACTICE_REMOVE_S = 1.2     # ... and how long each removal takes

GAMES = [
    {"id": "recall", "title": "Picture Recall", "desc": "Look, remember, build it!", "ready": True},
    {"id": "turns", "title": "Take Turns", "desc": "Build it together with Pip", "ready": False},
    {"id": "teach", "title": "Teach the Robot", "desc": "You build, Pip learns", "ready": False},
]


def default_colors() -> list[str]:
    try:
        from perception.robot_bridge import default_active_colors
        return default_active_colors()
    except Exception:
        return ["red", "yellow", "teal", "purple"]


def empty_board() -> dict[str, list[str]]:
    return {c: [] for c in CELLS}


class Profile:
    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.data = {"name": "", "avatar": "teal", "xp": 0, "best_span": 0, "days": [], "lessons": 0,
                     "stars": 0, "difficulty": "easy", "build_s": 15, "source": "practice"}
        try:
            self.data.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def played_today(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today not in self.data["days"]:
            self.data["days"].append(today)

    def streak(self) -> int:
        """Days in a row up to today (or up to yesterday, if not played yet today)."""
        days = set(self.data["days"])
        t = time.time()
        if time.strftime("%Y-%m-%d", time.localtime(t)) not in days:
            t -= 86400
        n = 0
        while time.strftime("%Y-%m-%d", time.localtime(t)) in days:
            n += 1
            t -= 86400
        return n


class BuddySession:
    def __init__(self, table: Optional[TableBoard] = None, pip: Optional[Pip] = None,
                 clock=time.time, rng: Optional[random.Random] = None):
        self.clock = clock
        self.rng = rng or random.Random()
        self.lock = threading.RLock()
        self.profile = Profile()
        self.colors = default_colors()
        self.table = table or TableBoard()
        self.pip = pip or Pip(self.table)
        self._reset()

    def _reset(self) -> None:
        self.phase = "home"
        self.game: Optional[str] = None
        self.session_id: Optional[str] = None
        self.round = 0
        self.attempt = 1
        self.helped: Optional[str] = None       # "hint" or "fix" once chosen this round
        self.pattern: dict[str, str] = {}
        self.board = empty_board()               # practice: the phone's board
        self.seen = empty_board()                # the board that was graded
        self.until = 0.0                         # end of the current timed phase
        self.scan_since = 0.0
        self.scan_stuck = False
        self.grade: Optional[recall.Grade] = None
        self.hint: Optional[str] = None
        self.removing: list[str] = []
        self.arm_used = False
        self.stars: list[int] = []
        self.xp = 0
        self.gained = 0
        self.mood, self.say = "happy", ""
        self.rounds: list[dict] = []

    @property
    def source(self) -> str:
        return self.profile.data["source"]

    # ------------------------------------------------------------ settings and profile

    def set_profile(self, name: str, avatar: str) -> None:
        with self.lock:
            self.profile.data["name"] = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -'")[:NAME_MAX]
            if avatar in AVATARS:
                self.profile.data["avatar"] = avatar
            self.profile.save()

    def set_settings(self, difficulty: Optional[str], build_s: Optional[int], source: Optional[str]) -> None:
        with self.lock:
            if difficulty in recall.LOOK_S:
                self.profile.data["difficulty"] = difficulty
            if build_s in recall.BUILD_S:
                self.profile.data["build_s"] = build_s
            if source in SOURCES and self.phase == "home":
                self.profile.data["source"] = source
            self.profile.save()

    # ------------------------------------------------------------ commands

    def start(self, game: str) -> None:
        with self.lock:
            if game != "recall":
                raise ValueError("That game is coming soon!")
            if self.source == "table" and not self.table.ready():
                raise ValueError("I can't see the table yet. Start the camera first (scripts/jetson_up.sh), "
                                 "or switch to Practice.")
            self._reset()
            self.game = game
            self.session_id = time.strftime("b-%Y%m%d-%H%M%S")
            p = self.profile.data
            self._log("session_start", {"game": game, "source": self.source, "colors": self.colors,
                                        "difficulty": p["difficulty"], "build_s": p["build_s"],
                                        "profile": p.get("name", "")})
            self._next_round()

    def set_cell(self, cell: str, color: Optional[str]) -> None:
        with self.lock:
            self.tick()
            if self.phase != "build" or self.source != "practice" or cell not in CELLS:
                return
            if color is not None and color not in self.colors:
                return
            self.board[cell] = [color] if color else []

    def rescan(self) -> None:
        with self.lock:
            if self.phase == "scan":
                self.scan_since, self.scan_stuck = self.table.begin_scan(), False
                self.until = self.clock() + SCAN_STUCK_S

    def choose(self, kind: str) -> None:
        """The child's pick after a miss: "hint" (Pip points) or "fix" (Pip takes wrong blocks off)."""
        with self.lock:
            if self.phase != "choice" or kind not in ("hint", "fix") or self.grade is None:
                return
            self.helped, self.attempt = kind, 2
            g = self.grade
            arm = self.source == "table" and self.pip.reachable()
            self.arm_used = arm
            self._log("help", {"round": self.round, "kind": kind, "arm": arm})
            if kind == "hint":
                self.hint = recall.hint_cell(self.pattern, g)
                self.mood = "look"
                self.say = recall.hint_line(self.pattern, g, self.hint) + (" Look where I'm pointing!" if arm else "")
                if arm:
                    self.pip.point(self.hint, should_stop=lambda: self.phase != "helping")
                self.until = self.clock() + PRACTICE_HINT_S
            else:
                self.removing = recall.fix_cells(g)
                if not self.removing:
                    self._start_build(f"Nothing to take off! Just add the missing blocks. {recall.RETRY_S} seconds!")
                    return
                self.mood = "think"
                self.say = ("I'll take off the blocks that don't belong. Hands away, please!" if arm
                            else "Take off the blocks with a red X, then try again!" if self.source == "table"
                            else "I'll take off the blocks that don't belong!")
                if arm:
                    self.pip.remove(self.removing, should_stop=lambda: self.phase != "helping")
                self.until = self.clock() + PRACTICE_REMOVE_S * len(self.removing)
            self.phase = "helping"

    def next(self) -> None:
        with self.lock:
            if self.phase != "result":
                return
            if self.round >= len(recall.BLOCKS_PER_ROUND):
                self._finish("complete")
            else:
                self._next_round()

    def quit(self) -> None:
        with self.lock:
            if self.phase not in ("home", "over"):
                if self.rounds:  # keep what the child earned, even when stopping early
                    self._bank()
                self._log("session_end", {"outcome": "quit", **self.summary()})
            self._reset()

    # ------------------------------------------------------------ the loop

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            if self.phase == "clear":
                stable = self.table.scanned(0.0)
                if stable is not None and recall.block_count(stable) == 0 and self.table.hands_away():
                    self._show()
            elif self.phase == "show" and now >= self.until:
                self._start_build(at=self.until)   # the build clock starts when the look ended, not when polled
            if self.phase == "build" and now >= self.until:
                if self.source == "practice":
                    self._graded(self.board)
                else:
                    self.phase = "scan"
                    self.scan_since, self.scan_stuck = self.table.begin_scan(), False
                    self.until = now + SCAN_STUCK_S
                    self.mood, self.say = "look", "Time's up! Hands up so I can see!"
            elif self.phase == "scan":
                board = self.table.scanned(self.scan_since)
                if board is None:
                    if not self.table.hands_away():
                        self.mood, self.say = "look", "Hands up! I need to see the board."
                    elif now >= self.until and not self.scan_stuck:
                        self.scan_stuck = True
                        self.mood, self.say = "think", "Hmm, I can't see the board well. Tap Scan again!"
                elif recall.has_unknown(board):
                    cell = next(c for c, s in board.items() if "unknown" in s)
                    self.mood, self.say = "think", f"I can't tell the color on {cell}. Nudge it to the middle!"
                    self.scan_since = self.table.begin_scan()
                else:
                    self._graded(board)
            elif self.phase == "helping":
                done = (not self.pip.busy) if self.arm_used else now >= self.until
                if done:
                    say = (f"{recall.hint_line(self.pattern, self.grade, self.hint)} You have {recall.RETRY_S} seconds!"
                           if self.helped == "hint" else f"Now add the right blocks. You have {recall.RETRY_S} seconds!")
                    if self.helped == "fix":
                        if self.source == "practice":
                            for cell in self.removing:
                                self.board[cell] = []
                            self.removing = []
                        elif self.arm_used:
                            self.removing = self.pip.left_over if self.pip.error else []
                        if self.removing:  # the arm could not (or is not running): the child takes them off
                            say = f"Can you take off the blocks with a red X? Then add the right ones. {recall.RETRY_S} seconds!"
                    self._start_build(say)

    def _next_round(self) -> None:
        self.round += 1
        self.attempt, self.helped, self.hint, self.removing, self.grade = 1, None, None, [], None
        self.pattern = recall.make_pattern(recall.BLOCKS_PER_ROUND[self.round - 1], self.colors, self.rng)
        self.board = empty_board()
        if self.source == "table":
            stable = self.table.scanned(0.0)
            if stable is None or recall.block_count(stable) > 0:
                self.phase = "clear"
                self.mood, self.say = "happy", "Clear the board, and I'll show you the next pattern!"
                return
        self._show()

    def _show(self) -> None:
        self.phase = "show"
        self.until = self.clock() + recall.LOOK_S[self.profile.data["difficulty"]]
        n = len(self.pattern)
        self.mood, self.say = "look", f"Remember where the block goes!" if n == 1 else f"Remember all {n} blocks!"

    def _start_build(self, say: Optional[str] = None, at: Optional[float] = None) -> None:
        self.phase = "build"
        self.until = (at or self.clock()) + (recall.RETRY_S if self.attempt == 2 else self.profile.data["build_s"])
        if self.attempt == 1:
            self.mood = "think"
            self.say = "Now build it on the table!" if self.source == "table" else "Now build it! Pick a color, tap a square."
        else:
            self.mood = "encourage"
            self.say = say or f"You have {recall.RETRY_S} seconds to fix it!"

    def _graded(self, board: dict) -> None:
        g = recall.grade(self.pattern, board)
        self.grade, self.seen = g, {c: list(s) for c, s in board.items()}
        entry = {"round": self.round, "attempt": self.attempt, "blocks": g.total, "pattern": self.pattern,
                 "built": recall.flat(board), **g.to_dict()}
        if not g.perfect and self.attempt == 1:
            self._log("attempt", entry)
            self.phase = "choice"
            self.mood, self.say = recall.mood_line(g, self.profile.data.get("name") or None)
            return
        stars = recall.stars_for(g, helped=self.attempt == 2)
        self.gained = recall.xp_for(g, stars)
        self.xp += self.gained
        self.stars.append(stars)
        if g.perfect:
            self.profile.data["best_span"] = max(self.profile.data["best_span"], g.total)
        self.mood, self.say = recall.mood_line(g, self.profile.data.get("name") or None, helped=self.attempt == 2)
        entry.update({"stars": stars, "xp": self.gained, "helped": self.helped,
                      "difficulty": self.profile.data["difficulty"], "build_s": self.profile.data["build_s"]})
        self.rounds.append(entry)
        self._log("round", entry)
        self.phase = "result"

    def _bank(self) -> None:
        self.profile.data["xp"] += self.xp
        self.profile.data["stars"] = self.profile.data.get("stars", 0) + sum(self.stars)
        self.profile.played_today()
        self.profile.save()

    def _finish(self, outcome: str) -> None:
        self.phase = "over"
        self.profile.data["lessons"] += 1
        self._bank()
        self._log("session_end", {"outcome": outcome, **self.summary()})
        name = self.profile.data.get("name") or "friend"
        self.mood, self.say = "cheer", f"Lesson complete! Great job, {name}!"

    def summary(self) -> dict:
        played = len(self.rounds)
        first_try = sum(1 for r in self.rounds if r["perfect"] and r["attempt"] == 1)
        right = sum(r["right"] for r in self.rounds)
        total = sum(r["total"] for r in self.rounds)
        errors: dict[str, int] = {}
        for r in self.rounds:
            for k, v in r["errors"].items():
                errors[k] = errors.get(k, 0) + v
        return {
            "rounds": played,
            "first_try": first_try,
            "hints": sum(1 for r in self.rounds if r["helped"] == "hint"),
            "fixes": sum(1 for r in self.rounds if r["helped"] == "fix"),
            "stars": sum(self.stars),
            "max_stars": 3 * played,
            "accuracy": round(right / total, 2) if total else 0.0,
            "memory_span": max((r["total"] for r in self.rounds if r["perfect"] and r["attempt"] == 1), default=0),
            "xp": self.xp,
            "errors": errors,
        }

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
            now = self.clock()
            p = self.profile.data
            # practice: the phone's board. table: what the camera graded while Pip reacts to it,
            # otherwise the live camera board (the phone covers it while the pattern is hidden).
            if self.source == "practice":
                board = self.board
            elif self.phase in ("choice", "helping", "result"):
                board = self.seen
            else:
                board = self.table.latest or empty_board()
            return {
                "phase": self.phase,
                "game": self.game,
                "games": GAMES,
                "colors": self.colors,
                "settings": {"difficulty": p["difficulty"], "build_s": p["build_s"], "source": p["source"],
                             "look_s": recall.LOOK_S, "build_options": list(recall.BUILD_S), "retry_s": recall.RETRY_S},
                "profile": {k: p[k] for k in ("name", "avatar", "xp", "best_span", "lessons", "stars")}
                           | {"streak": self.profile.streak(), "days": len(p["days"])},
                "table": self.table.status(),
                "pip": {"busy": self.pip.busy, "step": self.pip.step, "arm": self.arm_used},
                "round": self.round,
                "rounds": len(recall.BLOCKS_PER_ROUND),
                "attempt": self.attempt,
                "helped": self.helped,
                "xp": self.xp,
                "gained": self.gained,
                "stars": self.stars,
                "round_stars": self.stars[-1] if self.phase == "result" and self.stars else 0,
                # the pattern stays hidden while the child builds
                "pattern": self.pattern if self.phase in ("show", "result") else None,
                "pattern_size": len(self.pattern),
                "board": board,
                "time_left": round(max(0.0, self.until - now), 2) if self.phase in ("show", "build") else 0,
                "time_total": (recall.LOOK_S[p["difficulty"]] if self.phase == "show"
                               else recall.RETRY_S if self.attempt == 2 else p["build_s"]),
                "scan_stuck": self.scan_stuck,
                "grade": self.grade.to_dict() if self.grade and self.phase in ("result", "choice") else None,
                "hint": self.hint,
                "removing": self.removing if self.phase in ("helping", "build") else [],
                "mood": self.mood,
                "say": self.say,
                "summary": self.summary() if self.phase == "over" else None,
            }
