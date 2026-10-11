"""The game loop around buddy/recall.py: phases, hearts, XP, the child's profile, and a log.

Phases (Picture Recall):
  home     the path screen, nothing running
  show     the pattern is visible for level.show_s seconds
  build    the pattern is hidden; the child builds it (practice: taps on the phone)
  result   graded, with the buddy's reaction; CONTINUE goes to the next round
  over     out of hearts or all rounds done: the lesson summary

One child, one session at a time (a single table). Time-based moves (show -> build) happen in
tick(), which every state read calls, so no background thread is needed.

Every round is appended to data/buddy/sessions/<id>.jsonl, never rewritten (like the brain's
journal). The profile (name, avatar, XP, best memory span, days played) is data/buddy/profile.json.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Optional

from buddy import recall
from shared.config import CELLS, REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "buddy"
PROFILE_PATH = DATA_DIR / "profile.json"
SESSIONS_DIR = DATA_DIR / "sessions"

HEARTS = 3
ROUNDS = 6                # rounds in one lesson
AVATARS = ("teal", "purple", "red", "yellow")
NAME_MAX = 16

GAMES = [
    {"id": "recall", "title": "Picture Recall", "desc": "Look, remember, build it!", "ready": True},
    {"id": "simon", "title": "Robot Simon", "desc": "Follow the robot's pointing", "ready": False},
    {"id": "change", "title": "Spot the Change", "desc": "What did the robot move?", "ready": False},
    {"id": "mirror", "title": "Mirror Me", "desc": "Build the other half", "ready": False},
]


def default_colors() -> list[str]:
    try:
        from perception.robot_bridge import default_active_colors
        return default_active_colors()
    except Exception:
        return ["red", "yellow", "teal", "purple"]


class Profile:
    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.data = {"name": "", "avatar": "teal", "xp": 0, "best_span": 0, "days": [], "lessons": 0}
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
    def __init__(self, clock=time.time, rng: Optional[random.Random] = None):
        self.clock = clock
        self.rng = rng or random.Random()
        self.lock = threading.RLock()
        self.profile = Profile()
        self.colors = default_colors()
        self.source = "tap"           # "tap": practice on the phone; "camera" comes later
        self._reset()

    def _reset(self) -> None:
        self.phase = "home"
        self.game: Optional[str] = None
        self.session_id: Optional[str] = None
        self.level_n = 1
        self.round = 0
        self.hearts = HEARTS
        self.xp = 0
        self.combo = 0
        self.pattern: dict[str, str] = {}
        self.board = {c: [] for c in CELLS}
        self.show_until = 0.0
        self.build_started = 0.0
        self.grade: Optional[recall.Grade] = None
        self.gained = 0
        self.mood, self.say = "happy", ""
        self.rounds: list[dict] = []

    # ------------------------------------------------------------ commands

    def set_profile(self, name: str, avatar: str) -> None:
        with self.lock:
            name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -'")[:NAME_MAX]
            self.profile.data["name"] = name
            if avatar in AVATARS:
                self.profile.data["avatar"] = avatar
            self.profile.save()

    def start(self, game: str) -> None:
        with self.lock:
            if game != "recall":
                raise ValueError("That game is coming soon!")
            self._reset()
            self.game = game
            self.session_id = time.strftime("b-%Y%m%d-%H%M%S")
            self._log("session_start", {"game": game, "source": self.source, "colors": self.colors,
                                        "profile": self.profile.data.get("name", "")})
            self._next_round()

    def set_cell(self, cell: str, color: Optional[str]) -> None:
        with self.lock:
            self.tick()
            if self.phase != "build" or self.source != "tap" or cell not in CELLS:
                return
            if color is not None and color not in self.colors:
                return
            self.board[cell] = [color] if color else []

    def check(self) -> None:
        with self.lock:
            self.tick()
            if self.phase != "build":
                return
            g = recall.grade(self.pattern, self.board)
            self.grade = g
            self.gained = recall.xp_for(g, self.level_n)
            self.xp += self.gained
            self.mood, self.say = recall.mood_line(g, self.profile.data.get("name") or None)
            build_s = round(self.clock() - self.build_started, 1)
            if g.perfect:
                self.combo += 1
                self.profile.data["best_span"] = max(self.profile.data["best_span"], g.total)
            else:
                self.combo = 0
                self.hearts -= 1
            entry = {"round": self.round, "level": self.level_n, "blocks": g.total,
                     "show_s": recall.level(self.level_n).show_s, "build_s": build_s,
                     "pattern": self.pattern, "built": recall.flat(self.board), **g.to_dict(),
                     "xp": self.gained}
            self.rounds.append(entry)
            self._log("round", entry)
            if g.perfect:
                self.level_n += 1
            self.phase = "result"

    def next(self) -> None:
        with self.lock:
            if self.phase != "result":
                return
            if self.hearts <= 0 or self.round >= ROUNDS:
                self._finish("out_of_hearts" if self.hearts <= 0 else "complete")
            else:
                self._next_round()

    def quit(self) -> None:
        with self.lock:
            if self.phase not in ("home", "over"):
                if self.rounds:  # keep what the child earned, even when stopping early
                    self.profile.data["xp"] += self.xp
                    self.profile.played_today()
                    self.profile.save()
                self._log("session_end", {"outcome": "quit", **self.summary()})
            self._reset()

    # ------------------------------------------------------------ internals

    def _next_round(self) -> None:
        lv = recall.level(self.level_n)
        self.round += 1
        self.pattern = recall.make_pattern(lv.blocks, self.colors, self.rng)
        self.board = {c: [] for c in CELLS}
        self.grade = None
        self.show_until = self.clock() + lv.show_s
        self.phase = "show"
        self.mood, self.say = "look", f"Remember where the {lv.blocks} blocks go!"

    def _finish(self, outcome: str) -> None:
        self.phase = "over"
        self.profile.data["xp"] += self.xp
        self.profile.data["lessons"] += 1
        self.profile.played_today()
        self.profile.save()
        s = self.summary()
        self._log("session_end", {"outcome": outcome, **s})
        self.mood = "cheer" if outcome == "complete" else "encourage"
        name = self.profile.data.get("name") or "friend"
        self.say = (f"Lesson complete! Great job, {name}!" if outcome == "complete"
                    else f"Great practice, {name}! Your brain got stronger today.")

    def tick(self) -> None:
        with self.lock:
            if self.phase == "show" and self.clock() >= self.show_until:
                self.phase = "build"
                self.build_started = self.clock()
                self.mood, self.say = "think", "Now build it on the table!" if self.source == "camera" \
                    else "Now build it! Pick a color, then tap a square."

    def summary(self) -> dict:
        played = len(self.rounds)
        right = sum(r["right"] for r in self.rounds)
        total = sum(r["total"] for r in self.rounds)
        errors: dict[str, int] = {}
        for r in self.rounds:
            for k, v in r["errors"].items():
                errors[k] = errors.get(k, 0) + v
        span = max((r["total"] for r in self.rounds if r["perfect"]), default=0)
        return {
            "rounds": played,
            "perfect_rounds": sum(1 for r in self.rounds if r["perfect"]),
            "accuracy": round(right / total, 2) if total else 0.0,
            "memory_span": span,
            "xp": self.xp,
            "avg_build_s": round(sum(r["build_s"] for r in self.rounds) / played, 1) if played else 0.0,
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
            show = self.phase in ("show", "result")
            p = self.profile.data
            return {
                "phase": self.phase,
                "game": self.game,
                "games": GAMES,
                "source": self.source,
                "colors": self.colors,
                "profile": {**p, "streak": self.profile.streak(), "days": len(p["days"])},
                "level": self.level_n,
                "round": self.round,
                "rounds": ROUNDS,
                "hearts": self.hearts,
                "max_hearts": HEARTS,
                "xp": self.xp,
                "gained": self.gained,
                "combo": self.combo,
                "pattern": self.pattern if show else None,   # hidden while the child builds
                "pattern_size": len(self.pattern),
                "board": self.board,
                "show_left": round(max(0.0, self.show_until - now), 2) if self.phase == "show" else 0,
                "show_s": recall.level(self.level_n).show_s,
                "grade": self.grade.to_dict() if self.grade else None,
                "mood": self.mood,
                "say": self.say,
                "summary": self.summary() if self.phase == "over" else None,
            }
