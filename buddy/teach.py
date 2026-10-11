"""Teach the Robot: the child builds anything, Pip learns the steps, and can rebuild it later.

The child taps Start, builds on the grid (at most RECORD_S seconds, or until "I'm done!"), and the
recording goes through the team's extractor (brain/extractor.py, from video-to-lesson): settled
boards, history-based stacks, corrections (a block taken back off deletes its step), and the gate.
The result is an ordinary Lesson, saved under data/buddy/lessons/ with the child's name for it.
Later, "Build it, Pip!" rebuilds it step by step with the arm (buddy/teach_session.py).

What is recorded on the table:
  frames   every GridState perception sent while recording (what the extractor reads)
  video    the camera's frames from perception's frame server, VIDEO_FPS, saved as WebM (VP8: the
           Jetson's OpenCV cannot write H.264, and phones play WebM) so the lesson has a real source
           video the child can watch back (data/buddy/videos/<lesson id>.webm)
  colors   one look by the robot's own eyes at the end: the top block of each cell, so the colors
           Pip rebuilds with are the colors Pip sees (a "manual" color source for the extractor)

Rules for the kids' version: blocks at most MAX_HIGH high; the robot's bank holds BANK_SIZE blocks,
so a rebuild asks for the next colors as it goes.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from shared.config import COLORS, REPO_ROOT, UNKNOWN
from shared.schemas import GridState, Lesson

RECORD_S = 60.0            # the longest recording
MAX_HIGH = 2               # blocks per stack in this game
BANK_SIZE = 2              # blocks the robot's bank beside the sheet holds at a time
VIDEO_FPS = 8.0
VIDEO_WIDTH = 960
VIDEO_EXT, VIDEO_FOURCC = ".webm", "VP80"
FRAME_URL = "http://127.0.0.1:8771/frame"
LESSONS_DIR = REPO_ROOT / "data" / "buddy" / "lessons"
VIDEOS_DIR = REPO_ROOT / "data" / "buddy" / "videos"

# What the waiting child sees while Pip works, in order. Each is a real step (see learn()).
STAGES = [
    ("rewind", "Rewinding the tape"),
    ("flummox", "Flummoxing"),
    ("count", "Counting blocks"),
    ("squint", "Squinting at colors"),
    ("imagine", "Imagining"),
    ("write", "Writing it down"),
    ("check", "Checking my work"),
]

# Gate checks -> what Pip says. Kind, no blame, and always a way forward.
PROBLEMS = {
    "one_cube_per_change": "I got confused when more than one block moved at once. Let's teach it again, one block at a time!",
    "nothing": "I didn't see any blocks! Let's try again. Build something on the grid while I watch.",
    "too_high": f"Wow, that's tall! In this game, towers can be {MAX_HIGH} blocks high. Let's teach it again!",
    "unknown_color": "I couldn't tell the color of one block. Let's try again with the blocks in the middle of their squares!",
    "not_empty": "The board needs to start empty. Clear it, and let's try again!",
    "one_color": "I saw two blocks of the same color. Each color can be used once, so I don't mix them up!",
}
DEFAULT_PROBLEM = "Something got mixed up in my head. Let's teach it again!"


def kid_problem(reasons: dict[str, str]) -> str:
    for check, line in PROBLEMS.items():
        if any(check in name for name in reasons):
            return line
    return DEFAULT_PROBLEM


# ---------------------------------------------------------------- recording on the table


class VideoRecorder:
    """Saves the camera's frames (perception's frame server, which owns the webcam) as a WebM video."""

    def __init__(self, path: Path, url: str = FRAME_URL, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH):
        self.path, self.url, self.fps, self.width = path, url, fps, width
        self.frames = 0
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _grab(self):
        import numpy as np
        with urllib.request.urlopen(self.url, timeout=2) as r:
            shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
            return np.frombuffer(r.read(), np.uint8).reshape(shape)

    def _run(self) -> None:
        import cv2
        writer = None
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = self._grab()
                h, w = frame.shape[:2]
                size = (self.width, int(h * self.width / w))
                if writer is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*VIDEO_FOURCC), self.fps, size)
                writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
                self.frames += 1
                self._stop.wait(max(0.0, 1.0 / self.fps - (time.monotonic() - started)))
        except Exception as e:  # no video is not a reason to lose the lesson
            self.error = str(e)
        finally:
            if writer is not None:
                writer.release()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[Path]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.path if self.frames else None


# ---------------------------------------------------------------- learning (the extractor)


def rebase(frames: list[dict]) -> list[GridState]:
    """Recorded GridState dicts -> GridStates with t in seconds since the first frame (as the
    extractor's own live recorder does)."""
    out: list[GridState] = []
    t0 = None
    for d in frames:
        try:
            gs = GridState.from_dict(d)
        except Exception:
            continue
        t0 = gs.t if t0 is None else t0
        gs.t -= t0
        out.append(gs)
    return out


def learn(frames: list[dict], lesson_id: str, title: str, source_video: Optional[str],
          final_tops: Optional[dict[str, str]] = None,
          stage: Callable[[str, float], None] = lambda name, frac: None) -> dict:
    """Recording -> lesson, through brain/extractor.py exactly as its own command line runs it.

    final_tops: the robot eyes' color of each cell's top block at the end; it names those steps
    (the extractor's "manual" source, which wins over the camera's reads).
    Returns {ok: True, lesson, steps} or {ok: False, problem (what Pip says), reasons}.
    """
    from brain import extractor as ext

    stage("rewind", 0.0)
    states = rebase(frames)
    stage("rewind", 1.0)
    if not states:
        return {"ok": False, "problem": PROBLEMS["nothing"], "reasons": {"nothing": "no frames recorded"}}

    stage("flummox", 0.0)
    total = max(1, len(states))
    seen = [0]

    def on_settle(_s) -> None:
        seen[0] += 1
        stage("flummox", min(1.0, seen[0] / max(4, total / 20)))
    try:
        ex = ext.extract(states, on_settle=on_settle)
    except ext.ExtractError as e:
        return {"ok": False, "problem": DEFAULT_PROBLEM, "reasons": {"extract": str(e)}}
    stage("count", 0.5)
    ext.split_transients(ex)
    if not ex.steps:
        return {"ok": False, "problem": PROBLEMS["nothing"], "reasons": {"nothing": "no steps"}}
    if any(c.level >= MAX_HIGH for c in ex.steps):
        return {"ok": False, "problem": PROBLEMS["too_high"], "reasons": {"too_high": f"max {MAX_HIGH} high"}}
    stage("count", 1.0)

    stage("squint", 0.0)
    manual: dict[int, str] = {}
    if final_tops:
        top_level = {}
        for c in ex.steps:
            top_level[c.cell] = max(top_level.get(c.cell, -1), c.level)
        for i, c in enumerate(ex.steps):
            color = final_tops.get(c.cell)
            if c.level == top_level[c.cell] and color in COLORS:
                manual[i + 1] = color
    rows = ext.step_rows(ex, manual)
    stage("squint", 1.0)
    if any(r.color == UNKNOWN for r in rows):
        return {"ok": False, "problem": PROBLEMS["unknown_color"],
                "reasons": {"unknown_color": ", ".join(f"step {r.number} on {r.cell}" for r in rows if r.color == UNKNOWN)}}

    stage("imagine", 0.5)
    lesson = ext.build_lesson(rows, lesson_id, title, source_video)
    stage("imagine", 1.0)
    stage("check", 0.0)
    gate = ext.gate_lesson(lesson, ex, reference_empty=True, rows=rows)
    stage("check", 1.0)
    if not gate.gate.passed:
        return {"ok": False, "problem": kid_problem(gate.reasons), "reasons": gate.reasons}
    return {"ok": True, "lesson": lesson, "steps": [s.to_dict() for s in lesson.steps]}


def learn_taps(taps: list[dict], lesson_id: str, title: str) -> dict:
    """Practice mode: the phone's taps are the recording, so the steps are simply the taps that
    were not undone, in order."""
    from shared.schemas import Step, empty_board
    steps = [Step(t["color"], t["cell"], t["level"], round(t["t"], 3), None) for t in taps]
    if not steps:
        return {"ok": False, "problem": PROBLEMS["nothing"], "reasons": {"nothing": "no taps"}}
    final = empty_board()
    for s in steps:
        final[s.cell].append(s.color)
    lesson = Lesson(lesson_id, title, None, steps, final)
    return {"ok": True, "lesson": lesson, "steps": [s.to_dict() for s in steps]}


# ---------------------------------------------------------------- saved lessons


def new_id(name: str) -> str:
    from brain.extractor import new_lesson_id
    return new_lesson_id(name or "lesson", LESSONS_DIR)


def save(lesson: Lesson, taught_by: str, extra: Optional[dict] = None) -> Path:
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    path = LESSONS_DIR / f"{lesson.lesson_id}.json"
    data = {**lesson.to_dict(), "taught_by": taught_by, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def saved_lessons() -> list[dict]:
    out = []
    for path in sorted(LESSONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if LESSONS_DIR.is_dir() else []:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            out.append({"id": d["lesson_id"], "title": d["title"], "steps": len(d["steps"]),
                        "video": video_url(d["lesson_id"]),
                        "colors": [s["color"] for s in d["steps"]], "taught_by": d.get("taught_by", ""),
                        "created": d.get("created", "")})
        except (OSError, ValueError, KeyError):
            continue
    return out


def video_file(name: str) -> Optional[Path]:
    """The recording for a lesson id (or an unsaved recording's session id), if there is one."""
    path = VIDEOS_DIR / f"{Path(name).name}{VIDEO_EXT}"
    return path if path.is_file() and path.stat().st_size > 0 else None


def video_url(name: str) -> Optional[str]:
    return f"/buddy/video/{name}" if video_file(name) else None


def load(lesson_id: str) -> Lesson:
    path = LESSONS_DIR / f"{Path(lesson_id).name}.json"
    return Lesson.from_dict(json.loads(path.read_text(encoding="utf-8")))


def bank_colors(steps: list, start: int, size: int = BANK_SIZE) -> list[str]:
    """The colors the bank needs for the next `size` steps of a rebuild."""
    return [s.color for s in steps[start:start + size]]
