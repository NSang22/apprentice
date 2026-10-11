"""Teach the Robot: the child builds anything, Pip learns the steps, and can rebuild it later.

The child taps Start, builds on the grid (at most RECORD_S seconds, or until "I'm done!"), and the
recording goes through the team's extractor (brain/extractor.py, from video-to-lesson): settled
boards, history-based stacks, corrections (a block taken back off deletes its step), and the gate.
The result is an ordinary Lesson, saved under data/buddy/lessons/ with the child's name for it.
Later, "Build it, Pip!" rebuilds it step by step with the arm (buddy/teach_session.py).

How Pip learns on the table: the extractor runs on the recorded video itself (learn_video:
python -m brain.extractor --video in the main checkout, sampling 5 frames a second, with the robot's
colour memory and the Gemini fallback). If the video gives no answer, the live frames are used
instead (learn).

What is recorded on the table:
  frames   every GridState perception sent while recording (the fallback's input)
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
VIDEO_WIDTH = 1280        # the extractor finds the 4 tags in this video, so not too small
VIDEO_EXT, VIDEO_FOURCC = ".webm", "VP80"
FRAME_URL = "http://127.0.0.1:8771/frame"
LESSONS_DIR = REPO_ROOT / "data" / "buddy" / "lessons"
VIDEOS_DIR = REPO_ROOT / "data" / "buddy" / "videos"

# What the waiting child sees while Pip works, in order. Each is a real step (see learn()).
STAGES = [
    ("rewind", "Rewinding the tape"),
    ("listen", "Listening to what you said"),
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
    "replay_matches_final_frame": "When I built it in my head, one block came out a different color than the one on the board. "
                                  "Let's teach it again, with each block in the middle of its square!",
    "starts_empty": "The board needs to start empty. Clear it, and let's try again!",
    "one_cube_per_color": "I saw two blocks of the same color. Each color can be used once, so I don't mix them up!",
    "levels_match_stack_height": "I lost count of how tall a tower was. Let's teach it again, one block at a time!",
    "colors_known": "I couldn't tell the color of one block. Let's try again with the blocks in the middle of their squares!",
    "has_steps": "I didn't see any blocks! Let's try again. Build something on the grid while I watch.",
    "nothing": "I didn't see any blocks! Let's try again. Build something on the grid while I watch.",
    "too_high": f"Wow, that's tall! In this game, towers can be {MAX_HIGH} blocks high. Let's teach it again!",
    "unknown_color": "I couldn't tell the color of one block. Let's try again with the blocks in the middle of their squares!",
    "not_empty": "The board needs to start empty. Clear it, and let's try again!",
    "one_color": "I saw two blocks of the same color. Each color can be used once, so I don't mix them up!",
}
DEFAULT_PROBLEM = "Something went wrong on my side. A grown-up can see why on the debug page. Let's teach it again!"


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


MIC = "hw:CARD=Camera,DEV=0"   # the overhead webcam's own microphone (arecord -l: card "Camera")


class AudioRecorder:
    """The webcam microphone to an Opus file while the child builds (GStreamer: alsasrc, opusenc).
    The narration is a second source for the steps: Gemini transcribes it afterwards."""

    def __init__(self, path: Path, device: str = MIC):
        self.path, self.device = path, device
        self.proc = None
        self.error: Optional[str] = None

    def start(self) -> None:
        import subprocess
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.proc = subprocess.Popen(
                ["gst-launch-1.0", "-q", "-e", "alsasrc", f"device={self.device}", "!", "audioconvert", "!",
                 "audioresample", "!", "opusenc", "!", "oggmux", "!", "filesink", f"location={self.path}"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except OSError as e:  # no GStreamer: the lesson still records video
            self.error = str(e)

    def stop(self) -> Optional[Path]:
        import signal
        import subprocess
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)   # -e: SIGINT sends end-of-stream, so the file is closed properly
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc.returncode not in (0, None) and self.proc.stderr:
            self.error = (self.proc.stderr.read() or "").strip()[-300:] or self.error
        return self.path if self.path.is_file() and self.path.stat().st_size > 2000 else None


def mux(video: Path, audio: Path, out: Path) -> bool:
    """Video and narration in one WebM (VP8 + Opus) for playback, with GStreamer's webmmux."""
    import subprocess
    cmd = ["gst-launch-1.0", "-q", "-e", "webmmux", "name=m", "!", "filesink", f"location={out}",
           "filesrc", f"location={video}", "!", "matroskademux", "!", "queue", "!", "m.",
           "filesrc", f"location={audio}", "!", "oggdemux", "!", "opusparse", "!", "queue", "!", "m."]
    try:
        subprocess.run(cmd, timeout=60, capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.is_file() and out.stat().st_size > video.stat().st_size // 2


def check_narration(steps: list, placements: list[dict]) -> dict:
    """Spoken placements vs the steps the camera saw. The grid decides (AGENTS.md rule 2): this
    only fills each matching step's narration and lists where speech and grid disagree."""
    notes, matched = [], 0
    for p in placements:
        if not isinstance(p, dict) or p.get("action", "place") != "place":
            continue
        cell, color, text = p.get("cell"), p.get("color"), str(p.get("text") or "").strip()
        if cell not in {s.cell for s in steps}:
            if cell:
                notes.append(f"I heard \"{text or color + ' on ' + cell}\", but I didn't see a block on {cell}.")
            continue
        here = [s for s in steps if s.cell == cell]
        same = [s for s in here if s.color == color]
        if same:
            matched += 1
            if text and not same[0].narration:
                same[0].narration = text
        elif color in COLORS:
            notes.append(f"I heard {color} on {cell}, but I saw {' and '.join(s.color for s in here)} there.")
    return {"matched": matched, "notes": notes}


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
        in_play = set(active_colors())  # the robot also knows sky blue and orange: never let those in
        for i, c in enumerate(ex.steps):
            color = final_tops.get(c.cell)
            if c.level == top_level[c.cell] and color in in_play:
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


EXTRACT_TIMEOUT_S = 240.0   # the video run gives up after this long (the live frames take over)
EXTRACT_S_PER_VIDEO_S = 0.6  # first guess at run time per second of video, for the progress bar
# Lines the extractor prints, in the order it prints them -> the stage the bar is in by then.
EXTRACT_MILESTONES = [("sampling", "flummox"), ("hand band", "count"), ("gemini", "squint"), ("run time", "check")]


def learn_video(video: Path, repo: Path, title: str, video_s: float,
                stage: Callable[[str, float], None] = lambda name, frac: None, log: Callable[[str], None] = print) -> dict:
    """The recording's video -> lesson with the team's extractor on the video itself
    (python -m brain.extractor --video, 5 frames a second by default), run in `repo` (the main
    checkout) so it uses the calibration, robot colour memory and Gemini key everything else uses.

    Returns like learn(), or {ok: None, error} when the video run could not produce an answer
    (unreadable video, no tags in the first frame, a crash, a timeout): the caller then learns
    from the live frames instead.
    """
    import os
    import queue
    import subprocess
    import sys

    work = LESSONS_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{video.stem}.json"
    # the colors in play and the block size, as scripts/jetson_up.sh runs perception: the extractor's
    # default is every shared color, which the 4-color calibration refuses (exit code 2)
    cmd = [sys.executable, "-u", "-m", "brain.extractor", "--video", str(video.resolve()), "--out", str(out),
           "--title", title, "--colors", ",".join(active_colors()), "--block-mm", os.environ.get("BLOCK_MM", "47")]
    proc = subprocess.Popen(cmd, cwd=str(repo), env={**os.environ, "PYTHONPATH": str(repo), "PYTHONUNBUFFERED": "1"},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l.rstrip("\n")) for l in proc.stdout], daemon=True).start()

    names = [n for n, _ in STAGES]
    trace = {"command": " ".join(cmd[1:]), "cwd": str(repo), "lines": []}
    estimate = max(6.0, video_s * EXTRACT_S_PER_VIDEO_S)
    started, current = time.monotonic(), "flummox"
    stage(current, 0.0)
    while proc.poll() is None or not lines.empty():
        try:
            line = lines.get(timeout=0.3)
            log(line)
            trace["lines"].append(line)
            for key, name in EXTRACT_MILESTONES:
                if key in line and names.index(name) > names.index(current):
                    current = name
                    stage(current, 0.0)
        except queue.Empty:
            pass
        elapsed = time.monotonic() - started
        if elapsed > EXTRACT_TIMEOUT_S:
            proc.kill()
            return {"ok": None, "error": f"the video run took over {EXTRACT_TIMEOUT_S:.0f} s", "trace": trace}
        # between milestones the bar creeps on with time, never past 90% of the current stage
        stage(current, min(0.9, elapsed / estimate))
    code = proc.wait()
    log(f"[teach] extractor exit code {code}")
    trace.update({"exit_code": code, "seconds": round(time.monotonic() - started, 1)})
    extract_files = sorted((work / "extract").glob(f"{video.stem}*.json"), key=lambda p: p.stat().st_mtime)
    details = json.loads(extract_files[-1].read_text(encoding="utf-8")) if extract_files else {}
    reasons = (details.get("gate") or {}).get("reasons") or {}
    trace["report"] = report(details) if details else None
    if code == 0 and out.is_file():
        lesson = Lesson.from_dict(json.loads(out.read_text(encoding="utf-8")))
        if any(s.level >= MAX_HIGH for s in lesson.steps):
            return {"ok": False, "problem": PROBLEMS["too_high"], "reasons": {"too_high": f"max {MAX_HIGH} high"}, "trace": trace}
        if not lesson.steps:
            return {"ok": False, "problem": PROBLEMS["nothing"], "reasons": {"nothing": "no steps"}, "trace": trace}
        return {"ok": True, "lesson": lesson, "steps": [s.to_dict() for s in lesson.steps], "via": "video", "trace": trace}
    if code == 1:   # a lesson, but the gate refused it
        return {"ok": False, "problem": kid_problem(reasons), "reasons": reasons, "via": "video", "trace": trace}
    if code == 3:   # a step whose color no source could name
        return {"ok": False, "problem": PROBLEMS["unknown_color"], "reasons": {"unknown_color": "draft only"},
                "via": "video", "trace": trace}
    last = next((l for l in reversed(trace["lines"]) if l.startswith("[extract]") and "run time" not in l), "")
    return {"ok": None, "error": f"extractor exit code {code}: {last}", "trace": trace}


def active_colors() -> list[str]:
    """The colors in play (robot ACTIVE_COLORS, shared spelling): what perception is calibrated for."""
    try:
        from perception.robot_bridge import default_active_colors
        return default_active_colors()
    except Exception:
        return ["red", "purple", "yellow", "teal"]


def report(details: dict) -> dict:
    """The extractor's own record (lessons/extract/<id>.json), cut to what the debug page shows."""
    gate = details.get("gate") or {}
    return {
        "steps": [{k: r.get(k) for k in ("step", "cell", "level", "color", "color_source", "camera_color",
                                          "robot_memory_color", "camera_reads")} for r in details.get("steps") or []],
        "gate": {"passed": gate.get("passed"), "reasons": gate.get("reasons"), "notes": gate.get("notes"),
                 "checks": gate.get("checks")},
        "frames": details.get("frames"), "skipped_boards": details.get("skipped_boards"),
        "corrections": details.get("removed") or details.get("corrections"), "transients": details.get("transients"),
        "unusable": details.get("unusable"), "gemini": details.get("gemini"),
    }


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
