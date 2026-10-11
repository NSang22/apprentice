"""Teach the Robot: the child builds anything, Pip learns the steps, and can rebuild it later.

The child taps Start, builds on the grid (at most RECORD_S seconds, or until "I'm done!"), and the
recording goes through the team's extractor (brain/extractor.py, from video-to-lesson): settled
boards, history-based stacks, corrections (a block taken back off deletes its step), and the gate.
The result is an ordinary Lesson, saved under data/buddy/lessons/ with the child's name for it.
Later, "Build it, Pip!" rebuilds it step by step with the arm (buddy/teach_session.py).

How Pip learns on the table: the extractor runs on the recorded video itself (learn_video:
python -m brain.extractor --video in the main checkout, sampling 5 frames a second, with the robot's
colour memory and the vision-model fallback). If the video gives no answer, the live frames are used
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
EXTRACT_EXT = ".avi"      # MJPG at a fixed rate: what the extractor reads (VideoRecorder)
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
    """The camera's frames (perception's frame server, which owns the webcam) at a fixed VIDEO_FPS,
    as MJPG in an AVI for the extractor.

    One thread fetches frames as fast as the frame server gives them; another writes the newest one
    every 1/VIDEO_FPS seconds, repeating it when the next is late, so the video's clock is the real
    clock. (Encoding WebM while recording managed about 2 frames a second at 1280 px on the Jetson:
    a 43 s build came out as a 12 s video and the extractor saw too few clear frames.) MJPG costs one
    JPEG per frame. The WebM with sound for playback is made afterwards (make_playback)."""

    def __init__(self, path: Path, url: str = FRAME_URL, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH):
        self.path, self.url, self.fps, self.width = path, url, fps, width
        self.frames = 0                   # written (including repeats)
        self.fetched = 0                  # distinct frames from the camera
        self.started = 0.0
        self.error: Optional[str] = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _grab(self):
        import numpy as np
        with urllib.request.urlopen(self.url, timeout=2) as r:
            shape = tuple(int(v) for v in r.headers["X-Shape"].split(","))
            return np.frombuffer(r.read(), np.uint8).reshape(shape)

    def _fetch(self) -> None:
        import cv2
        while not self._stop.is_set():
            try:
                frame = self._grab()
            except Exception as e:
                self.error = str(e)
                self._stop.wait(0.2)
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest = small
                self.fetched += 1

    def _write(self) -> None:
        import cv2
        writer = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    frame = self._latest
                if frame is None:
                    self._stop.wait(0.05)
                    continue
                if writer is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps,
                                             (frame.shape[1], frame.shape[0]))
                    self.started = time.monotonic()
                due = int((time.monotonic() - self.started) * self.fps) + 1
                while self.frames < due:          # catch up with the clock (repeats the newest frame)
                    writer.write(frame)
                    self.frames += 1
                self._stop.wait(max(0.0, self.started + self.frames / self.fps - time.monotonic()))
        except Exception as e:
            self.error = str(e)
        finally:
            if writer is not None:
                writer.release()

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._fetch, daemon=True), threading.Thread(target=self._write, daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self) -> Optional[Path]:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        return self.path if self.frames else None


def make_playback(video: Path, audio: Optional[Path], out: Path, width: int = 640) -> bool:
    """The recording as WebM (VP8, plus the narration as Opus) that phones play, with GStreamer.
    Small and fast settings: it runs while Pip thinks, next to the extractor."""
    import subprocess
    cmd = ["gst-launch-1.0", "-q", "-e", "webmmux", "name=m", "!", "filesink", f"location={out}",
           "filesrc", f"location={video}", "!", "avidemux", "!", "jpegdec", "!", "videoconvert", "!", "videoscale", "!",
           f"video/x-raw,width={width},height={width * 9 // 16}", "!", "vp8enc", "deadline=1", "cpu-used=8",
           "target-bitrate=1500000", "!", "queue", "!", "m."]
    if audio:
        cmd += ["filesrc", f"location={audio}", "!", "oggdemux", "!", "opusparse", "!", "queue", "!", "m."]
    try:
        subprocess.run(cmd, timeout=180, capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.is_file() and out.stat().st_size > 10000


MIC = "hw:CARD=Camera,DEV=0"   # the overhead webcam's own microphone (arecord -l: card "Camera")


class AudioRecorder:
    """The webcam microphone to an Opus file while the child builds (GStreamer: alsasrc, opusenc).
    The narration is a second source for the steps: OpenAI transcribes it afterwards."""

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


# Colors people say for each block, and blocks that are easy to mix up in speech ("light blue" for teal).
SAID_LIKE = {"skyblue": "teal", "teal": "skyblue", "orange": "yellow", "yellow": "orange"}


def _step_fields(s) -> tuple[str, str, int]:
    """(color, cell, level) of a Step or of a step dict from the extractor's report."""
    if isinstance(s, dict):
        return s.get("color"), s.get("cell"), int(s.get("level") or 0)
    return s.color, s.cell, s.level


def _align(said: list[str], seen: list[str]) -> list[tuple[int, int]]:
    """Longest common subsequence of two color lists: (spoken index, step index) pairs, in order.
    A near-miss color ("light blue" for teal) counts too, but an exact color wins a tie."""
    def score(a, b):
        # a step whose color nobody could read lines up with any spoken color (speech then names it)
        return 2 if a == b else 1 if SAID_LIKE.get(a) == b or b == UNKNOWN else 0
    n, m = len(said), len(seen)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            s = score(said[i], seen[j])
            best[i][j] = max(best[i + 1][j], best[i][j + 1], best[i + 1][j + 1] + s if s else 0)
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        s = score(said[i], seen[j])
        if s and best[i][j] == best[i + 1][j + 1] + s:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def check_narration(steps: list, placements: list[dict]) -> dict:
    """Natural speech vs the steps the camera saw, in order. Nobody has to say "B2": the spoken
    colors are lined up with the steps' colors (longest common subsequence), and a square or "on
    the red" is checked only when it was said. The grid decides (AGENTS.md rule 2): this fills each
    matched step's narration and lists what speech and grid disagree on:
      unseen   said, but no step has it (a block covered too fast, or never placed)
      unheard  a step nobody mentioned
    """
    spoken = [p for p in placements if isinstance(p, dict) and p.get("action", "place") == "place"
              and p.get("color") in COLORS]
    fields = [_step_fields(s) for s in steps]
    pairs = _align([p["color"] for p in spoken], [f[0] for f in fields])
    notes, matched = [], 0
    hit_spoken, hit_steps = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        p, (color, cell, level) = spoken[i], fields[j]
        text = str(p.get("text") or "").strip()
        matched += 1
        if p.get("cell") and p["cell"] != cell:
            notes.append(f"You said the {color} block goes on {p['cell']}, but I saw it on {cell}.")
        below = next((c for c in COLORS if f"on the {c}" in text.lower() or f"on top of the {c}" in text.lower()), None)
        if below and level == 0:
            notes.append(f"You said the {color} block goes on the {below}, but I saw it on the board.")
        if p["color"] != color:
            notes.append(f"You called the {color} block \"{p['color']}\". I went with what I saw: {color}.")
        if text and not isinstance(steps[j], dict) and not steps[j].narration:
            steps[j].narration = text
    unseen = [spoken[i] for i in range(len(spoken)) if i not in hit_spoken]
    unheard = [fields[j] for j in range(len(fields)) if j not in hit_steps]
    for p in unseen:
        again = sum(1 for q in spoken if q["color"] == p["color"]) > 1
        notes.append(f"I heard you put {'another' if again else 'a'} {p['color']} block, but I didn't see it"
                     f"{' (each color can be used once)' if again else ''}.")
    if spoken:
        for color, cell, _ in unheard:
            notes.append(f"I saw {color} on {cell}, but I didn't hear it.")
    return {"matched": matched, "notes": notes, "unseen": [p["color"] for p in unseen],
            "unheard": [f[0] for f in unheard], "spoken": len(spoken), "steps": len(fields)}


def speech_colors(steps: list, placements: list[dict]) -> dict[int, str]:
    """{step number (1-based): spoken color} for the steps whose color nobody could read, lined up
    with what was said, in order (narration is the ground truth people give naturally). A spoken
    color already used by another step is not given twice."""
    spoken = [p for p in placements if isinstance(p, dict) and p.get("action", "place") == "place"
              and p.get("color") in COLORS]
    fields = [_step_fields(s) for s in steps]
    used = {f[0] for f in fields if f[0] in COLORS}
    out = {}
    for i, j in _align([p["color"] for p in spoken], [f[0] for f in fields]):
        color = spoken[i]["color"]
        if fields[j][0] == UNKNOWN and color not in used:
            out[j + 1] = color
            used.add(color)
    return out


def reconcile(steps: list[dict], truth: dict[str, list[str]], placements: list[dict]) -> tuple[list[dict], list[str]]:
    """The extractor's steps checked against the final board the vision model read (cell -> colors,
    bottom to top). The extractor is good at order and timing, and unsure of colors and stacks;
    the final board is the truth for what is there. Returns (steps, what changed):
      a step on a block the board has, in another color   takes the board's color (no false duplicates)
      a step the final board has no block for              is dropped (a ghost, or a stack it lost)
      a block on the board no step placed                  is added, in the order it was said
    Steps whose cell the vision model could not see (no blob there) are kept as they are."""
    changes, out = [], []
    for st in steps:
        cell, level = st["cell"], int(st["level"])
        if cell not in truth:
            out.append(dict(st))
            continue
        stack = truth[cell]
        if level >= len(stack):
            changes.append(f"dropped {st['color']} on {cell} level {level}: not on the final board")
            continue
        if st["color"] != stack[level]:
            changes.append(f"{cell} level {level}: {st['color']} -> {stack[level]}")
            st = {**st, "color": stack[level], "color_source": "vision model"}
        out.append(dict(st))
    have = {(st["cell"], int(st["level"])) for st in out}
    spoken = [p.get("color") for p in placements if isinstance(p, dict) and p.get("action", "place") == "place"]
    for cell, stack in truth.items():
        for level, color in enumerate(stack):
            if (cell, level) in have:
                continue
            new = {"cell": cell, "level": level, "color": color, "color_source": "vision model"}
            # place it after the step whose color was said just before it
            at = len(out)
            if color in spoken:
                before = spoken[:spoken.index(color)]
                idx = [i for i, st in enumerate(out) if st["color"] in before]
                at = (max(idx) + 1) if idx else 0
            below = [i for i, st in enumerate(out) if st["cell"] == cell and int(st["level"]) < level]
            if below:
                at = max(at, max(below) + 1)          # never before the block it stands on
            out.insert(at, new)
            have.add((cell, level))
            changes.append(f"added {color} on {cell} level {level}: on the final board, no step placed it")
    return out, changes


def stacks_ok(steps: list[dict]) -> bool:
    """Replaying the steps stacks properly: each block's level is the number already under it."""
    height: dict[str, int] = {}
    for st in steps:
        if int(st["level"]) != height.get(st["cell"], 0):
            return False
        height[st["cell"]] = height.get(st["cell"], 0) + 1
    return True


def merge_ghost(steps: list[dict], reasons: dict, placements: list[dict]) -> Optional[list[dict]]:
    """A 2-high tower leans in the camera's picture (the camera is not straight above), so for one
    block put on a stack, "on this square" and "on the neighbour" can fit the image almost equally:
    the extractor then records two blocks at once, one with the color (the neighbour, a ghost) and
    one it could not read (the square where the change really happened). Seen in a real run: orange
    on the teal at B2 came out as "orange on A1" plus "unknown on B2".

    When the gate says two blocks were added at once, exactly one of them has a color, and the
    narration counts one block fewer than the extractor's steps, they are one block: it keeps the
    unread step's square and level and takes the ghost's color. Returns the repaired steps, or None.
    """
    import re
    m = re.search(r"added (\w\d) L(\d), (\w\d) L(\d)", str(reasons.get("one_cube_per_change", "")))
    spoken = [p for p in placements if isinstance(p, dict) and p.get("action", "place") == "place" and p.get("color") in COLORS]
    if not m or len(spoken) != len(steps) - 1:
        return None
    slots = {(m.group(1), int(m.group(2))), (m.group(3), int(m.group(4)))}
    pair = [st for st in steps if (st["cell"], int(st["level"])) in slots]
    known = [st for st in pair if st["color"] in COLORS]
    unread = [st for st in pair if st["color"] not in COLORS]
    if len(pair) != 2 or len(known) != 1 or len(unread) != 1:
        return None
    ghost, real = known[0], unread[0]
    out = []
    for st in steps:
        if st is ghost:
            continue
        out.append({**st, "color": ghost["color"], "color_source": "merged"} if st is real else dict(st))
    # the replay must still stack properly: each block's level is the number already under it
    height: dict[str, int] = {}
    for st in out:
        if int(st["level"]) != height.get(st["cell"], 0):
            return None
        height[st["cell"]] = height.get(st["cell"], 0) + 1
    return out


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
          stage: Callable[[str, float], None] = lambda name, frac: None,
          spoken: Optional[list[dict]] = None) -> dict:
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
    if spoken and any(r.color == UNKNOWN for r in rows):   # what was said names what nobody could read
        manual.update(speech_colors([{"color": r.color, "cell": r.cell, "level": r.level} for r in rows], spoken))
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
EXTRACT_MILESTONES = [("sampling", "flummox"), ("hand band", "count"), ("] model:", "squint"), ("run time", "check")]


def learn_video(video: Path, repo: Path, title: str, video_s: float,
                stage: Callable[[str, float], None] = lambda name, frac: None, log: Callable[[str], None] = print,
                colors_file: Optional[list[dict]] = None) -> dict:
    """The recording's video -> lesson with the team's extractor on the video itself
    (python -m brain.extractor --video, 5 frames a second by default), run in `repo` (the main
    checkout) so it uses the calibration, robot colour memory and OpenAI key everything else uses.

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
    out = work / f"{video.stem}{'-said' if colors_file else ''}.json"
    # the colors in play and the block size, as scripts/jetson_up.sh runs perception: the extractor's
    # default is every shared color, which the 4-color calibration refuses (exit code 2)
    cmd = [sys.executable, "-u", "-m", "brain.extractor", "--video", str(video.resolve()), "--out", str(out),
           "--title", title, "--colors", ",".join(calibrated_colors(repo)), "--block-mm", os.environ.get("BLOCK_MM", "47")]
    if colors_file:   # step colors from the narration, for steps no source could read
        cf = work / f"{video.stem}-said-colors.json"
        cf.write_text(json.dumps(colors_file), encoding="utf-8")
        cmd += ["--colors-file", str(cf)]
    extractor_src = (repo / "brain" / "extractor.py").read_text(encoding="utf-8", errors="ignore")
    if "--no-gemini" in extractor_src:  # an older main checkout: never call Gemini (the project uses OpenAI)
        cmd.append("--no-gemini")
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
    extract_files = sorted((work / "extract").glob(f"{out.stem}*.json"), key=lambda p: p.stat().st_mtime)
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


def calibrated_colors(repo: Path) -> list[str]:
    """The colors perception's calibration has (repo/perception/calib/colors.json). The extractor's
    camera reads only these; every other color still comes from the robot's color memory, which
    knows all six (it named orange and sky blue right in real recordings)."""
    try:
        data = json.loads((repo / "perception" / "calib" / "colors.json").read_text(encoding="utf-8"))
        have = [c for c in COLORS if c in data.get("colors", {})]
        return have or active_colors()
    except (OSError, ValueError):
        return active_colors()


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
        "unusable": details.get("unusable"), "model_colors": details.get("model"),
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
