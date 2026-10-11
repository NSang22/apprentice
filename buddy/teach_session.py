"""The Teach the Robot game loop (rules and the extractor in buddy/teach.py).

Phases:
  home        the intro and the child's saved lessons
  clear       (table) waiting for an empty board, before recording or before a rebuild
  recording   the child builds while Pip watches: at most RECORD_S seconds, or "I'm done!"
  processing  Pip turns the recording into steps; the phone shows a fun stage name and a bar
  problem     the gate (or the rules) refused it: a kind line and "Teach it again"
  review      the steps Pip learned; the child names the lesson and saves it
  saved       saved: "Build it, Pip!" now, or later from the saved lessons
  bank        rebuild: Pip asks for the next colors in its bank (it holds BANK_SIZE), then Ready
  bank_check  (table) the vision model looks in the bank: before placing, and when robot.api says a
              color is missing (its sheet calibration can misread blocks on the green box)
  building    Pip places one step (the arm on the table; it appears on the phone in practice)
  build_check (table) did the block land on its square and level?
  build_fix   it did not: the child puts it right, then taps Fixed it
  build_done  rebuilt

Practice mode: the phone is the board. Taps are the recording (Undo takes the last one back),
and a rebuild fills the phone's board step by step.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

from buddy import recall, teach
from buddy.scanner import Scanner
from buddy.session import SESSIONS_DIR, Profile, default_colors, empty_board
from buddy.table import Pip, RobotEyes, TableBoard, missing_from_bank

MIN_STAGE_S = 0.9          # each fun stage name stays up at least this long, so a child can read it
PRACTICE_PLACE_S = 1.5     # practice rebuild: one block appears this often


class TeachSession:
    def __init__(self, table: TableBoard, eyes: RobotEyes, pip: Pip, profile: Profile, clock=time.time):
        self.table, self.eyes, self.pip, self.profile = table, eyes, pip, profile
        self.clock = clock
        self.lock = threading.RLock()
        self.colors = default_colors()
        self.scanner = Scanner(table, eyes, clock)
        self._reset()

    def _reset(self) -> None:
        self.phase = "home"
        self.session_id: Optional[str] = None
        self.after_clear = ""                # "record" or "build"
        self.record_started = 0.0
        self.recorder: Optional[teach.VideoRecorder] = None
        self.taps: list[dict] = []           # practice recording
        self.board = empty_board()           # practice board (recording and rebuild)
        self.stage = {"name": "", "label": "", "progress": 0.0}
        self.lesson = None                   # the lesson being reviewed or rebuilt
        self.steps: list[dict] = []
        self.video_path = None
        self.problem = ""
        self.build_i = 0
        self.bank: list[str] = []
        self.glow: Optional[str] = None
        self.until = 0.0
        self.arm_used = False
        self.scan_stuck = False
        self.audio = None                    # the microphone recorder while recording
        self.audio_path = None
        self.heard: dict = {}                # the narration's transcript and where speech and grid disagree
        self.bank_mode = ""                  # "ready" (check before placing) or "retry" (after a miss)
        self.bank_result: Optional[dict] = None
        self.place_retried = False
        self.place_fails = 0                 # failed places on the current rebuild step
        self.mood, self.say = "happy", ""

    @property
    def source(self) -> str:
        return self.profile.data["source"]

    # ------------------------------------------------------------ teaching

    def start_teaching(self) -> None:
        with self.lock:
            if self.source == "table" and not self.table.ready():
                raise ValueError("I can't see the table yet. Turn on the table (⚙), or switch to Practice.")
            self._reset()
            self.session_id = time.strftime("teach-%Y%m%d-%H%M%S")
            self._log("session_start", {"game": "teach", "source": self.source, "profile": self.profile.data.get("name", "")})
            if self.source == "table":
                self.eyes.warm()
                self._clear_then("record")
            else:
                self._record()

    def _clear_then(self, what: str) -> None:
        stable = self.table.scanned(0.0)
        if stable is not None and recall.block_count(stable) == 0:
            self._record() if what == "record" else self._bank_or_place()
            return
        self.phase, self.after_clear = "clear", what
        self.mood, self.say = "happy", "Clear the board first, please!"

    def _record(self) -> None:
        self.phase, self.record_started = "recording", self.clock()
        self.taps, self.board = [], empty_board()
        self.mood = "look"
        self.say = ("I'm watching! Build something, one block at a time. Towers can be 2 high."
                    if self.source == "table" else "I'm watching! Pick a color and tap squares to build. Towers can be 2 high.")
        if self.source == "table":
            self.table.record_start()
            self.recorder = teach.VideoRecorder(teach.VIDEOS_DIR / f"{self.session_id}_silent{teach.VIDEO_EXT}")
            self.recorder.start()
            self.audio = teach.AudioRecorder(teach.VIDEOS_DIR / f"{self.session_id}.ogg")
            self.audio.start()

    def tap(self, cell: str, color: str) -> None:
        """Practice recording: add a block on top of the cell (at most MAX_HIGH high)."""
        with self.lock:
            if self.phase != "recording" or self.source != "practice" or cell not in self.board or color not in self.colors:
                return
            if len(self.board[cell]) >= teach.MAX_HIGH:
                self.say = f"That tower is {teach.MAX_HIGH} high already, the tallest in this game!"
                return
            self.taps.append({"t": self.clock() - self.record_started, "cell": cell, "color": color,
                              "level": len(self.board[cell])})
            self.board[cell].append(color)

    def undo(self) -> None:
        with self.lock:
            if self.phase == "recording" and self.taps:
                t = self.taps.pop()
                self.board[t["cell"]].pop()

    def done_teaching(self) -> None:
        with self.lock:
            if self.phase == "recording":
                self._process()

    def _process(self) -> None:
        self.phase, self.mood, self.say = "processing", "think", "Hmm, let me think about what you taught me..."
        frames = self.table.record_stop() if self.source == "table" else []
        recorder, self.recorder = self.recorder, None
        audio, self.audio = self.audio, None
        taps = list(self.taps)
        threading.Thread(target=self._learn, args=(frames, recorder, taps, audio), daemon=True).start()

    def _set_stage(self, name: str, frac: float) -> None:
        """Called from the learning thread. A new stage waits until the last one was on screen
        MIN_STAGE_S, so the funny words are readable even when the work itself is quick."""
        names = [n for n, _ in teach.STAGES]
        i = names.index(name)
        if self.stage["name"] != name:
            wait = MIN_STAGE_S - (time.monotonic() - getattr(self, "_stage_t", 0.0))
            if self.stage["name"] and wait > 0:
                time.sleep(wait)
            self._stage_t = time.monotonic()
        with self.lock:
            self.stage = {"name": name, "label": dict(teach.STAGES)[name], "progress": round((i + frac) / len(names), 3)}

    def _trace(self, step: str, title: str, status: str, summary: str, seconds: Optional[float] = None,
               details=None) -> None:
        """One step of turning the recording into a lesson, for the debug page (/buddy/debug).
        status: ok, warn (worked, with a caveat), fail, or skip."""
        self._log("trace", {"step": step, "title": title, "status": status, "summary": summary,
                            "seconds": None if seconds is None else round(seconds, 2), "details": details})

    def _learn(self, frames: list[dict], recorder, taps: list[dict], audio=None) -> None:
        heard: dict = {}
        video = sound = None
        t_all = time.monotonic()
        try:
            self._set_stage("rewind", 0.0)
            if self.source == "practice":
                self._trace("record", "Recording (practice)", "ok" if taps else "warn",
                            f"{len(taps)} taps on the phone", details={"taps": taps})
            else:
                secs = frames and (frames[-1].get("t", 0) - frames[0].get("t", 0)) or 0
                self._trace("record", "Recording", "ok" if frames else "fail",
                            f"{secs:.0f} s, {len(frames)} camera boards from perception", seconds=secs)
            t = time.monotonic()
            silent = recorder.stop() if recorder else None
            if recorder:
                self._trace("video", "Video", "ok" if silent else "fail",
                            f"{recorder.frames} frames at {teach.VIDEO_FPS:g} fps, {teach.VIDEO_WIDTH} px wide"
                            if silent else f"no video: {recorder.error or 'no frames from the frame server'}",
                            time.monotonic() - t, {"file": str(silent) if silent else None, "error": recorder.error,
                                                    "bytes": silent.stat().st_size if silent else 0})
            t = time.monotonic()
            sound = audio.stop() if audio else None
            if audio:
                self._trace("audio", "Microphone", "ok" if sound else "warn",
                            f"{sound.stat().st_size // 1024} KB of sound" if sound else f"no sound: {audio.error or 'silent'}",
                            time.monotonic() - t, {"file": str(sound) if sound else None, "device": audio.device,
                                                    "error": audio.error})
            video = silent
            if silent and sound:  # one WebM with the narration, for playback and the record
                t = time.monotonic()
                joined = silent.with_name(silent.name.replace("_silent", ""))
                ok = teach.mux(silent, sound, joined)
                video = joined if ok else silent
                self._trace("join", "Video + sound", "ok" if ok else "warn",
                            "joined into one WebM" if ok else "could not join; the video has no sound",
                            time.monotonic() - t, {"file": str(video)})
            listener, t_listen = None, time.monotonic()
            if self.source == "table" and sound:
                # OpenAI transcribes the narration while the extractor works on the video
                def listen():
                    heard.update(self.eyes.narration(sound))
                listener = threading.Thread(target=listen, daemon=True)
                listener.start()
                self._set_stage("listen", 0.5)
            source_video = str(video.relative_to(teach.REPO_ROOT)).replace("\\", "/") if video else None
            title = "My lesson"
            result = {"ok": None}
            if self.source == "table" and video:
                # the team's extractor on the video itself (fast 5 Hz sampling, robot colour memory,
                # the vision model for any color nobody could name), run in the main checkout
                video_s = recorder.frames / teach.VIDEO_FPS if recorder else 0.0
                t = time.monotonic()
                result = teach.learn_video(video, self.eyes.repo, title, video_s, stage=self._set_stage,
                                           log=lambda line: None)
                tr = result.get("trace") or {}
                code = tr.get("exit_code")
                status = "ok" if result.get("ok") else "warn" if result.get("ok") is False else "fail"
                summary = (f"{len(result['steps'])} steps" if result.get("ok") else
                           f"refused: {', '.join(result.get('reasons') or {}) or result.get('problem')}" if result.get("ok") is False
                           else result.get("error", "no answer"))
                self._trace("extractor", "Extractor on the video", status, f"exit code {code}: {summary}",
                            time.monotonic() - t, {"command": tr.get("command"), "cwd": tr.get("cwd"),
                                                    "output": tr.get("lines"), "exit_code": code})
                rep = tr.get("report")
                if rep:
                    gate = rep.get("gate") or {}
                    self._trace("gate", "Gate checks", "ok" if gate.get("passed") else "fail",
                                "all checks passed" if gate.get("passed") else
                                "; ".join(f"{k}: {v}" for k, v in (gate.get("reasons") or {}).items()),
                                details=rep)
            if self.source == "table" and result.get("ok") is None:
                # no answer from the video: learn from perception's live frames, with the robot's
                # own eyes naming each cell's top block
                t = time.monotonic()
                tops = None
                res = self.eyes.scan(all_colors=True)
                if res.get("ok"):
                    tops = {b["cell"]: b["color"] for b in res["blocks"] if b.get("cell")}
                self._trace("eyes", "Robot's eyes on the final board", "ok" if res.get("ok") else "warn",
                            ", ".join(f"{c} {col}" for c, col in (tops or {}).items()) or res.get("error", "nothing on the grid"),
                            time.monotonic() - t, res)
                t = time.monotonic()
                result = teach.learn(frames, self.session_id, title, source_video, tops, stage=self._set_stage)
                self._trace("fallback", "Extractor on the live camera boards", "ok" if result["ok"] else "fail",
                            f"{len(result['steps'])} steps" if result["ok"] else
                            "; ".join(f"{k}: {v}" for k, v in (result.get("reasons") or {}).items()),
                            time.monotonic() - t, {"frames": len(frames), "tops_from_eyes": tops,
                                                    "reasons": result.get("reasons"), "steps": result.get("steps")})
            elif self.source == "practice":
                for name, _ in teach.STAGES[:4]:
                    self._set_stage(name, 1.0)
                result = teach.learn_taps(taps, self.session_id, title)
            if listener is not None:
                listener.join(timeout=60)
                ok = heard.get("ok")
                self._trace("narration", "What you said (speech to text)", "ok" if ok else "fail",
                            (f"\"{heard.get('transcript', '')[:120]}\"" if ok else heard.get("error", "no answer")),
                            heard.get("seconds") or (time.monotonic() - t_listen),
                            {k: heard.get(k) for k in ("transcript", "placements", "model", "error")})
            seen_steps = (result["lesson"].steps if result.get("ok") else
                          ((result.get("trace") or {}).get("report") or {}).get("steps") or [])
            if heard.get("ok") and seen_steps:   # speech vs the grid, in order; the grid decides
                check = teach.check_narration(seen_steps, heard.get("placements", []))
                if result.get("ok"):
                    result["steps"] = [s.to_dict() for s in result["lesson"].steps]
                elif check["unseen"]:            # a refused recording: say what the narration explains
                    colors_seen = [st.get("color") for st in seen_steps]
                    again = next((c for c in check["unseen"] if c in colors_seen), None)
                    result["problem"] = (f"I heard you use {again} twice. Each color can be used once, so I don't mix them up!"
                                         if again else
                                         f"I heard you put a {check['unseen'][0]} block, but I didn't see it. It may have been "
                                         "covered too quickly. Let's teach it again, and take your hands away after each block!")
                heard.update(check)
                self._trace("speech_vs_grid", "Speech vs what the camera saw", "ok" if not check["notes"] else "warn",
                            f"{check['matched']} placements matched" + (f", {len(check['notes'])} disagree" if check["notes"] else ""),
                            details=check)
            for name, _ in teach.STAGES:  # walk the rest of the bar, so it always reaches the end
                if [n for n, _ in teach.STAGES].index(name) >= [n for n, _ in teach.STAGES].index(self.stage["name"] or "rewind"):
                    self._set_stage(name, 1.0)
        except Exception as e:  # never strand the child on a loading screen
            import traceback
            result = {"ok": False, "problem": teach.DEFAULT_PROBLEM, "reasons": {"error": f"{type(e).__name__}: {e}"}}
            self._trace("crash", "Crash", "fail", f"{type(e).__name__}: {e}", details=traceback.format_exc())
        with self.lock:
            self.video_path = video
            self.audio_path = sound
            self.heard = {k: heard.get(k) for k in ("ok", "transcript", "notes", "matched", "error")} if heard else {}
            self._trace("result", "Result", "ok" if result["ok"] else "fail",
                        f"learned {len(result['steps'])} steps" if result["ok"] else f"Pip said: {result['problem']}",
                        time.monotonic() - t_all, {"via": result.get("via", "frames" if self.source == "table" else "taps"),
                                                   "steps": result.get("steps"), "reasons": result.get("reasons")})
            if result["ok"]:
                self.lesson, self.steps = result["lesson"], result["steps"]
                self.phase, self.mood = "review", "cheer"
                n = len(self.steps)
                self.say = f"I learned {n} step{'s' if n != 1 else ''}! What should we call it?"
            else:
                self.phase, self.mood, self.problem = "problem", "encourage", result["problem"]
                self.say = result["problem"]

    def save(self, name: str) -> None:
        with self.lock:
            if self.phase != "review" or self.lesson is None:
                return
            name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -'")[:24] or "My lesson"
            self.lesson.lesson_id, self.lesson.title = teach.new_id(name), name
            if self.video_path and self.video_path.exists():  # the video takes the lesson's name too
                target = self.video_path.with_name(f"{self.lesson.lesson_id}{teach.VIDEO_EXT}")
                self.video_path.rename(target)
                self.video_path = target
                self.lesson.source_video = str(target.relative_to(teach.REPO_ROOT)).replace("\\", "/")
            if self.audio_path and self.audio_path.exists():
                target = self.audio_path.with_name(f"{self.lesson.lesson_id}.ogg")
                self.audio_path.rename(target)
                self.audio_path = target
            teach.save(self.lesson, self.profile.data.get("name", ""),
                       {"source": self.source, "narration": self.heard or None,
                        "audio": str(self.audio_path.relative_to(teach.REPO_ROOT)).replace("\\", "/") if self.audio_path else None})
            self.profile.data["xp"] += 10 + 2 * len(self.steps)
            self.profile.data["taught"] = self.profile.data.get("taught", 0) + 1
            self.profile.played_today()
            self.profile.save()
            self._log("saved", {"lesson_id": self.lesson.lesson_id, "title": name, "steps": len(self.steps)})
            self.phase, self.mood = "saved", "cheer"
            self.say = f"Saved \"{name}\"! Want me to build it now?"

    # ------------------------------------------------------------ rebuilding

    def build(self, lesson_id: Optional[str] = None) -> None:
        with self.lock:
            if lesson_id:
                lesson = teach.load(lesson_id)
                self._reset()
                self.lesson, self.steps = lesson, [s.to_dict() for s in lesson.steps]
                self.session_id = time.strftime("teach-%Y%m%d-%H%M%S")
            if self.lesson is None:
                return
            if self.source == "table" and not self.table.ready():
                raise ValueError("I can't see the table yet. Turn on the table (⚙), or switch to Practice.")
            self._log("build_start", {"lesson_id": self.lesson.lesson_id, "source": self.source})
            self.build_i, self.board = 0, empty_board()
            if self.source == "table":
                self._clear_then("build")
            else:
                self._bank_or_place()

    def _bank_or_place(self) -> None:
        """Every BANK_SIZE steps (and at the start) the table rebuild stops to refill the bank."""
        if self.build_i >= len(self.lesson.steps):
            self._build_done()
            return
        if self.source == "table" and self.build_i % teach.BANK_SIZE == 0:
            self._ask_bank(teach.bank_colors(self.lesson.steps, self.build_i))
            return
        self._place_next()

    def _ask_bank(self, colors: list[str], empty: bool = False) -> None:
        self.phase, self.bank, self.glow, self.mood = "bank", colors, None, "happy"
        what = " and ".join(f"a {c}" for c in colors)
        self.say = (f"My bank is out of {colors[0]}! Can you put {what} block in it, then tap Ready?" if empty else
                    f"Can you put {what} block{'s' if len(colors) > 1 else ''} in my bank, next to the grid? Then tap Ready!")

    def ready(self) -> None:
        with self.lock:
            if self.phase == "bank":
                if self.source == "table" and self.eyes.info.get("model"):
                    self._check_bank("ready")
                else:
                    self._place_next()
            elif self.phase == "build_fix":
                self.place_fails, self.place_retried = 0, False
                self.build_i += 1
                self._bank_or_place()
            elif self.phase in ("build_check",) and self.scan_stuck:
                self.scan_stuck = False
                self.scanner.start()

    def _check_bank(self, mode: str) -> None:
        """The vision model lists the colors in the bank (worker thread); tick() acts on the answer."""
        self.phase, self.bank_mode, self.bank_result, self.mood = "bank_check", mode, None, "look"
        self.say = "Let me look in my bank..."

        def job():
            result = self.eyes.bank()
            with self.lock:
                self.bank_result = result
        threading.Thread(target=job, daemon=True).start()

    def _bank_checked(self) -> None:
        res, s = self.bank_result, self.lesson.steps[self.build_i]
        self._log("bank_check", {"mode": self.bank_mode, "need": self.bank or [s.color], **(res or {})})
        if not res or not res.get("ok"):          # no second opinion: go on as before
            self._place_next() if self.bank_mode == "ready" else self._ask_bank([s.color], empty=True)
            return
        seen = res["colors"]
        if self.bank_mode == "ready":
            missing = missing_from_bank(self.bank, seen)
            if missing:
                self._ask_bank(missing)
                what = ", ".join(seen) if seen else "nothing"
                self.say = (f"I see {what} in my bank. I still need "
                            + " and ".join(f"a {c}" for c in missing) + " block. Then tap Ready!")
                return
            self._place_next()
            return
        if s.color in seen and not self.place_retried:   # it is there: robot.api misread it, try again
            self.place_retried = True
            self._place_next()
            self.say = f"I can see the {s.color} block in my bank! Let me try again."
        elif s.color in seen:
            self._ask_bank([s.color])
            self.say = (f"I can see the {s.color} block, but I can't pick it out. Can you move it to the "
                        "middle of the green box, then tap Ready?")
        else:
            self._ask_bank([s.color], empty=True)
            self.say = f"I don't see a {s.color} block in my bank. Can you put one in the green box, then tap Ready?"

    def _place_next(self) -> None:
        s = self.lesson.steps[self.build_i]
        self.phase, self.glow, self.mood = "building", s.cell, "happy"
        where = "on top" if s.level else "here"
        self.say = f"Step {self.build_i + 1} of {len(self.lesson.steps)}: a {s.color} block {where}!"
        self.arm_used = self.source == "table" and self.pip.reachable()
        if self.arm_used:
            self.pip.place(s.cell, s.color, should_stop=lambda: self.phase != "building", level=s.level)
        elif self.source == "table":  # the arm is off: the child places it for Pip
            self.phase, self.mood = "build_fix", "encourage"
            self.say = f"My arm is resting. Can you put a {s.color} block on the glowing square? Then tap Done."
        else:
            self.until = self.clock() + PRACTICE_PLACE_S

    def _build_done(self) -> None:
        self.phase, self.glow, self.mood = "build_done", None, "cheer"
        self.say = f"Ta-da! I built \"{self.lesson.title}\". Did I get it right?"
        self._log("build_done", {"lesson_id": self.lesson.lesson_id})

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            if self.phase == "clear":
                stable = self.table.scanned(0.0)
                if stable is not None and recall.block_count(stable) == 0 and self.table.hands_away():
                    self._record() if self.after_clear == "record" else self._bank_or_place()
            elif self.phase == "recording" and now - self.record_started >= teach.RECORD_S:
                self._process()
            elif self.phase == "building":
                if not self.arm_used:
                    if now >= self.until:  # practice: the block appears
                        s = self.lesson.steps[self.build_i]
                        self.board[s.cell].append(s.color)
                        self.build_i += 1
                        self._bank_or_place()
                elif not self.pip.busy:
                    s = self.lesson.steps[self.build_i]
                    if self.pip.error == "hands":
                        self._ask_bank([s.color])
                        self.say = "Hands away from the board, please, so I can move! Then tap Ready."
                    elif self.pip.error:
                        self._log("place_error", {"step": self.build_i + 1, "error": self.pip.error})
                        self.place_fails += 1
                        if self.place_fails >= 2:   # don't loop: the child places this one for Pip
                            self.phase, self.mood, self.glow = "build_fix", "encourage", s.cell
                            self.say = (f"I can't pick up the {s.color} block from my bank. Can you put it "
                                        f"{'on top ' if s.level else ''}on the glowing square for me? Then tap Done.")
                            self.arm_used = False
                        elif self.eyes.info.get("model"):
                            self._check_bank("retry")
                        else:
                            self._ask_bank([s.color], empty=True)
                    else:
                        self.phase, self.scan_stuck = "build_check", False
                        self.scanner.start()
                        self.mood, self.say = "look", "Let me check..."
            elif self.phase == "bank_check" and self.bank_result is not None:
                self._bank_checked()
            elif self.phase == "build_check":
                st = self.scanner.poll()
                if st["status"] == "done":
                    s = self.lesson.steps[self.build_i]
                    stack = st["board"].get(s.cell) or []
                    # the robot's eyes see only the top block, so a landed block is the top color there
                    landed = bool(stack) and stack[-1] == s.color
                    self._log("placed", {"step": self.build_i + 1, "cell": s.cell, "level": s.level, "color": s.color,
                                         "landed": landed, "sync": st.get("sync")})
                    if landed:
                        self.place_retried, self.place_fails = False, 0
                        self.build_i += 1
                        self._bank_or_place()
                    else:
                        self.phase, self.mood = "build_fix", "think"
                        self.say = f"Oops, I missed! Can you put my {s.color} block on the glowing square? Then tap Fixed it."
                elif st["status"] == "hands":
                    self.say = "Hands up! I need to see the board."
                elif st["status"] == "nudge":
                    self.say = f"I can't tell the block on {st['cell']}. Nudge it to the middle!"
                elif st["status"] == "stuck" and not self.scan_stuck:
                    self.scan_stuck = True
                    self.say = "Hmm, I can't see the board well. Tap Scan again!"

    def quit(self) -> None:
        with self.lock:
            if self.phase == "recording" and self.source == "table":
                self.table.record_stop()
            if self.recorder:
                self.recorder.stop()
            if self.audio:
                self.audio.stop()
            self.scanner.stop()
            if self.session_id and self.phase not in ("home",):
                self._log("session_end", {"phase": self.phase})
            self._reset()

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
            else:
                board = self.table.latest or empty_board()
            p = self.profile.data
            total = len(self.lesson.steps) if self.lesson else 0
            return {
                "phase": self.phase,
                "source": self.source,
                "colors": self.colors,
                "profile": {k: p.get(k) for k in ("name", "avatar", "xp", "stars", "taught")},
                "table": self.table.status(),
                "record_left": round(max(0.0, teach.RECORD_S - (self.clock() - self.record_started)), 1)
                               if self.phase == "recording" else 0,
                "record_s": teach.RECORD_S,
                "max_high": teach.MAX_HIGH,
                "taps": len(self.taps),
                "stage": self.stage,
                "stages": [label for _, label in teach.STAGES],
                "steps": self.steps,
                "title": self.lesson.title if self.lesson else "",
                "build": {"i": self.build_i, "total": total},
                "bank": self.bank,
                "board": board,
                "glow": self.glow,
                "problem": self.problem,
                "scan_stuck": self.scan_stuck,
                "pip": {"busy": self.pip.busy, "step": self.pip.step, "arm": self.arm_used},
                "lessons": teach.saved_lessons() if self.phase == "home" else [],
                "heard": self.heard,
                # what Pip watched: the unsaved recording while reviewing, the lesson's once saved
                "video": teach.video_url(self.video_path.stem) if self.video_path else None,
                "mood": self.mood,
                "say": self.say,
            }
