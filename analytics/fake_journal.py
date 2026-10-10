"""Write realistic sample journals so analytics can run before brain/ exists.

    python -m analytics.fake_journal                         # seed 7, 10 human + 6 robot sessions
    python -m analytics.fake_journal --seed 3 --humans 12 --robots 8

Writes data/sessions/sample_s-XXXX.jsonl (one file per session) and
lessons/sample_staircase.json. Output is deterministic for a given --seed.

Every session sets session_start.data.sample = true, and every file name starts
with "sample_", so the report excludes these unless --include-samples is set.
The generator only ever writes files whose names start with "sample_"; it never
touches a real journal. Rerunning overwrites earlier sample files of the same name.

What the samples contain:
  - human sessions with every failing verdict type, WAITING verdicts (hands in
    frame), repeated verdicts for the same stable board, and hints at levels 1 to 4
    (level 4 makes the robot hover over the correct cell)
  - robot sessions on two backends ("taught_poses", "learned_policy"), with
    retries that pass and one session that stops after MAX_ROBOT_RETRIES
  - one human session cut off mid-line, as if the laptop crashed
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from shared.config import CELLS, COLORS, COLUMNS, LESSONS_DIR, MAX_LEVEL, MAX_ROBOT_RETRIES, ROWS, SESSIONS_DIR
from shared.schemas import (
    Board,
    Gate,
    GateCheck,
    Lesson,
    Step,
    Verdict,
    apply_step,
    copy_board,
    empty_board,
)

SAMPLE_PREFIX = "sample_"
LESSON_ID = "sample-staircase"
LESSON_FILE = "sample_staircase.json"

# Same staircase as tests/scripts/staircase_clean.json.
STAIRCASE = [
    Step("red", "A1", 0, 4.2, "Start with red in the front-left corner."),
    Step("green", "B1", 0, 9.8, "Green goes next to it."),
    Step("blue", "C1", 0, 15.1, "Blue finishes the front row."),
    Step("blue", "A1", 1, 21.5, "Now stack blue on the red cube."),
    Step("red", "B1", 1, 27.0, "Red goes on top of green."),
    Step("yellow", "A1", 2, 33.4, "Yellow tops off the staircase."),
]

BASE_T = datetime(2026, 10, 10, 13, 0, tzinfo=timezone.utc).timestamp()
SESSION_GAP_S = 900.0

PASS = Verdict.PASS.value
WAITING = Verdict.WAITING.value
WRONG_CELL = Verdict.WRONG_CELL.value
WRONG_COLOR = Verdict.WRONG_COLOR.value
WRONG_LEVEL = Verdict.WRONG_LEVEL.value
EXTRA = Verdict.EXTRA.value
MISSING = Verdict.MISSING.value
FAIL_VERDICTS = [WRONG_CELL, WRONG_COLOR, WRONG_LEVEL, EXTRA, MISSING]

# Human behavior.
MAX_HINT_LEVEL = 4  # 1 text, 2 ghost cube, 3 show-me clip, 4 robot hovers
MAX_HUMAN_ATTEMPTS = 6  # the learner always gets it by then (the robot is pointing)
FIRST_TRY_P = {0: 0.75, 1: 0.6, 2: 0.45}  # by step level: higher stacks are harder
HUMAN_FAIL_WEIGHTS = {WRONG_CELL: 0.35, WRONG_COLOR: 0.25, WRONG_LEVEL: 0.15, EXTRA: 0.10, MISSING: 0.15}
# Chance the attempt after a hint passes, by hint level and the mistake it answers.
# Text fixes color and count mistakes; spatial mistakes need the ghost cube or more.
FIX_P = {
    1: {WRONG_CELL: 0.30, WRONG_COLOR: 0.80, WRONG_LEVEL: 0.35, EXTRA: 0.70, MISSING: 0.70},
    2: {WRONG_CELL: 0.80, WRONG_COLOR: 0.75, WRONG_LEVEL: 0.80, EXTRA: 0.70, MISSING: 0.75},
    3: {v: 0.70 for v in FAIL_VERDICTS},
    4: {v: 0.90 for v in FAIL_VERDICTS},
}
REPEAT_MISTAKE_P = 0.6

# Robot behavior.
BACKEND_PASS_P = {"taught_poses": 0.85, "learned_policy": 0.65}
BACKENDS = list(BACKEND_PASS_P)
ROBOT_FAIL_WEIGHTS = {WRONG_CELL: 0.6, MISSING: 0.3, WRONG_COLOR: 0.1}

# Journal noise.
HUMAN_WAITING_P = 0.35  # hands still over the grid
ROBOT_WAITING_P = 0.25  # arm not home yet
DUPLICATE_VERDICT_P = 0.2  # checker logs the same stable board twice


# ---------------------------------------------------------------- lesson


def build_lesson() -> Lesson:
    board = empty_board()
    for step in STAIRCASE:
        board = apply_step(board, step)
    return Lesson(
        lesson_id=LESSON_ID,
        title="Sample staircase",
        source_video=None,  # no real video behind the sample lesson
        steps=[Step(**s.to_dict()) for s in STAIRCASE],
        final_board=board,
        gate=Gate(passed=True, checks=[GateCheck("replay_matches_final_frame", True)]),
    )


# ---------------------------------------------------------------- boards


def _neighbors(cell: str) -> list[str]:
    col, row = COLUMNS.index(cell[0]), ROWS.index(cell[1])
    out = []
    for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c, r = col + dc, row + dr
        if 0 <= c < len(COLUMNS) and 0 <= r < len(ROWS):
            out.append(COLUMNS[c] + ROWS[r])
    return out


def _has_room(board: Board, cell: str) -> bool:
    return len(board[cell]) <= MAX_LEVEL


def mistake_board(rng: random.Random, before: Board, step: Step, verdict: str) -> Optional[tuple[Board, str]]:
    """Return (board the camera sees, fault_cell) for a failed placement of step,
    or None when that mistake is physically impossible on this board."""
    if verdict == WRONG_CELL:
        options = [c for c in _neighbors(step.cell) if _has_room(before, c)]
        options = options or [c for c in CELLS if c != step.cell and _has_room(before, c)]
        if not options:
            return None
        cell = rng.choice(options)
        board = copy_board(before)
        board[cell].append(step.color)
        return board, cell
    if verdict == WRONG_COLOR:
        board = copy_board(before)
        board[step.cell].append(rng.choice([c for c in COLORS if c != step.color]))
        return board, step.cell
    if verdict == WRONG_LEVEL:
        # Right cell and color, one level too high on top of a stray cube.
        if step.level + 1 > MAX_LEVEL:
            return None
        board = copy_board(before)
        board[step.cell] += [rng.choice(COLORS), step.color]
        return board, step.cell
    if verdict == EXTRA:
        board = apply_step(before, step)
        options = [c for c in CELLS if c != step.cell and _has_room(board, c)]
        if not options:
            return None
        cell = rng.choice(options)
        board[cell].append(rng.choice(COLORS))
        return board, cell
    if verdict == MISSING:
        # Either a cube from an earlier step got knocked off, or nothing was placed.
        board = apply_step(before, step)
        knockable = [c for c in CELLS if c != step.cell and board[c]]
        if knockable and rng.random() < 0.6:
            cell = rng.choice(knockable)
            board[cell].pop()
            return board, cell
        return copy_board(before), step.cell
    raise ValueError(f"not a failing verdict: {verdict}")


def _pick_mistake(rng: random.Random, weights: dict[str, float], before: Board, step: Step) -> tuple[str, Board, str]:
    choices = dict(weights)
    while choices:
        verdict = rng.choices(list(choices), weights=list(choices.values()))[0]
        result = mistake_board(rng, before, step, verdict)
        if result is not None:
            return verdict, result[0], result[1]
        del choices[verdict]
    raise RuntimeError(f"no possible mistake for step {step}")


def hint_text(level: int, step: Step, verdict: str, fault_cell: Optional[str]) -> str:
    if level >= 4:
        return "Follow the robot. It is pointing at the right cell."
    if level == 3:
        return "Watch how the expert did it."
    if level == 2:
        return f"Put the {step.color} cube where the glowing cube is."
    if verdict == WRONG_CELL:
        return f"Close! Move the {step.color} cube to {step.cell}."
    if verdict == WRONG_COLOR:
        return f"Almost. This spot needs a {step.color} cube."
    if verdict == WRONG_LEVEL:
        return f"Nice try. The {step.color} cube sits at level {step.level} on {step.cell}."
    if verdict == EXTRA:
        return f"One cube too many. Take the cube off {fault_cell}."
    if fault_cell != step.cell:
        return f"A cube moved off {fault_cell}. Put it back."
    return f"Place a {step.color} cube on {step.cell}."


# ---------------------------------------------------------------- sessions


class _Journal:
    def __init__(self, session: str, mode: str, t0: float):
        self.session, self.mode, self.t = session, mode, t0
        self.lines: list[str] = []

    def emit(self, dt: float, kind: str, data: dict) -> None:
        self.t += dt
        event = {"t": round(self.t, 3), "session": self.session, "mode": self.mode, "kind": kind, "data": data}
        self.lines.append(json.dumps(event))


def human_session(
    rng: random.Random, session: str, t0: float, steps: list[Step], learner_id: str, plan: dict[int, list[str]]
) -> list[str]:
    """plan maps a step index to verdicts forced on its first attempts."""
    j = _Journal(session, "human", t0)
    j.emit(0.0, "session_start", {"lesson_id": LESSON_ID, "learner_id": learner_id, "sample": True})
    board = empty_board()
    j.emit(1.0, "board", {"board": board})
    skill = rng.uniform(-0.1, 0.1)

    for k, step in enumerate(steps):
        forced = plan.get(k, [])
        hint_level = 0
        last_fail: Optional[str] = None
        for attempt in range(1, MAX_HUMAN_ATTEMPTS + 1):
            think = rng.uniform(4.0, 10.0) + 3.0 * hint_level
            if rng.random() < HUMAN_WAITING_P:
                j.emit(think * 0.7, "verdict", {"step": k, "verdict": WAITING, "fault_cell": None})
                think *= 0.3

            if attempt <= len(forced):
                verdict = forced[attempt - 1]
            elif attempt == MAX_HUMAN_ATTEMPTS:
                verdict = PASS
            else:
                p = FIRST_TRY_P[step.level] + skill if last_fail is None else FIX_P[hint_level][last_fail]
                verdict = PASS if rng.random() < p else None

            fault: Optional[str] = None
            if verdict == PASS:
                seen = apply_step(board, step)
            elif verdict is not None:
                seen, fault = mistake_board(rng, board, step, verdict)  # forced plans are always possible
            elif last_fail and rng.random() < REPEAT_MISTAKE_P and (r := mistake_board(rng, board, step, last_fail)):
                verdict, (seen, fault) = last_fail, r
            else:
                verdict, seen, fault = _pick_mistake(rng, HUMAN_FAIL_WEIGHTS, board, step)

            j.emit(think, "board", {"board": seen})
            result = {"step": k, "verdict": verdict, "fault_cell": fault}
            j.emit(0.6, "verdict", result)
            if rng.random() < DUPLICATE_VERDICT_P:
                j.emit(0.5, "verdict", dict(result))
            if verdict == PASS:
                board = seen
                break

            last_fail = verdict
            hint_level = min(hint_level + 1, MAX_HINT_LEVEL)
            j.emit(0.3, "hint", {"step": k, "level": hint_level, "text": hint_text(hint_level, step, verdict, fault)})
            if hint_level == MAX_HINT_LEVEL:
                j.emit(0.1, "robot_cmd", {"cmd": "hover", "cell": step.cell})
                j.emit(3.0, "robot_result", {"cmd": "hover", "ok": True})
                j.emit(2.0, "robot_cmd", {"cmd": "home"})
                j.emit(2.5, "robot_result", {"cmd": "home", "ok": True})

        if k < len(steps) - 1:
            j.emit(0.5, "step_advance", {"step": k + 1})
    j.emit(1.0, "session_end", {"completed": True, "steps_passed": len(steps), "reason": "complete"})
    return j.lines


def robot_session(
    rng: random.Random, session: str, t0: float, steps: list[Step], backend: str, plan: dict[int, list[str]]
) -> list[str]:
    """plan maps a step index to verdicts forced on its first tries."""
    j = _Journal(session, "robot", t0)
    j.emit(0.0, "session_start", {"lesson_id": LESSON_ID, "backend": backend, "sample": True})
    board = empty_board()
    j.emit(1.0, "board", {"board": board})
    episode = 0

    for k, step in enumerate(steps):
        forced = plan.get(k, [])
        for retry in range(MAX_ROBOT_RETRIES + 1):
            episode_id = None
            if backend == "learned_policy":
                episode += 1
                episode_id = f"{session}-ep{episode:03d}"
            cmd = {"cmd": "place", "color": step.color, "cell": step.cell, "level": step.level}
            j.emit(1.0, "robot_cmd", {**cmd, "backend": backend, "episode_id": episode_id})

            if retry < len(forced):
                verdict = forced[retry]
            else:
                verdict = PASS if rng.random() < BACKEND_PASS_P[backend] else None
            fault: Optional[str] = None
            place_result = {"cmd": "place", "ok": True}
            if verdict == PASS:
                seen = apply_step(board, step)
            elif verdict == MISSING:
                seen, fault = copy_board(board), step.cell
                place_result = {"cmd": "place", "ok": False, "error": "cube slipped from the gripper"}
            elif verdict is not None:
                seen, fault = mistake_board(rng, board, step, verdict)
            else:
                verdict, seen, fault = _pick_mistake(rng, ROBOT_FAIL_WEIGHTS, board, step)
                if verdict == MISSING:
                    seen, fault = copy_board(board), step.cell
                    place_result = {"cmd": "place", "ok": False, "error": "cube slipped from the gripper"}
            j.emit(rng.uniform(7.0, 11.0), "robot_result", place_result)

            if rng.random() < ROBOT_WAITING_P:
                j.emit(0.3, "verdict", {"step": k, "verdict": WAITING, "fault_cell": None, "retry": retry})
            j.emit(0.2, "robot_cmd", {"cmd": "home"})
            j.emit(rng.uniform(2.0, 3.0), "robot_result", {"cmd": "home", "ok": True})
            j.emit(0.6, "board", {"board": seen})
            result = {"step": k, "verdict": verdict, "fault_cell": fault, "retry": retry}
            j.emit(0.6, "verdict", result)
            if rng.random() < DUPLICATE_VERDICT_P:
                j.emit(0.5, "verdict", dict(result))
            if verdict == PASS:
                board = seen
                break
            if retry < MAX_ROBOT_RETRIES and verdict in (WRONG_CELL, WRONG_COLOR):
                # Take the misplaced cube back to the tray before trying again.
                j.emit(0.5, "robot_cmd", {"cmd": "pick", "cell": fault, "level": len(seen[fault]) - 1})
                j.emit(rng.uniform(6.0, 9.0), "robot_result", {"cmd": "pick", "ok": True})
        else:
            j.emit(0.5, "note", {"text": f"Robot stopped on step {k} after {MAX_ROBOT_RETRIES} retries. Operator check needed."})
            j.emit(1.0, "session_end", {"completed": False, "steps_passed": k, "reason": "robot_stopped"})
            return j.lines

        if k < len(steps) - 1:
            j.emit(0.5, "step_advance", {"step": k + 1})
    j.emit(1.0, "session_end", {"completed": True, "steps_passed": len(steps), "reason": "complete"})
    return j.lines


def _cut_off(rng: random.Random, lines: list[str]) -> str:
    """Keep the first part of a session and end on half a line, like a crash mid-write."""
    keep = rng.randint(max(2, len(lines) // 3), max(2, 2 * len(lines) // 3))
    partial = lines[keep][: len(lines[keep]) // 2]
    return "\n".join(lines[:keep]) + "\n" + partial


# ---------------------------------------------------------------- generate


def generate(seed: int, n_human: int = 10, n_robot: int = 6) -> tuple[dict, dict[str, str]]:
    """Pure: return (lesson dict, {file name: journal text}). No I/O."""
    rng = random.Random(seed)
    lesson = build_lesson()
    steps = lesson.steps
    modes = ["human"] * n_human + ["robot"] * n_robot
    rng.shuffle(modes)
    n_learners = max(1, n_human - 2)  # a few learners come back for a second session

    files: dict[str, str] = {}
    human_i = robot_i = 0
    for idx, mode in enumerate(modes, start=1):
        session = f"{SAMPLE_PREFIX}s-{idx:04d}"
        t0 = BASE_T + idx * SESSION_GAP_S + rng.uniform(0.0, 120.0)
        if mode == "human":
            # Rotate the forced first mistake so every failing verdict type shows up.
            first = FAIL_VERDICTS[human_i % len(FAIL_VERDICTS)]
            k = rng.choice([i for i, s in enumerate(steps) if first != WRONG_LEVEL or s.level < MAX_LEVEL])
            forced = [first] + {0: [WRONG_CELL] * 3, 1: [WRONG_CELL]}.get(human_i, [])  # reach hint level 4 at least once
            learner = f"{SAMPLE_PREFIX}learner-{human_i % n_learners + 1:02d}"
            lines = human_session(rng, session, t0, steps, learner, {k: forced})
            text = _cut_off(rng, lines) if n_human >= 2 and human_i == n_human - 1 else "\n".join(lines) + "\n"
            human_i += 1
        else:
            k = rng.randrange(len(steps))
            plan = {0: {k: [WRONG_CELL, PASS]}, 1: {k: [WRONG_CELL, MISSING, WRONG_CELL]}}.get(robot_i, {})
            lines = robot_session(rng, session, t0, steps, BACKENDS[robot_i % len(BACKENDS)], plan)
            text = "\n".join(lines) + "\n"
            robot_i += 1
        files[session + ".jsonl"] = text
    return lesson.to_dict(), files


def coverage(files: dict[str, str]) -> dict:
    """Summarize what the generated journals contain, so we can fail loudly on gaps."""
    verdicts: set[str] = set()
    hint_levels: set[int] = set()
    retry_passes = robot_stops = bad_lines = 0
    modes: dict[str, int] = {}
    for text in files.values():
        for line in text.split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            kind, data = event["kind"], event["data"]
            if kind == "session_start":
                modes[event["mode"]] = modes.get(event["mode"], 0) + 1
            elif kind == "verdict":
                verdicts.add(data["verdict"])
                if data["verdict"] == PASS and data.get("retry"):
                    retry_passes += 1
            elif kind == "hint":
                hint_levels.add(data["level"])
            elif kind == "session_end" and data.get("reason") == "robot_stopped":
                robot_stops += 1
    return {
        "sessions": modes,
        "verdicts": sorted(verdicts),
        "hint_levels": sorted(hint_levels),
        "robot_retry_passes": retry_passes,
        "robot_stops": robot_stops,
        "cut_off_lines": bad_lines,
    }


def missing_coverage(cov: dict) -> list[str]:
    gaps = [f"verdict {v}" for v in [v.value for v in Verdict] if v not in cov["verdicts"]]
    gaps += [f"hint level {lvl}" for lvl in range(1, MAX_HINT_LEVEL + 1) if lvl not in cov["hint_levels"]]
    if not cov["robot_retry_passes"]:
        gaps.append("a robot retry that passes")
    if not cov["robot_stops"]:
        gaps.append("a robot session that stops")
    if cov["cut_off_lines"] != 1:
        gaps.append("exactly one cut-off line")
    return gaps


def write_samples(seed: int, n_human: int, n_robot: int, sessions_dir: Path, lessons_dir: Path) -> list[Path]:
    lesson, files = generate(seed, n_human, n_robot)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    lessons_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in [(LESSON_FILE, json.dumps(lesson, indent=2) + "\n")] + [
        (n, files[n]) for n in sorted(files)
    ]:
        if not name.startswith(SAMPLE_PREFIX):
            raise ValueError(f"refusing to write a non-sample file: {name}")
        path = (lessons_dir if name == LESSON_FILE else sessions_dir) / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--humans", type=int, default=10, help="human sessions (at least 5)")
    ap.add_argument("--robots", type=int, default=6, help="robot sessions (at least 2)")
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--lessons-dir", type=Path, default=LESSONS_DIR)
    args = ap.parse_args()
    if args.humans < 5 or args.robots < 2:
        ap.error("need at least 5 human and 2 robot sessions to cover every verdict type and robot case")

    _, files = generate(args.seed, args.humans, args.robots)
    cov = coverage(files)
    gaps = missing_coverage(cov)
    if gaps:
        raise SystemExit(f"seed {args.seed} does not cover: {', '.join(gaps)}. Try another seed.")

    written = write_samples(args.seed, args.humans, args.robots, args.sessions_dir, args.lessons_dir)
    for path in written:
        print(f"wrote {path}")
    print(json.dumps(cov))
    stale = sorted(
        p.name for p in args.sessions_dir.glob(f"{SAMPLE_PREFIX}*.jsonl") if p not in written
    )
    if stale:
        print(f"note: older sample files left in place: {', '.join(stale)}")


if __name__ == "__main__":
    main()
