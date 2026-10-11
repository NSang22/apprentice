"""Write synthetic session journals by driving the real brain with simulated students.

    python -m analytics.fake_journal                                              # demo lesson, seed 7
    python -m analytics.fake_journal --lesson tests/fixtures/staircase-v1.json --seed 3

The journals are written by brain/session.py and brain/journal.py themselves, so every
event kind and data field is exactly what the brain writes. This module only plays the
student: a simulated human who places cubes (sometimes wrong), and the brain's FakeRobot
(which sometimes lands on the wrong cell or fails a place). A fake clock replaces wall time,
so output is deterministic for a given --lesson and --seed.

Output: data/sessions/synthetic_s-XXXX.jsonl. Each session_start carries data.synthetic = true
on top of the brain's own fields; the report leaves these out unless asked. The generator
only writes files whose names start with "synthetic_" and replaces earlier ones of the same name.

What the journals contain:
  - human sessions with every failing verdict the checker gives (WRONG_CELL, WRONG_COLOR,
    WRONG_LEVEL, EXTRA, MISSING for a knocked cube) and hint levels 1 to 4 (level 4 makes
    the fake robot hover over the right cell)
  - robot sessions with retries that pass, a cube a judge moved, a failed place, and one
    session that hits the retry cap and stops
  - one human session stopped early (outcome server_stopped) and one cut off mid-line
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from brain.checker import expected_board
from brain.fake_robot import FakeRobot
from brain.journal import Journal
from brain.session import RUNNING, Session
from brain.settings import DEMO_LESSON
from shared.config import CELLS, COLORS, COLUMNS, MAX_LEVEL, ROWS, SESSIONS_DIR
from shared.schemas import Board, Lesson, Step, Verdict, apply_step, copy_board

SYNTHETIC_PREFIX = "synthetic_"
BASE_T = datetime(2026, 10, 10, 13, 0, tzinfo=timezone.utc).timestamp()
SESSION_GAP_S = 900.0

PASS = Verdict.PASS.value
WRONG_CELL = Verdict.WRONG_CELL.value
WRONG_COLOR = Verdict.WRONG_COLOR.value
WRONG_LEVEL = Verdict.WRONG_LEVEL.value
EXTRA = Verdict.EXTRA.value
MISSING = Verdict.MISSING.value  # here: an older cube knocked off, so fault_cell is set
MISTAKES = [WRONG_CELL, WRONG_COLOR, WRONG_LEVEL, EXTRA, MISSING]

# Simulated human.
MAX_HUMAN_ATTEMPTS = 6  # by then the robot is pointing at the cell, so they get it
FIRST_TRY_P = {0: 0.75, 1: 0.6, 2: 0.45}  # by step level: higher stacks are harder
HUMAN_MISTAKE_WEIGHTS = {WRONG_CELL: 0.35, WRONG_COLOR: 0.25, WRONG_LEVEL: 0.15, EXTRA: 0.10, MISSING: 0.15}
# Chance the attempt after a hint passes, by the hint level now showing and the last mistake.
# Text fixes color and count mistakes; spatial mistakes need the ghost cube or more.
FIX_P = {
    1: {WRONG_CELL: 0.30, WRONG_COLOR: 0.80, WRONG_LEVEL: 0.35, EXTRA: 0.70, MISSING: 0.70},
    2: {WRONG_CELL: 0.80, WRONG_COLOR: 0.75, WRONG_LEVEL: 0.80, EXTRA: 0.70, MISSING: 0.75},
    3: {m: 0.70 for m in MISTAKES},
    4: {m: 0.90 for m in MISTAKES},
}
REPEAT_MISTAKE_P = 0.5
THINK_S = (4.0, 10.0)  # seconds between settles, plus 3 s per hint level

# Fake robot.
ROBOT_WRONG_CELL_P = 0.15  # chance any one place lands on the nearest other cell
ROBOT_FAIL_P = 0.05        # chance any one place returns ok=false
ROBOT_KNOCK_P = 0.3        # chance a judge moves one cube during a robot session
ROBOT_MOTION_S = 3.0
SETTLE_S = 0.6             # stabilizer time after a move
MAX_ROBOT_BOARDS = 200     # guard against a loop that never ends


class FakeClock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = round(self.t + seconds, 3)


# What brain/server.py passes as the session's provenance hook for a simulated run. The git fields
# keep the Session's "unknown" defaults: synthetic data does not come from a checked-out run.
SIM_PROVENANCE = {"grid_source": "sim", "robot": "fake"}


def _start(session: Session, journal: Journal) -> None:
    """Start like brain/server.py does: it notes the grid source when the first frame arrives
    (server.py, "grid_source" note), and with --autostart only then starts the session."""
    journal.write("note", {"grid_source": SIM_PROVENANCE["grid_source"]})
    session.start()


class SyntheticJournal(Journal):
    """The brain's Journal, plus data.synthetic = true on session_start."""

    def write(self, kind: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if kind == "session_start":
            data = {**(data or {}), "synthetic": True}
        return super().write(kind, data)


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


def mistake_board(rng: random.Random, before: Board, step: Step, mistake: str) -> Optional[Board]:
    """The board a human leaves when making this mistake on step, or None if it can't happen here.
    The brain's checker grades the board; the label only says what the human meant to do wrong."""
    if mistake == WRONG_CELL:
        options = [c for c in _neighbors(step.cell) if _has_room(before, c)]
        options = options or [c for c in CELLS if c != step.cell and _has_room(before, c)]
        if not options:
            return None
        board = copy_board(before)
        board[rng.choice(options)].append(step.color)
        return board
    if mistake == WRONG_COLOR:
        board = copy_board(before)
        board[step.cell].append(rng.choice([c for c in COLORS if c != step.color]))
        return board
    if mistake == WRONG_LEVEL:
        # Right cell and color, one level too high on top of a stray cube of another color.
        if step.level + 1 > MAX_LEVEL:
            return None
        board = copy_board(before)
        board[step.cell] += [rng.choice([c for c in COLORS if c != step.color]), step.color]
        return board
    if mistake == EXTRA:
        board = apply_step(before, step)
        options = [c for c in CELLS if c != step.cell and _has_room(board, c)]
        if not options:
            return None
        board[rng.choice(options)].append(rng.choice(COLORS))
        return board
    if mistake == MISSING:
        # Placed the cube but knocked an older one off the grid.
        board = apply_step(before, step)
        knockable = [c for c in CELLS if c != step.cell and board[c]]
        if not knockable:
            return None
        board[rng.choice(knockable)].pop()
        return board
    raise ValueError(f"not a mistake: {mistake}")


def _possible(mistake: str, before: Board, step: Step) -> bool:
    return mistake_board(random.Random(0), before, step, mistake) is not None


def _pick_mistake(rng: random.Random, before: Board, step: Step, last: Optional[str]) -> tuple[str, Board]:
    if last is not None and rng.random() < REPEAT_MISTAKE_P:
        board = mistake_board(rng, before, step, last)
        if board is not None:
            return last, board
    choices = dict(HUMAN_MISTAKE_WEIGHTS)
    while choices:
        mistake = rng.choices(list(choices), weights=list(choices.values()))[0]
        board = mistake_board(rng, before, step, mistake)
        if board is not None:
            return mistake, board
        del choices[mistake]
    raise RuntimeError(f"no possible mistake for {step}")


def _settle(session: Session, board: Board, clock: FakeClock) -> None:
    """One stable board into the session. Like brain/server.py:174-176, a robot move resets the
    stabilizer, so the same board settles once more after any move (a hover, in human mode)."""
    moves = session.robot_moves
    session.on_board(copy_board(board))
    if session.robot_moves != moves and session.phase == RUNNING:
        clock.advance(SETTLE_S)
        session.on_board(copy_board(board))


# ---------------------------------------------------------------- sessions


def human_session(
    rng: random.Random,
    journal: Journal,
    lesson: Lesson,
    clock: FakeClock,
    forced: dict[int, list[str]],
    stop_at_step: Optional[int] = None,
) -> None:
    """forced maps a step index to mistakes the human makes on its first attempts."""
    robot = FakeRobot(motion_s=ROBOT_MOTION_S, sleep=clock.advance)  # only hovers in human mode
    session = Session(lesson, "human", journal, robot=robot, sleep=clock.advance, provenance=SIM_PROVENANCE.copy)
    _start(session, journal)
    skill = rng.uniform(-0.1, 0.1)

    for k, step in enumerate(lesson.steps):
        if k == stop_at_step:
            session.end("server_stopped")
            return
        before = expected_board(lesson, k - 1)
        plan = forced.get(k, [])
        last: Optional[str] = None
        for attempt in range(1, MAX_HUMAN_ATTEMPTS + 1):
            if session.step_index != k or session.phase != RUNNING:
                break
            clock.advance(rng.uniform(*THINK_S) + 3.0 * session.hint_level)
            if attempt <= len(plan):
                last = plan[attempt - 1]
                board = mistake_board(rng, before, step, last)
            elif attempt == MAX_HUMAN_ATTEMPTS:
                board = apply_step(before, step)
            else:
                p = FIRST_TRY_P.get(step.level, 0.5) + skill if last is None else FIX_P[session.hint_level][last]
                if rng.random() < p:
                    board = apply_step(before, step)
                else:
                    last, board = _pick_mistake(rng, before, step, last)
            _settle(session, board, clock)


def robot_session(
    rng: random.Random,
    journal: Journal,
    lesson: Lesson,
    clock: FakeClock,
    wrong_cell_on: set[int],
    fail_on: set[int],
    knock_at_step: Optional[int] = None,
) -> None:
    """wrong_cell_on / fail_on: 1-based place() counts the fake robot gets wrong."""
    robot = FakeRobot(wrong_cell_on=wrong_cell_on, fail_on=fail_on, motion_s=ROBOT_MOTION_S, sleep=clock.advance)
    session = Session(lesson, "robot", journal, robot=robot, sleep=clock.advance, provenance=SIM_PROVENANCE.copy)
    _start(session, journal)
    knocked = False
    for _ in range(MAX_ROBOT_BOARDS):
        if session.phase != RUNNING:
            return
        if not knocked and knock_at_step is not None and session.step_index >= knock_at_step:
            knocked = _judge_moves_a_cube(rng, robot.board)
        moves = session.robot_moves
        clock.advance(SETTLE_S)
        session.on_board(robot.snapshot()[0])
        if session.robot_moves == moves and session.phase == RUNNING:
            break  # nothing would change the board again; a real server would wait forever
    session.end("server_stopped")


def _judge_moves_a_cube(rng: random.Random, board: Board) -> bool:
    """A judge slides a lone cube to an empty neighbor cell (README demo step 3)."""
    lone = [c for c in CELLS if len(board[c]) == 1 and any(not board[n] for n in _neighbors(c))]
    if not lone:
        return False
    cell = rng.choice(lone)
    to = rng.choice([n for n in _neighbors(cell) if not board[n]])
    board[to].append(board[cell].pop())
    return True


# ---------------------------------------------------------------- generate


def _cut_off(rng: random.Random, text: str) -> str:
    """Keep the first part of a journal and end on half a line, like a crash mid-write."""
    lines = text.rstrip("\n").split("\n")
    keep = rng.randint(max(2, len(lines) // 3), max(2, 2 * len(lines) // 3))
    return "\n".join(lines[:keep]) + "\n" + lines[keep][: len(lines[keep]) // 2]


def _random_places(rng: random.Random, n_places: int, p: float) -> set[int]:
    return {n for n in range(1, n_places + 1) if rng.random() < p}


def generate(lesson: Lesson, seed: int = 7, n_human: int = 10, n_robot: int = 6) -> dict[str, str]:
    """Return {file name: journal text}. Journals go through a temporary directory only."""
    rng = random.Random(seed)
    steps = lesson.steps
    n_places = 3 * len(steps) + 3
    modes = ["human"] * n_human + ["robot"] * n_robot
    rng.shuffle(modes)

    files: dict[str, str] = {}
    human_i = robot_i = 0
    with tempfile.TemporaryDirectory() as tmp:
        for idx, mode in enumerate(modes, start=1):
            session_id = f"{SYNTHETIC_PREFIX}s-{idx:04d}"
            clock = FakeClock(round(BASE_T + idx * SESSION_GAP_S + rng.uniform(0.0, 120.0), 3))
            journal = SyntheticJournal(session_id, mode, Path(tmp), clock=clock)
            if mode == "human":
                # Rotate the first forced mistake so every verdict type shows up.
                first = MISTAKES[human_i % len(MISTAKES)]
                ks = [k for k in range(len(steps)) if _possible(first, expected_board(lesson, k - 1), steps[k])]
                forced = [first] + {0: [WRONG_CELL] * 3, 1: [WRONG_CELL]}.get(human_i, [])  # hint level 4 at least once
                stop = rng.randrange(1, len(steps)) if human_i == n_human - 2 and len(steps) > 1 else None
                human_session(rng, journal, lesson, clock, {rng.choice(ks): forced} if ks else {}, stop)
                human_i += 1
            else:
                wrong, fail = {
                    0: ({2}, set()),         # a retry that passes
                    1: ({2, 3, 4}, set()),   # hits the retry cap and stops
                    2: (set(), {3}),         # a place that returns ok=false
                }.get(robot_i, (_random_places(rng, n_places, ROBOT_WRONG_CELL_P), _random_places(rng, n_places, ROBOT_FAIL_P)))
                knock = rng.randrange(1, len(steps)) if len(steps) > 1 and rng.random() < ROBOT_KNOCK_P else None
                robot_session(rng, journal, lesson, clock, wrong, fail, knock)
                robot_i += 1
            journal.close()
            text = journal.path.read_text(encoding="utf-8")
            if mode == "human" and human_i == n_human and n_human >= 2:
                text = _cut_off(rng, text)
            files[journal.path.name] = text
    return files


def coverage(files: dict[str, str]) -> dict:
    """What the journals contain, so the CLI can refuse a seed that misses a case."""
    verdicts: set[str] = set()
    hint_levels: set[int] = set()
    outcomes: dict[str, int] = {}
    modes: dict[str, int] = {}
    retry_passes = bad_lines = 0
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
                still_placing = data["verdict"] == MISSING and data["fault_cell"] is None
                verdicts.add("MISSING (still placing)" if still_placing else data["verdict"])
                if data["verdict"] == PASS and data.get("retry"):
                    retry_passes += 1
            elif kind == "hint":
                hint_levels.add(data["hint_level"])
            elif kind == "session_end":
                outcomes[data["outcome"]] = outcomes.get(data["outcome"], 0) + 1
    return {
        "sessions": modes,
        "verdicts": sorted(verdicts),
        "hint_levels": sorted(hint_levels),
        "outcomes": outcomes,
        "robot_retry_passes": retry_passes,
        "cut_off_lines": bad_lines,
    }


def missing_coverage(cov: dict) -> list[str]:
    gaps = [f"verdict {v}" for v in [PASS] + MISTAKES if v not in cov["verdicts"]]
    gaps += [f"hint level {lvl}" for lvl in range(1, 5) if lvl not in cov["hint_levels"]]
    gaps += [f"outcome {o}" for o in ("complete", "robot_stopped", "server_stopped") if o not in cov["outcomes"]]
    if not cov["robot_retry_passes"]:
        gaps.append("a robot retry that passes")
    if cov["cut_off_lines"] != 1:
        gaps.append("exactly one cut-off line")
    return gaps


def write_journals(files: dict[str, str], sessions_dir: Path) -> list[Path]:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(files):
        if not name.startswith(SYNTHETIC_PREFIX):
            raise ValueError(f"refusing to write a non-synthetic file: {name}")
        path = sessions_dir / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(files[name])
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lesson", type=Path, default=DEMO_LESSON, help=f"lesson JSON (default: {DEMO_LESSON.name})")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--humans", type=int, default=10, help="human sessions (at least 5)")
    ap.add_argument("--robots", type=int, default=6, help="robot sessions (at least 3)")
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    args = ap.parse_args()
    if args.humans < 5 or args.robots < 3:
        ap.error("need at least 5 human and 3 robot sessions to cover every verdict and robot case")

    lesson = Lesson.from_dict(json.loads(args.lesson.read_text(encoding="utf-8")))
    files = generate(lesson, args.seed, args.humans, args.robots)
    cov = coverage(files)
    gaps = missing_coverage(cov)
    if gaps:
        raise SystemExit(f"lesson {lesson.lesson_id}, seed {args.seed} does not cover: {', '.join(gaps)}. Try another seed.")

    written = write_journals(files, args.sessions_dir)
    print(f"wrote {len(written)} journals for lesson {lesson.lesson_id} to {args.sessions_dir}")
    print(json.dumps(cov))
    stale = sorted(p.name for p in args.sessions_dir.glob(f"{SYNTHETIC_PREFIX}*.jsonl") if p not in written)
    if stale:
        print(f"note: older synthetic files left in place: {', '.join(stale)}")


if __name__ == "__main__":
    main()
