"""Write lessons/<lesson_id>.hints.json from real sessions: a "watch out" line and a starting
hint level per step, for the brain to read before a session.

    python -m analytics.hint_writeback --lesson staircase-v1 --sessions data/sessions

Only real sessions count (camera grid, see load.provenance). Synthetic, simulation, and
unverified sessions never produce a file. Thin data never changes behavior: a step needs
min_n samples before it gets an entry, and with no qualifying step no file is written.
Text comes from fixed templates; no model writes it. Directions reuse the brain's own
wording (brain/checker.py direction()) so the grid geometry lives in one place.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from analytics.load import DEFAULT_LESSON_DIRS, VALUES, Attempt, SessionRow, load_all
from analytics.metrics import hint_effectiveness, step_difficulty
from analytics.report import write_json_atomic
from shared.config import LESSONS_DIR, SESSIONS_DIR

try:  # the brain's pure wording helpers; without them, watch_out names cells instead
    from brain.checker import direction as brain_direction
    from brain.checker import level_phrase as brain_level_phrase
except Exception:  # noqa: BLE001 - any import problem means "fall back", never crash
    brain_direction: Optional[Callable[..., str]] = None
    brain_level_phrase: Optional[Callable[[int], str]] = None

MIN_N = 5               # samples a step needs before it gets an entry
TOP_SHARE = 0.5         # the top mistake must be at least this share of a step's failed attempts
FIRST_TRY_BELOW = 0.5   # start at level 2 only for steps whose first-try rate is under this
MIN_GAIN = 0.25         # ... and where level 2 fixes at least this much more often than level 1
MAX_START_LEVEL = 2

# One fixed template per verdict; WRONG_CELL falls back to cell names when no direction is safe.
TEMPLATES = {
    "WRONG_CELL": "Most people put this cube {direction} of its spot.",
    "WRONG_CELL_NAMED": "Most people put this in {fault}. It goes in {expected}.",
    "WRONG_COLOR": "Easy to mix up: this step needs a {color} cube.",
    "WRONG_LEVEL": "Most people stack this one at the wrong height. It goes {level}.",
    "EXTRA": "Most people leave an extra cube on {fault}. This step adds just one cube.",
    "MISSING": "Easy to bump the cube on {fault} here. Place this one gently.",
}


def _direction(expected: str, fault: str, sides: set) -> Optional[str]:
    """The brain's wording for where the cube went, or None when it is not safe to use."""
    if brain_direction is None or len(sides) != 1 or None in sides:
        return None
    try:
        words = brain_direction(expected, fault, next(iter(sides)))
    except (ValueError, KeyError, IndexError):
        return None
    return words or None


def watch_out(verdict: str, rows: list[Attempt], sides: set) -> Optional[str]:
    """Template text for the most common mistake; rows are the failed attempts behind it."""
    a = rows[0]
    expected = Counter(r.expected_cell for r in rows).most_common(1)[0][0]
    if verdict == "WRONG_CELL" and expected and a.fault_cell:
        d = _direction(expected, a.fault_cell, sides)
        if d is not None:
            return TEMPLATES["WRONG_CELL"].format(direction=d)
        return TEMPLATES["WRONG_CELL_NAMED"].format(fault=a.fault_cell, expected=expected)
    if verdict == "WRONG_COLOR" and a.step_color:
        return TEMPLATES["WRONG_COLOR"].format(color=a.step_color)
    if verdict == "WRONG_LEVEL" and a.step_level is not None:
        level = brain_level_phrase(a.step_level) if brain_level_phrase else f"on level {a.step_level + 1}"
        return TEMPLATES["WRONG_LEVEL"].format(level=level)
    if verdict in ("EXTRA", "MISSING") and a.fault_cell:
        return TEMPLATES[verdict].format(fault=a.fault_cell)
    return None


def build_hints(
    lesson_id: str,
    sessions: list[SessionRow],
    attempts: list[Attempt],
    min_n: int = MIN_N,
    first_try_below: float = FIRST_TRY_BELOW,
    min_gain: float = MIN_GAIN,
) -> Optional[dict[str, Any]]:
    """The hints file content, or None when no step qualifies. Uses real human rows only."""
    real = {s.session: s for s in sessions if s.provenance == "real" and s.lesson_id == lesson_id and s.mode == "human"}
    rows = [a for a in attempts if a.session in real and a.mode == "human"]
    if not rows:
        return None

    first_try = {d["step"]: d for d in step_difficulty(rows)}
    fixes: dict[int, dict[int, dict]] = defaultdict(dict)
    for e in hint_effectiveness(rows, by=("step", "hint_level")):
        fixes[e["step"]][e["hint_level"]] = e

    by_step: dict[int, list[Attempt]] = defaultdict(list)
    for a in rows:
        by_step[a.step].append(a)

    steps = []
    for step in sorted(by_step):
        tries = by_step[step]
        sides = {real[a.session].human_side for a in tries}
        failed = [a for a in tries if not a.passed]
        text = None
        if len(failed) >= min_n:
            groups: dict[tuple, list[Attempt]] = defaultdict(list)
            for a in failed:
                groups[(a.verdict, a.fault_cell)].append(a)
            (verdict, _), top = min(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0], str(kv[0][1])))
            if len(top) >= TOP_SHARE * len(failed):
                text = watch_out(verdict, top, sides)

        level = 1
        ft, l1, l2 = first_try.get(step), fixes[step].get(1), fixes[step].get(2)
        if (ft and ft["rate"] is not None and ft["rate"] < first_try_below
                and l1 and l1["n"] >= min_n and l2 and l2["n"] >= min_n
                and l2["rate"] - l1["rate"] >= min_gain):
            level = 2
        level = min(level, MAX_START_LEVEL)

        if text is not None or level != 1:
            steps.append({"step": step, "watch_out": text, "start_hint_level": level})

    if not steps:
        return None
    return {
        "lesson_id": lesson_id,
        "generated_from": {"sessions": len(real), "attempts": len(rows)},
        "steps": steps,
    }


def hints_path(lesson_id: str, lessons_dir: Path = LESSONS_DIR) -> Path:
    return Path(lessons_dir) / f"{lesson_id}.hints.json"


def write_hints(hints: dict[str, Any], lessons_dir: Path = LESSONS_DIR) -> tuple[Path, str]:
    """Write the file atomically. Returns (path, sha256 of the bytes written)."""
    if not hints.get("steps"):
        raise ValueError("refusing to write a hints file with no steps")
    path = hints_path(hints["lesson_id"], lessons_dir)
    data = write_json_atomic(path, hints)
    return path, hashlib.sha256(data).hexdigest()


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lesson", required=True, help="lesson_id")
    ap.add_argument("--sessions", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--lessons-dir", type=Path, default=LESSONS_DIR, help="where the hints file goes")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args(argv)

    t = load_all(args.sessions, DEFAULT_LESSON_DIRS)
    real = [s for s in t.sessions if s.provenance == "real" and s.lesson_id == args.lesson and s.mode == "human"]
    hints = build_hints(args.lesson, t.sessions, t.attempts, min_n=args.min_n)
    if hints is None:
        others = Counter(s.provenance for s in t.sessions if s.lesson_id == args.lesson and s.provenance != "real")
        print(f"no hints file written for {args.lesson}: {len(real)} real human sessions, no step with enough data"
              + (f" (not used: {dict(others)})" if others else ""))
        return
    path, digest = write_hints(hints, args.lessons_dir)
    print(f"wrote {path} ({len(hints['steps'])} steps from {len(real)} real human sessions)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
