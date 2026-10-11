"""Checker: grades a stable board against a lesson step (README section 6.2).

check() compares the WHOLE board to the expected board after steps 0..step_index, so it
also catches knocked or moved cubes, not just the newest one. Pure functions only.

Boards are compared slot by slot (cell, level). A slot that differs contributes the
expected cube to `missing` and the actual cube to `extra`. Verdicts are chosen in this
order, so the newest step is explained first and older damage after it:

  0. any cell reads "unknown"                       -> WAITING, fault_cell = that cell (never a pass)
  1. board matches expected                          -> PASS
  2. board matches expected minus the newest cube    -> MISSING (still placing)
  3. newest cube missing from its slot:
     a. same color elsewhere in its cell             -> WRONG_LEVEL
     b. same color in another cell (nearest wins)    -> WRONG_CELL, fault_cell = where it is
     c. another color in its exact slot              -> WRONG_COLOR
  4. an older cube's slot holds another color        -> WRONG_COLOR
  5. an older cube is missing and its color sits
     somewhere it shouldn't (moved or knocked)       -> WRONG_CELL, fault_cell = where it is
  6. any other cube where none belongs: if a slot open right now in the rest of the lesson
     needs its color                                      -> WRONG_CELL, expected_cell = that slot
     otherwise                                       -> EXTRA, fault_cell = where it is, and no
                                                        expected_cell (take it off, no arrow)
  7. cubes missing with nothing extra (knocked off)  -> MISSING, fault_cell = where it belongs

A block already sitting in its own slot from a later step (and on the right blocks) is not a
fault: it is set aside before grading, and the lesson counts it when it gets there.

progress_index() is the session's step counter: the first step whose slot does not hold its
color, so removing an earlier block moves the lesson back to that step.

Besides the verdict, a CheckResult says how to fix the board, one cube at a time:
  fault_cell, fault_level        the wrong cube to take off (fault_level is None when there
                                 is nothing to remove, e.g. a cube knocked off the grid)
  expected_cell, expected_level  the slot that needs a cube (None when nothing is missing)
The robot uses both for pick and re-place; the UI can use expected_cell for the ghost cube.

Hint directions ("left", "toward you") are phrased for the human's side of the grid,
HUMAN_SIDE in brain/settings.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

from brain.settings import HUMAN_SIDE, HUMAN_SIDES
from shared.config import CELLS, COLUMNS, ROWS, UNKNOWN
from shared.schemas import Board, Lesson, Step, Verdict, apply_step, boards_equal, empty_board


@dataclass(frozen=True)
class CheckResult:
    verdict: Verdict
    fault_cell: Optional[str] = None
    detail: str = ""
    hint_text: Optional[str] = None
    fault_level: Optional[int] = None
    expected_cell: Optional[str] = None
    expected_level: Optional[int] = None


class Cube(NamedTuple):
    cell: str
    level: int
    color: str


# ---------------------------------------------------------------- board helpers


def expected_board(lesson: Lesson, step_index: int) -> Board:
    """Board after steps 0..step_index have been placed. step_index = -1 gives an empty board."""
    if not -1 <= step_index < len(lesson.steps):
        raise IndexError(f"step_index {step_index} out of range for {len(lesson.steps)} steps")
    board = empty_board()
    for s in lesson.steps[: step_index + 1]:
        board = apply_step(board, s)
    return board


def slot_holds(board: Board, s: Step) -> bool:
    """The step's slot holds its color ("unknown" never matches)."""
    stack = board[s.cell]
    return len(stack) > s.level and stack[s.level] == s.color


def progress_index(board: Board, lesson: Lesson) -> int:
    """The first step whose slot does not hold its color; len(steps) when every slot does."""
    for i, s in enumerate(lesson.steps):
        if not slot_holds(board, s):
            return i
    return len(lesson.steps)


def unknown_cells(board: Board) -> list[str]:
    return [c for c in CELLS if UNKNOWN in board[c]]


def set_aside_early(board: Board, lesson: Lesson, step_index: int) -> Board:
    """The board without blocks that already sit in a later step's slot, on top of their
    stack, with every block under them as the lesson wants (placed early, not a mistake)."""
    final = lesson.final_board
    out = {c: list(v) for c, v in board.items()}
    for s in lesson.steps[step_index + 1:]:
        stack = out[s.cell]
        if len(stack) == s.level + 1 and stack == final[s.cell][: s.level + 1]:
            stack.pop()
    return out


def slot_diff(expected: Board, actual: Board) -> tuple[list[Cube], list[Cube]]:
    """(missing, extra): cubes in expected but not actual, and vice versa, slot by slot."""
    missing, extra = [], []
    for cell in CELLS:
        e, a = expected[cell], actual[cell]
        for level in range(max(len(e), len(a))):
            want = e[level] if level < len(e) else None
            got = a[level] if level < len(a) else None
            if want != got:
                if want is not None:
                    missing.append(Cube(cell, level, want))
                if got is not None:
                    extra.append(Cube(cell, level, got))
    return missing, extra


def cell_xy(cell: str) -> tuple[int, int]:
    """(column index left to right, row index front to back)."""
    return COLUMNS.index(cell[0]), ROWS.index(cell[1])


def distance(a: str, b: str) -> int:
    (ax, ay), (bx, by) = cell_xy(a), cell_xy(b)
    return abs(ax - bx) + abs(ay - by)


# ---------------------------------------------------------------- hint text

_COUNT = {1: "one cell", 2: "two cells"}
_LEVEL = {0: "directly on the board", 1: "on the second level", 2: "on the third level"}


def direction(src: str, dst: str, human_side: str = HUMAN_SIDE) -> str:
    """How to move a cube from src to dst, as seen by the human: "one cell left",
    "one cell right and two cells away from you".

    On the printed sheet, columns A to C run from tag 0 toward tag 1 and rows 1 to 3 from
    tag 0 toward tag 3. From the row3 side (opposite the arm) that puts A on the human's
    left and row 3 nearest. From the row1 side (the arm's side) the human faces the other
    way, so both flip: A is on their right and row 1 is nearest."""
    if human_side not in HUMAN_SIDES:
        raise ValueError(f"human_side must be one of {HUMAN_SIDES}, got {human_side!r}")
    (sx, sy), (dx, dy) = cell_xy(src), cell_xy(dst)
    sign = 1 if human_side == "row3" else -1
    right, away = sign * (dx - sx), -sign * (dy - sy)
    parts = []
    if right:
        parts.append(f"{_COUNT[abs(right)]} {'right' if right > 0 else 'left'}")
    if away:
        parts.append(f"{_COUNT[abs(away)]} {'away from you' if away > 0 else 'toward you'}")
    return " and ".join(parts)


def level_phrase(level: int) -> str:
    return _LEVEL.get(level, f"on level {level}")


def unknown_hint(cell: str) -> str:
    return f"I can't tell the color of the block on {cell}. Try nudging it to the center of the square."


def extra_hint(color: str, cell: str) -> str:
    return f"Please take the {color} block off {cell}."


def place_hint(s: Step) -> str:
    if s.level == 0:
        return f"Place a {s.color} cube on {s.cell}."
    return f"Stack a {s.color} cube on {s.cell}, {level_phrase(s.level)}."


# ---------------------------------------------------------------- check


def check(board: Optional[Board], lesson: Lesson, step_index: int, human_side: str = HUMAN_SIDE) -> CheckResult:
    target = lesson.steps[step_index]
    expected = expected_board(lesson, step_index)
    at_target = dict(expected_cell=target.cell, expected_level=target.level)
    if board is None:
        return CheckResult(Verdict.WAITING, detail="no stable board yet", **at_target)
    unreadable = unknown_cells(board)
    if unreadable:
        c = unreadable[0]
        return CheckResult(Verdict.WAITING, c, f"unknown color at {c}", unknown_hint(c), **at_target)
    board = set_aside_early(board, lesson, step_index)
    if boards_equal(board, expected):
        return CheckResult(Verdict.PASS, detail=f"step {step_index} matches", hint_text="Nice! That's right.",
                           **at_target)
    if boards_equal(board, expected_board(lesson, step_index - 1)):
        return CheckResult(Verdict.MISSING, detail=f"waiting for {target.color} at {target.cell} level {target.level}",
                           hint_text=place_hint(target), **at_target)

    missing, extra = slot_diff(expected, board)
    t = Cube(target.cell, target.level, target.color)

    # 3. The newest cube.
    if t in missing:
        same_color = [c for c in extra if c.color == t.color]
        in_cell = [c for c in same_color if c.cell == t.cell]
        if in_cell:
            c = in_cell[0]
            return CheckResult(
                Verdict.WRONG_LEVEL, t.cell,
                f"{t.color} at {t.cell} level {c.level}, expected level {t.level}",
                f"Right cell! The {t.color} cube goes {level_phrase(t.level)}.",
                fault_level=c.level, **at_target,
            )
        if same_color:
            c = min(same_color, key=lambda c: (distance(c.cell, t.cell), CELLS.index(c.cell)))
            return CheckResult(
                Verdict.WRONG_CELL, c.cell,
                f"{t.color} at {c.cell}, expected {t.cell}",
                f"Close! Move the {t.color} cube {direction(c.cell, t.cell, human_side)}.",
                fault_level=c.level, **at_target,
            )
        in_slot = [c for c in extra if (c.cell, c.level) == (t.cell, t.level)]
        if in_slot:
            c = in_slot[0]
            return CheckResult(
                Verdict.WRONG_COLOR, t.cell,
                f"{c.color} at {t.cell} level {t.level}, expected {t.color}",
                f"Right spot! Swap the {c.color} cube for a {t.color} one.",
                fault_level=t.level, **at_target,
            )

    older_missing = [m for m in missing if m != t]

    # 4. An older cube's slot holds the wrong color.
    for m in older_missing:
        for c in extra:
            if (c.cell, c.level) == (m.cell, m.level) and c.color != m.color:
                return CheckResult(
                    Verdict.WRONG_COLOR, m.cell,
                    f"{c.color} at {m.cell} level {m.level}, expected {m.color} (earlier step)",
                    f"The cube on {m.cell} should be {m.color}. Swap the {c.color} cube for a {m.color} one.",
                    fault_level=m.level, expected_cell=m.cell, expected_level=m.level,
                )

    # 5. An older cube moved or got knocked into another slot.
    for m in older_missing:
        moved = [c for c in extra if c.color == m.color and c.cell != m.cell]
        if moved:
            c = min(moved, key=lambda c: (distance(c.cell, m.cell), CELLS.index(c.cell)))
            return CheckResult(
                Verdict.WRONG_CELL, c.cell,
                f"{m.color} from {m.cell} is now at {c.cell} (earlier step)",
                f"A {m.color} cube got moved. Put it back {direction(c.cell, m.cell, human_side)}, on {m.cell}.",
                fault_level=c.level, expected_cell=m.cell, expected_level=m.level,
            )

    # 6. A cube where none belongs: move it if the rest of the lesson needs its color, else off.
    if extra:
        c = extra[0]
        open_slots = [Cube(s.cell, s.level, s.color) for s in lesson.steps[step_index:] if not slot_holds(board, s)]
        needed = [m for m in open_slots  # open now: the slot is the top of its stack
                  if m.color == c.color and m.cell != c.cell and len(board[m.cell]) == m.level]
        if needed:
            m = needed[0]  # the earliest step that needs this color
            return CheckResult(
                Verdict.WRONG_CELL, c.cell,
                f"{c.color} at {c.cell} belongs on {m.cell} level {m.level}",
                f"Close! Move the {c.color} block {direction(c.cell, m.cell, human_side)}, to {m.cell}.",
                fault_level=c.level, expected_cell=m.cell, expected_level=m.level,
            )
        return CheckResult(
            Verdict.EXTRA, c.cell,
            f"unexpected {c.color} at {c.cell} level {c.level}",
            extra_hint(c.color, c.cell),
            fault_level=c.level,
        )

    # 7. Older cubes gone with nothing extra on the board (knocked off the grid).
    m = older_missing[0] if older_missing else missing[0]
    return CheckResult(
        Verdict.MISSING, m.cell,
        f"{m.color} missing from {m.cell} level {m.level}",
        f"A {m.color} cube is missing from {m.cell}. Please put it back." if m != t else place_hint(target),
        expected_cell=m.cell, expected_level=m.level,
    )
