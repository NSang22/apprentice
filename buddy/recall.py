"""Picture Recall: a pattern shows for a few seconds, hides, and the child builds it from memory.

Pure logic (boards and patterns in, results out), so it is testable without a camera, a phone,
or a server. buddy/session.py runs the game loop around it.

A pattern is {cell: color}, flat (level 0 only), at most MAX_PER_COLOR blocks of one color (the
kit has two of each) and only colors perception can tell apart (robot ACTIVE_COLORS).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from shared.config import CELLS, UNKNOWN

MAX_PER_COLOR = 2


@dataclass(frozen=True)
class Level:
    blocks: int
    show_s: float


# Each level is a little harder than the last: one more block, or the same blocks for less time.
LEVELS = [
    Level(2, 6.0), Level(3, 6.0), Level(3, 4.5), Level(4, 6.0), Level(4, 4.5),
    Level(5, 6.0), Level(5, 5.0), Level(6, 7.0), Level(6, 5.5), Level(7, 7.0),
]


def level(n: int) -> Level:
    """Level n, counted from 1. Past the last level the last one repeats."""
    return LEVELS[max(0, min(n, len(LEVELS)) - 1)]


def make_pattern(blocks: int, colors: list[str], rng: random.Random) -> dict[str, str]:
    if blocks > min(len(CELLS), MAX_PER_COLOR * len(colors)):
        raise ValueError(f"cannot place {blocks} blocks with {len(colors)} colors")
    cells = rng.sample(CELLS, blocks)
    supply = [c for c in colors for _ in range(MAX_PER_COLOR)]
    rng.shuffle(supply)
    return {cell: supply[i] for i, cell in enumerate(cells)}


def flat(board: dict[str, list[str]]) -> dict[str, str]:
    """{cell: color} for every occupied cell (the bottom block; the game is flat)."""
    return {cell: stack[0] for cell, stack in board.items() if stack}


def mirror(cell: str) -> str:
    """The same row, other side (A <-> C): a left-right mix-up."""
    return {"A": "C", "B": "B", "C": "A"}[cell[0]] + cell[1]


RIGHT, WRONG_COLOR, MISSING, EXTRA = "right", "wrong_color", "missing", "extra"


@dataclass
class Grade:
    cells: dict[str, str]                       # cell -> RIGHT / WRONG_COLOR / MISSING / EXTRA
    right: int
    total: int
    errors: dict[str, int] = field(default_factory=dict)  # color, place, missing, extra, mirror

    @property
    def perfect(self) -> bool:
        return self.right == self.total and not any(v == EXTRA for v in self.cells.values())

    def to_dict(self) -> dict:
        return {"cells": self.cells, "right": self.right, "total": self.total,
                "errors": self.errors, "perfect": self.perfect}


def grade(target: dict[str, str], board: dict[str, list[str]]) -> Grade:
    """Compare the child's board to the pattern, cell by cell.

    errors counts kinds of mistakes for the learning metrics:
      color    the right cell holds the wrong color
      place    a block of a missing color sits in a cell the pattern leaves empty
      mirror   ... and that cell is the missing one's mirror image (left-right mix-up)
      missing  a pattern cell is empty and its color is nowhere else
      extra    a block the pattern has no place for
    """
    built = flat(board)
    cells: dict[str, str] = {}
    errors = {"color": 0, "place": 0, "mirror": 0, "missing": 0, "extra": 0}
    for cell, color in target.items():
        have = built.get(cell)
        cells[cell] = RIGHT if have == color else (MISSING if have is None else WRONG_COLOR)
        if cells[cell] == WRONG_COLOR:
            errors["color"] += 1
    extras = {cell: color for cell, color in built.items() if cell not in target}
    for cell in extras:
        cells[cell] = EXTRA
    unmatched = dict(extras)
    for cell, color in target.items():
        if cells[cell] != MISSING:
            continue
        spot = mirror(cell) if unmatched.get(mirror(cell)) == color else next(
            (c for c, col in unmatched.items() if col == color), None)
        if spot is None:
            errors["missing"] += 1
            continue
        errors["place"] += 1
        if spot == mirror(cell):
            errors["mirror"] += 1
        del unmatched[spot]
    errors["extra"] = len(unmatched)
    right = sum(1 for v in cells.values() if v == RIGHT)
    return Grade(cells, right, len(target), errors)


def has_unknown(board: dict[str, list[str]]) -> bool:
    return any(UNKNOWN in stack for stack in board.values())


def block_count(board: dict[str, list[str]]) -> int:
    return sum(1 for stack in board.values() if stack)


def xp_for(g: Grade, level_n: int, hints: int = 0) -> int:
    """5 per right block, a perfect bonus that grows with the level, never below 1 for trying."""
    xp = 5 * g.right + (5 + 2 * level_n if g.perfect else 0) - 3 * hints
    return max(1, xp)


def mood_line(g: Grade, name: Optional[str] = None) -> tuple[str, str]:
    """(mood, what the buddy says). Short, kind, never blaming."""
    who = f", {name}" if name else ""
    if g.perfect:
        return "cheer", f"Amazing{who}! You remembered all {g.total}!"
    if g.errors.get("mirror"):
        return "think", "So close! One block went to the other side. Left and right are tricky!"
    if g.right >= g.total - 1:
        return "think", f"Almost{who}! {g.right} of {g.total} right. Let's try another one!"
    if g.right:
        return "encourage", f"Good try! You got {g.right} right. Look closely next time!"
    return "encourage", "That one was tricky! Let's try again together."
