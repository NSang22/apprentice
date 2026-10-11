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


# One lesson: 1, 2, 3, then 4 blocks, then new 4-block patterns.
BLOCKS_PER_ROUND = [1, 2, 3, 4, 4, 4, 4, 4]

# How long the pattern shows, by difficulty (the grown-up picks it before the lesson).
LOOK_S = {"easy": 7.0, "medium": 5.0, "hard": 3.0}
# How long the child has to build, picked for the child's age.
BUILD_S = (10, 15, 20, 30)
# After a hint or a fix: time to finish the pattern.
RETRY_S = 20


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
    """The same row, other side: a left-right mix-up for the child. Cells are named as printed on
    the sheet (perception/cells.py): the letter is the row, the number the column (1 to 3, left to
    right for a child opposite the arm), so left and right are columns 1 and 3."""
    return cell[0] + {"1": "3", "2": "2", "3": "1"}[cell[1]]


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


def stars_for(g: Grade, helped: bool) -> int:
    """3 stars right first time, 2 right after a hint or fix, 1 for trying."""
    if g.perfect:
        return 2 if helped else 3
    return 1


def xp_for(g: Grade, stars: int) -> int:
    return 5 * stars + g.right


def hint_cell(target: dict[str, str], g: Grade) -> Optional[str]:
    """The one square Pip points at: a missing block first (where to put one), then a wrong
    color, then a block that does not belong."""
    for kind in (MISSING, WRONG_COLOR, EXTRA):
        for cell in sorted(g.cells):
            if g.cells[cell] == kind:
                return cell
    return None


def fix_cells(g: Grade) -> list[str]:
    """Blocks Pip takes off: wrong colors and blocks that do not belong."""
    return sorted(c for c, v in g.cells.items() if v in (WRONG_COLOR, EXTRA))


def hint_line(target: dict[str, str], g: Grade, cell: str) -> str:
    if g.cells[cell] == MISSING:
        return f"This square needs a {target[cell]} block!"
    if g.cells[cell] == WRONG_COLOR:
        return f"This one should be {target[cell]}!"
    return "This block doesn't belong here!"


def mood_line(g: Grade, name: Optional[str] = None, helped: bool = False) -> tuple[str, str]:
    """(mood, what the buddy says). Short, kind, never blaming."""
    who = f", {name}" if name else ""
    if g.perfect:
        return "cheer", (f"You fixed it{who}! Great job!" if helped else f"Amazing{who}! You remembered it all!")
    if helped:
        return "encourage", "Good try! Here's how it looked. Let's do the next one!"
    if g.errors.get("mirror"):
        return "think", "So close! A block went to the other side. Want some help?"
    return "think", "Not quite! Want a hint, or should I take off the wrong blocks?"
