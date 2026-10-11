"""Take Turns: the child and Pip build a pattern together, one block each, Pip first.

The pattern stays on the phone the whole time (a cooperative copy game, not a memory test). On
Pip's turn the arm places one block from the supply below the grid; on the child's turn they
place one block and tap "Your turn, Pip!". Progress always comes from the board itself, so a
block moved or taken off is noticed on the next look.

Pure logic here (boards and patterns in, decisions out); buddy/turns_session.py runs the game.
Cells are named as printed on the sheet (perception/cells.py): letter = row, A on Pip's edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from buddy.recall import flat

# One lesson: the pattern grows from 4 to 6 blocks (the kit has 2 of each color).
BLOCKS_PER_ROUND = [4, 4, 5, 6, 6]


@dataclass
class Look:
    """The board compared with the pattern."""
    right: list[str]                                     # pattern cells holding their color
    missing: list[str]                                   # pattern cells still empty
    wrong: list[dict] = field(default_factory=list)      # {cell, have, want}: want None = no block belongs here

    @property
    def done(self) -> bool:
        return not self.missing and not self.wrong


def look(target: dict[str, str], board: dict[str, list[str]]) -> Look:
    built = flat(board)
    right = sorted(c for c, col in target.items() if built.get(c) == col)
    missing = sorted(c for c in target if c not in built)
    wrong = [{"cell": c, "have": col, "want": target.get(c)} for c, col in sorted(built.items())
             if target.get(c) != col]
    return Look(right, missing, wrong)


def pip_next(target: dict[str, str], board: dict[str, list[str]]) -> Optional[tuple[str, str]]:
    """(cell, color) for Pip's next block: an empty pattern cell, nearest Pip's own edge first
    (row A, then B, then C) so the arm reaches less toward the child. None when nothing is left."""
    open_cells = look(target, board).missing
    if not open_cells:
        return None
    cell = min(open_cells, key=lambda c: (c[0], c[1]))
    return cell, target[cell]


def where_it_goes(target: dict[str, str], board: dict[str, list[str]], wrong: dict) -> Optional[str]:
    """The square Pip points at for a misplaced block: an empty pattern cell that needs this
    block's color, nearest the block. None when no open square wants that color (take it off)."""
    built = flat(board)
    homes = [c for c, col in target.items() if col == wrong["have"] and c not in built]
    if not homes:
        return None
    r, n = "ABC".index(wrong["cell"][0]), int(wrong["cell"][1])
    return min(homes, key=lambda c: (abs("ABC".index(c[0]) - r) + abs(int(c[1]) - n), c))


def child_turn(target: dict[str, str], board: dict[str, list[str]], right_before: int) -> dict:
    """Judge the child's turn once they tap "Your turn, Pip!".

    right_before: pattern cells that should already be right at this point (everything placed
    so far, including a block of Pip's the child was asked to move).
      ok      no wrong blocks and at least one more right block than before
      wrong   a block sits where the pattern does not want it: point (cell to point at) or None
      none    no new block yet
    """
    seen = look(target, board)
    if seen.wrong:
        w = seen.wrong[0]
        point = where_it_goes(target, board, w) if w["want"] is None else w["cell"]
        return {"result": "wrong", "wrong": w, "point": point, "look": seen}
    if len(seen.right) <= right_before:
        return {"result": "none", "look": seen}
    return {"result": "ok", "look": seen}


def pip_landed(target: dict[str, str], board: dict[str, list[str]], cell: str, color: str) -> bool:
    return flat(board).get(cell) == color


def stars_for(mistakes: int) -> int:
    return 3 if mistakes == 0 else 2 if mistakes == 1 else 1


def wrong_line(w: dict, point: Optional[str]) -> str:
    """What Pip says about a misplaced block. Short and kind."""
    if w["want"] is not None:
        return f"That square needs a {w['want']} block! Can you swap it?"
    if point:
        return f"Almost! That {w['have']} block goes on the glowing square. Can you move it?"
    return f"We don't need a {w['have']} block there. Can you take it off?"
