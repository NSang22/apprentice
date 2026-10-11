"""Stacks: which colors make up each cell's pile, across frames.

Three signals, combined so that disagreement reads "unknown" and is never guessed. The first
two come from the per-cell board model (perception/cellmodel.py):

  geometry   the stack height whose rendered column best explains the cell
  side bands with level >= 1, the color of each lower block's visible side band (or None when
             too little of it is in view, as near the camera's nadir)
  memory     StackMemory follows each cell across stable readings: the same cell reading one
             level higher with a new top is a push, one level lower with the old second color on
             top is a pop, empty resets it. Anything else (a stack swapped in while hands were
             over the board) marks the cell's history as conflicting until it reads empty.

resolve(): the stack is the memory when it matches the geometry and every counted side band;
with no usable memory, the side bands plus the top when all of them were seen; otherwise
"unknown" with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shared.config import CELLS, UNKNOWN

MEMORY_STABLE_FRAMES = 3   # a cell's (level, top) must hold this many usable frames to update memory


# ---------------------------------------------------------------- memory


@dataclass(frozen=True)
class Reading:
    """One cell in one frame: the stack height index (0 = one block), the top color, and the
    colors seen on the side bands under it (bottom first, None = not seen)."""
    level: Optional[int]
    top: str
    sides: tuple[Optional[str], ...] = ()


_NO_HISTORY = object()


@dataclass
class StackMemory:
    stable_frames: int = MEMORY_STABLE_FRAMES
    stacks: dict = field(default_factory=dict)    # cell -> list of colors, or None (conflict)
    pending: dict = field(default_factory=dict)   # cell -> ((level, top) or None, frames held)

    def update(self, readings: dict[str, Optional[Reading]], usable: bool) -> None:
        """Feed one frame's readings (None = the cell is empty). Unusable frames (hands, grid
        not locked) teach nothing and restart every cell's count."""
        if not usable:
            self.pending.clear()
            return
        for cell in CELLS:
            r = readings.get(cell)
            key = None if r is None else (r.level, r.top)
            held = self.pending.get(cell)
            count = held[1] + 1 if held and held[0] == key else 1
            self.pending[cell] = (key, count)
            if count == self.stable_frames:  # just became stable: act once
                self._transition(cell, r)

    def _transition(self, cell: str, r: Optional[Reading]) -> None:
        if r is None:
            self.stacks[cell] = []        # empty: forget
            return
        m = self.stacks.get(cell, [])
        if m is None or r.level is None or r.top == UNKNOWN:
            return                        # conflicting history waits for empty; no information
        level, top = r.level, r.top
        if not m:
            if level == 0:
                self.stacks[cell] = [top]
            elif all(s is not None for s in r.sides):
                self.stacks[cell] = list(r.sides) + [top]   # first seen as a stack: trust the sides
            else:
                self.stacks.pop(cell, None)                  # no history to go on
        elif len(m) == level + 1 and m[-1] == top:
            pass                                             # unchanged
        elif level == len(m):
            self.stacks[cell] = m + [top]                    # push
        elif level == len(m) - 2 and m[-2] == top:
            self.stacks[cell] = m[:-1]                       # pop
        else:
            self.stacks[cell] = None                         # conflict until the cell reads empty

    def resolve(self, cell: str, r: Reading) -> tuple[Optional[list[str]], str]:
        """(stack bottom to top, "") or (None, why the cell is unknown)."""
        if r.level is None:
            return None, "stack height unclear"
        if r.top == UNKNOWN:
            return None, "top color unclear"
        m = self.stacks.get(cell, _NO_HISTORY)
        if m is None:
            return None, "stack history conflicts: clear this cell"
        seen = [(k, c) for k, c in enumerate(r.sides) if c is not None]
        if m is _NO_HISTORY or len(m) != r.level + 1 or m[-1] != r.top:
            if len(seen) == r.level:
                return list(r.sides) + [r.top], ""
            return None, f"{r.level + 1} high but the colors under the top are hidden"
        for k, c in seen:
            if m[k] != c:
                return None, f"level {k + 1} looks {c}, but {m[k]} was placed there"
        return list(m), ""


def unknown_stack(level: Optional[int]) -> list[str]:
    return [UNKNOWN] * ((level if level is not None else 0) + 1)

