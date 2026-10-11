"""Gate: checks an extracted lesson against the boards it came from (README section 6.4).

run_gate() replays the steps on an empty board and runs every check below over the steps and
the settled boards of the recording (brain/extractor.py). Each check is recorded by name in a
shared Gate; the gate passes only if every check passes. Every failed check also gets a one-line
reason for a person to read (Gate has no field for it, so the extractor saves it next to the
lesson).

  replay_matches_final_frame  the steps, replayed on an empty board, put cubes in exactly the
                              slots the last settled board has
  starts_empty                the first settled board has no cubes, and the reference the
                              camera compared against was itself checked empty (reference_empty
                              is False when that independent check failed or could not run)
  one_cube_per_change         each settled board differs from the previous one by one cube
                              added, one removed, or one moved to another cell
  levels_match_stack_height   every step's level is the stack height of its cell at that moment
  has_steps                   at least one step
  colors_known                every step has a known color

Every check but colors_known works on positions (which slots hold a cube), so a recording whose
colors could not be read can still pass them. Pure functions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shared.config import CELLS, COLORS
from shared.schemas import Board, Gate, GateCheck, Step

Slot = tuple[str, int]  # (cell, level)

CHECK_NAMES = (
    "replay_matches_final_frame",
    "starts_empty",
    "one_cube_per_change",
    "levels_match_stack_height",
    "has_steps",
    "colors_known",
)


@dataclass
class GateResult:
    gate: Gate
    reasons: dict[str, str] = field(default_factory=dict)  # check name -> why it failed


def heights(board: Board) -> dict[str, int]:
    return {cell: len(board[cell]) for cell in CELLS}


def slots(board: Board) -> set[Slot]:
    return {(cell, level) for cell in CELLS for level in range(len(board[cell]))}


def slot_change(old: Board, new: Board) -> tuple[list[Slot], list[Slot]]:
    """(removed, added) slots between two boards, by position only."""
    a, b = slots(old), slots(new)
    return sorted(a - b), sorted(b - a)


def one_cube(removed: list[Slot], added: list[Slot]) -> bool:
    """One cube added, one removed, or one moved to another cell."""
    if (len(removed), len(added)) in ((0, 1), (1, 0)):
        return True
    return len(removed) == 1 and len(added) == 1 and removed[0][0] != added[0][0]


def replay_heights(steps: list[Step]) -> tuple[dict[str, int], Optional[int]]:
    """Stack heights after placing every step, and the index of the first step whose level was
    not its cell's height at that moment (None if every level matched)."""
    h = {cell: 0 for cell in CELLS}
    bad = None
    for i, s in enumerate(steps):
        if s.level != h[s.cell] and bad is None:
            bad = i
        h[s.cell] += 1
    return h, bad


def _describe(removed: list[Slot], added: list[Slot]) -> str:
    def fmt(xs: list[Slot]) -> str:
        return ", ".join(f"{c} L{lv}" for c, lv in xs) or "nothing"
    return f"removed {fmt(removed)}; added {fmt(added)}"


def run_gate(steps: list[Step], boards: list[Board], times: Optional[list[float]] = None,
             reference_empty: Optional[bool] = None) -> GateResult:
    """steps: the lesson's steps; boards: every settled board of the recording, in order;
    times: their timestamps (only for the reasons); reference_empty: None when the camera's
    empty-board reference did not come from the recording, else whether it was checked empty."""
    times = times if times is not None else [float(i) for i in range(len(boards))]
    results: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    final_h, bad_level = replay_heights(steps)
    if not boards:
        results["replay_matches_final_frame"] = False
        reasons["replay_matches_final_frame"] = "no settled board in the recording"
    else:
        last = heights(boards[-1])
        diff = [c for c in CELLS if final_h[c] != last[c]]
        results["replay_matches_final_frame"] = not diff
        if diff:
            reasons["replay_matches_final_frame"] = "replay and last board differ on " + ", ".join(
                f"{c} ({final_h[c]} vs {last[c]} cubes)" for c in diff)

    first_empty = bool(boards) and not slots(boards[0])
    results["starts_empty"] = first_empty and reference_empty is not False
    if not boards:
        reasons["starts_empty"] = "no settled board in the recording"
    elif not first_empty:
        reasons["starts_empty"] = "the first settled board already has cubes on " + ", ".join(
            sorted({c for c, _ in slots(boards[0])}))
    elif reference_empty is False:
        reasons["starts_empty"] = ("the empty-board picture came from the video and could not be "
                                   "confirmed empty on its own")

    bad_change = None
    for i in range(1, len(boards)):
        removed, added = slot_change(boards[i - 1], boards[i])
        if (removed or added) and not one_cube(removed, added):
            bad_change = (i, removed, added)
            break
    results["one_cube_per_change"] = bad_change is None
    if bad_change is not None:
        i, removed, added = bad_change
        reasons["one_cube_per_change"] = f"board at {times[i]:.1f} s: {_describe(removed, added)}"

    results["levels_match_stack_height"] = bad_level is None
    if bad_level is not None:
        s = steps[bad_level]
        reasons["levels_match_stack_height"] = f"step {bad_level + 1} sits at level {s.level} on {s.cell}, " \
                                               "which was not its stack height then"

    results["has_steps"] = bool(steps)
    if not steps:
        reasons["has_steps"] = "no cube appeared in the recording"

    unknown = [i + 1 for i, s in enumerate(steps) if s.color not in COLORS]
    results["colors_known"] = not unknown
    if unknown:
        reasons["colors_known"] = "no known color for step " + ", ".join(map(str, unknown))

    checks = [GateCheck(name, results[name]) for name in CHECK_NAMES]
    return GateResult(Gate(passed=all(c.passed for c in checks), checks=checks), reasons)
