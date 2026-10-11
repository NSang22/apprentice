"""Gate: checks an extracted lesson against the history it came from (README section 6.4).

run_gate() replays the steps on an empty board and runs every check below over the steps and the
history of the recording (brain/extractor.py builds it: which cubes each settled board added and
removed, what each cell's top read at the end, and the events the history refused). Each check is
recorded by name in a shared Gate; the gate passes only if every check passes. Every failed check
also gets a one-line reason for a person to read (Gate has no field for it, so the extractor
saves it next to the lesson).

  replay_matches_final_frame  the steps, replayed on an empty board, leave the same cells occupied
                              as the last settled board, with the same top color. A final read that
                              is unclear but close to the expected top passes ("weak read"); a step
                              colored by the manual colors file stands in for an unreadable final
                              read ("manual, unverified"). A clear different color, or an unclear
                              read that is not close and not manual, fails
  starts_empty                the first settled board has no cubes, and the reference the camera
                              compared against was itself checked empty (reference_empty is False
                              when that independent check failed or could not run)
  one_cube_per_change         each settled board adds one cube, removes one, or moves one to
                              another cell (a cell with 2 or 3 cubes going empty at once fails)
  one_cube_per_color          a known color never shows up while a cube of that color is still on
                              the board, and no two steps share a color (the robot refers to
                              blocks by color, and the team uses one cube of each)
  levels_match_stack_height   every step's level is the number of cubes already in its cell, at
                              most MAX_LEVEL, and no fourth cube was seen on a cell
  has_steps                   at least one step
  colors_known                every step has a known color

None of these use the camera's own stack height. Pure functions only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from shared.config import CELLS, COLORS, MAX_LEVEL, UNKNOWN
from shared.schemas import Gate, GateCheck, Step

Slot = tuple[str, int]  # (cell, level)

CHECK_NAMES = (
    "replay_matches_final_frame",
    "starts_empty",
    "one_cube_per_change",
    "one_cube_per_color",
    "levels_match_stack_height",
    "has_steps",
    "colors_known",
)


@dataclass
class Change:
    t: float
    added: list[Slot]
    removed: list[Slot]


@dataclass
class History:
    first_occupied: Optional[list[str]]          # cells occupied on the first settled board (None: no board)
    final_tops: Optional[dict[str, Optional[str]]]  # last settled board: cell -> top read, None if empty
    changes: list[Change] = field(default_factory=list)  # one per settled board after the first
    problems: list[tuple[str, str]] = field(default_factory=list)  # (check name, reason) the history refused
    final_distances: dict[str, dict[str, float]] = field(default_factory=dict)  # last board: cell -> color -> distance
    weak_limit: Optional[float] = None   # an unclear final read this close to the expected top is a weak read


@dataclass
class GateResult:
    gate: Gate
    reasons: dict[str, str] = field(default_factory=dict)  # check name -> why it failed
    notes: dict[str, list[str]] = field(default_factory=dict)  # check name -> why it passed with a caveat


def one_cube(change: Change) -> bool:
    """Nothing, one cube added, one removed, or one moved to another cell."""
    a, r = change.added, change.removed
    if len(a) + len(r) <= 1:
        return True
    return len(a) == 1 and len(r) == 1 and a[0][0] != r[0][0]


def _slots(xs: list[Slot]) -> str:
    return ", ".join(f"{c} L{lv}" for c, lv in xs) or "nothing"


def run_gate(steps: list[Step], history: History, reference_empty: Optional[bool] = None,
             manual_steps: frozenset[int] | set[int] = frozenset()) -> GateResult:
    """reference_empty: None when the camera's empty-board reference did not come from the
    recording, else whether it was checked empty. manual_steps: indexes of steps whose color came
    from the manual colors file."""
    results: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    notes: dict[str, list[str]] = {}

    def fail(name: str, reason: str) -> None:
        results[name] = False
        reasons.setdefault(name, reason)

    for name in CHECK_NAMES:
        results[name] = True
    for name, reason in history.problems:
        fail(name, reason)

    replay: dict[str, list[str]] = {cell: [] for cell in CELLS}
    top_step: dict[str, int] = {}
    for i, s in enumerate(steps):
        top_step[s.cell] = i
        if s.level != len(replay[s.cell]) or s.level > MAX_LEVEL:
            fail("levels_match_stack_height", f"step {i + 1} sits at level {s.level} on {s.cell}, "
                                              f"which then held {len(replay[s.cell])} cube(s)")
        replay[s.cell].append(s.color)

    if history.final_tops is None:
        fail("replay_matches_final_frame", "no settled board in the recording")
    else:
        for cell in CELLS:
            top = replay[cell][-1] if replay[cell] else None
            seen = history.final_tops.get(cell)
            if (top is None) != (seen is None):
                fail("replay_matches_final_frame", f"{cell}: the replay leaves it {'empty' if top is None else 'occupied'}, "
                                                   f"the last board shows it {'empty' if seen is None else 'occupied'}")
            elif top is not None and seen == UNKNOWN:
                d = history.final_distances.get(cell, {}).get(top)
                if history.weak_limit is not None and d is not None and d <= history.weak_limit:
                    notes.setdefault("replay_matches_final_frame", []).append(
                        f"{cell}: weak read, {d:.1f} from {top} (limit {history.weak_limit:g})")
                elif top_step.get(cell) in manual_steps:
                    notes.setdefault("replay_matches_final_frame", []).append(f"{cell}: manual, unverified ({top})")
                else:
                    fail("replay_matches_final_frame", f"{cell}: the last board cannot read the top color")
            elif top is not None and top != seen:
                fail("replay_matches_final_frame", f"{cell}: the replay's top is {top}, the last board reads {seen}")

    if history.first_occupied is None:
        fail("starts_empty", "no settled board in the recording")
    elif history.first_occupied:
        fail("starts_empty", "the first settled board already has cubes on " + ", ".join(history.first_occupied))
    elif reference_empty is False:
        fail("starts_empty", "the empty-board picture came from the video and could not be confirmed empty on its own")

    for c in history.changes:
        if not one_cube(c):
            fail("one_cube_per_change", f"board at {c.t:.1f} s: removed {_slots(c.removed)}; added {_slots(c.added)}")
            break

    twice = sorted(color for color, n in Counter(s.color for s in steps if s.color in COLORS).items() if n > 1)
    if twice:
        fail("one_cube_per_color", "more than one step uses " + ", ".join(twice))

    if not steps:
        fail("has_steps", "no cube appeared in the recording")

    unknown = [i + 1 for i, s in enumerate(steps) if s.color not in COLORS]
    if unknown:
        fail("colors_known", "no known color for step " + ", ".join(map(str, unknown)))

    checks = [GateCheck(name, results[name]) for name in CHECK_NAMES]
    return GateResult(Gate(passed=all(c.passed for c in checks), checks=checks), reasons, notes)
