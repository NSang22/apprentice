"""Extractor: turns a recorded demonstration into a Lesson (README section 6.3).

    python -m brain.extractor --video videos/expert_take1.mp4 --color-memory videos/calib_oct11/color_memory.json
    python -m brain.extractor --script tests/scripts/staircase_clean.json
    python -m brain.extractor --grid ws://localhost:8765 --source-video videos/take.mp4   # Ctrl+C ends the take

The extractor only ever sees a stream of GridState. --video runs perception on the file inside
this process (brain/video_source.py, imported only for --video); --script builds the frames of a
fake feed script in memory; --grid reads a live feed.

Usable frames: the stabilizer only takes a frame perception calls clear; with --video the hand band
in brain/video_source.py also has to be unchanged. At the end one more board is always taken from
the last frames the stabilizer would accept (the final read, used by the gate and for any Gemini
crop). A recording with no usable frame in its last FINAL_GAP_S seconds stops with an error. A
settled board whose occupancy differs from the last accepted board in BACKSTOP_CELLS or more cells
is skipped and logged (a sleeve or arm read as blocks).

Height comes from history, not from the camera's height read. From every frame only two facts per
cell are used: is it occupied, and what is its top color (the camera's own stack is kept in the
extract record as evidence only). The existing stabilizer settles those boards. One map for the
whole recording holds each cell's cubes, bottom to top, at most MAX_LEVEL + 1. Between settled
boards, removals come first, then new cubes on empty cells (the strongest evidence), then what the
tops of occupied cells read:
  a cell goes empty                     its cube is removed (2 or 3 cubes at once: the gate fails)
  the top reads the color right under   the top cube was removed
  an empty cell becomes occupied        a step at level 0
  the top reads a clear new color       a step at level = cubes already there (a fourth: refused)
  the top reads unclear, but close      a weak read of the same cube: it keeps its color (logged)
  the top reads unclear, and not close  something else on the cell: a step at the next level, color
                                        unknown (a draft until a colors file names it); if the old
                                        top color comes back, it was a transient
  the top reads the same                nothing happened
A top read is clear when a source knows it: the robot's color memory first, then the camera. It is
easier to keep a color than to get one: an unclear read is close when its robot-memory distance to
the top's color is at most WEAK_READ_MAX_DISTANCE, looser than the memory's own limit for a clear
name (robot/perception/color.py KNOWN_MAX 18 plus MIN_MARGIN 8). The camera gives no distance, so
without a color memory an unclear read is never close.
A new cube on an empty cell whose color is already on the board fails the gate (one cube per
color); an occupied cell whose top reads a color already on the board keeps its old color, adds no
step, and the read is logged as conflicting. A new cube is named only among colors not yet on the
board: with robot-memory distances, the clear-read limits are measured against those colors only, and
an unclear top that is nearer such a color than its own is a new cube of that color. Position first, color second, per level: a cube is its slot (cell, level) from the moment
it appears until it is removed, and every read taken while it is the top is kept. A cube that is
removed deletes the step that placed it (a correction); t_video is when the board with the new
cube first appeared.

Hidden, not removed: a cube that disappears and whose color later comes back on the same cell at
the same level was hidden by an arm. Its removal is undone (restore_hidden): it keeps its step and
time, no new step, no correction; it is logged as hidden.

Transients: something that appeared in a slot, was removed before the end, and never had a known
color from any source (camera or robot memory) while it was there, such as a hand read as a block.
It is not a cube: no step, no correction, and one_cube_per_change does not count it. Each one is
printed with a warning and listed in the extract file. Anything that ever had a known color, or
is still on the board at the end, is a cube.

Colors are never guessed. Three sources, in this order of trust (choose_color(), the one place
a step's color is decided, so a color model can plug in there later):
  manual        --colors-file, per step (step numbers as printed in the step table):
                [{"step": 2, "cell": "B2", "color": "orange"}]. It colors steps; it cannot
                create or remove them.
  robot_memory  --color-memory (--video; default robot/outputs/color_memory.json if it exists):
                the robot's ColorMemory file, blocks named the way the robot names them
  camera        perception's reads in the GridState stream
  gemini        (--video, offline; --no-gemini turns it off) only for a step still in the lesson at
                the end and still unknown: the crop from the clearest settled board where it was on
                top, a closed answer (a shared color or "none", brain/gemini_colors.py). "none", or a
                color already on the board, is rejected and the step stays unknown (a draft). Answers
                are saved in the extract file and reused on a rerun of the same video, so runs match.
                No GEMINI_API_KEY: skipped with a warning. The robot's color memory is never written.
The first source with a known color wins; the extract file keeps every source's read per step.

Output, next to --out (default lessons/<lesson_id>.json):
  lessons/<id>.json             only when every step has a known color; always loads with
                                Lesson.from_dict, gate passed or failed
  lessons/extract/<id>.json     always: where the lesson came from, per-step color reads and
                                sources, corrections, transients, frame counts, gate reasons, and
                                the draft lesson plus a colors-file template when a color is unknown
  lessons/extract/<id>/step_<n>_<cell>.png   (--video) the cell when that step's board settled
Exit code: 0 gate passed, 1 lesson written but the gate failed, 3 draft only (unknown colors),
2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from brain.gate import Change, GateResult, History, run_gate
from brain.gemini_colors import GeminiUnavailable
from brain.provenance import git_info
from brain.stabilizer import StabilizerState, frame_board, step as stabilizer_step
from shared.config import CELLS, COLORS, GRID_WS_PORT, LESSONS_DIR, MAX_LEVEL, REPO_ROOT, UNKNOWN
from shared.schemas import Board, GridState, Lesson, Step, copy_board, empty_board

EXIT_PASSED, EXIT_GATE_FAILED, EXIT_BAD_INPUT, EXIT_DRAFT = 0, 1, 2, 3
DEFAULT_BLOCK_MM = 48.0

RobotRead = Callable[[int, str], Optional[str]]  # (frame id, cell) -> robot-memory name of the top block
RobotDistance = Callable[[int, str], Optional[dict[str, float]]]  # (frame id, cell) -> color -> distance

# An unclear read within this robot-memory distance of the top's color is a weak read of the same
# cube. Measured on expert_take1 (videos/calib_oct11/color_memory.json): the one board where no source
# read a placed cube clearly was B3 orange at 22.9; the hand over C3 read 29.3 from its nearest color.
WEAK_READ_MAX_DISTANCE = 26.0
# A clear robot-memory read: nearest color within this distance and ahead of the next by this margin
# (robot/perception/color.py KNOWN_MAX, MIN_MARGIN; --video passes the robot's own values).
CLEAR_MAX_DISTANCE, CLEAR_MIN_MARGIN = 18.0, 8.0
FINAL_GAP_S = 3.0       # the recording must have a usable frame within this many seconds of its end
BACKSTOP_CELLS = 3      # a settled board changing this many cells' occupancy at once is skipped


# ---------------------------------------------------------------- settling and tracking (pure)


@dataclass(frozen=True)
class Settled:
    t: float           # when this board first appeared (start of its stable run), source seconds
    t_settled: float   # the frame that settled it
    frame_id: int
    board: Board       # occupancy and top only: each cell [] or [top color]
    camera_board: Optional[Board] = None  # the camera's own stacks on that frame (evidence only)
    camera_confidence: Optional[dict[str, float]] = None  # the camera's confidence per cell on that frame
    final: bool = False  # the final read, taken at the end even if nothing changed


@dataclass
class Cube:
    cell: str
    level: int
    t_video: float
    frame_id: int                  # frame of the settled board where it first appeared
    reads: list[str] = field(default_factory=list)            # camera top read, per board it was the top
    robot: list[Optional[str]] = field(default_factory=list)  # robot-memory read, same boards (None: not read)
    seen: list[str] = field(default_factory=list)             # the clear read used (robot, else camera)
    named: list[str] = field(default_factory=list)            # robot-memory names among colors not on the board
    settles: list[int] = field(default_factory=list)          # frame ids of those boards
    baseline: bool = False         # already on the first settled board (not a step)
    t_removed: Optional[float] = None
    seq: int = 0                   # creation order, so steps keep the order the cubes appeared in

    def color(self) -> str:
        return agreed_color(self.seen)

    def camera_color(self) -> str:
        return agreed_color(self.reads)

    def read_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.reads).items()))

    def robot_reads(self) -> list[str]:
        return [r for r in self.robot if r is not None]


def agreed_color(reads: list[str]) -> str:
    """The color all known reads agree on; "unknown" when there is none or they disagree."""
    known = {r for r in reads if r != UNKNOWN}
    return known.pop() if len(known) == 1 else UNKNOWN


@dataclass
class Extraction:
    settled: list[Settled]
    steps: list[Cube]            # cubes still on the board at the end, in the order they appeared
    removed: list[Cube]          # cubes that appeared and later left (expert corrections)
    baseline: list[Cube]
    transients: list[Cube] = field(default_factory=list)  # not cubes: see split_transients()
    changes: list[tuple[float, list[Cube], list[Cube]]] = field(default_factory=list)  # (t, added, removed)
    problems: list[tuple[str, str]] = field(default_factory=list)  # (gate check, reason) the history refused
    emptied: list[tuple[float, str, list[Cube]]] = field(default_factory=list)  # (t, cell, cubes) gone at once
    tops: list[dict[str, str]] = field(default_factory=list)  # per settled board: occupied cell -> top read
    distances: list[dict[str, dict[str, float]]] = field(default_factory=list)  # per board: cell -> color -> distance
    weak_reads: list[dict[str, Any]] = field(default_factory=list)  # unclear reads that kept the top's color
    conflicts: list[dict[str, Any]] = field(default_factory=list)   # tops that read a color already on the board
    skipped: list[dict[str, Any]] = field(default_factory=list)     # settled boards the backstop skipped
    hidden: list[dict[str, Any]] = field(default_factory=list)      # removals undone: an arm hid the cube
    robot_used: bool = False
    frames: int = 0
    unusable: dict[str, int] = field(default_factory=dict)  # reason -> frames the stabilizer could not use

    @property
    def boards_with_unknown(self) -> int:
        return sum(1 for s in self.settled if any(UNKNOWN in s.board[c] for c in CELLS))


def unusable_reason(gs: GridState) -> Optional[str]:
    """Why the stabilizer cannot use this frame, or None if it can."""
    if frame_board(gs) is not None:
        return None
    if not gs.hands_clear:
        return "hands or an object over the grid"
    if not gs.arm_clear:
        return "arm not clear"
    return "an empty cell read below MIN_CONFIDENCE"


def occupancy_and_top(gs: GridState) -> GridState:
    """The frame with each cell reduced to the two facts the extractor uses: empty, or [top]."""
    board = {cell: gs.board[cell][-1:] for cell in CELLS}
    return GridState(gs.t, board, gs.arm_clear, gs.hands_clear, dict(gs.confidence), gs.frame_id)


class ExtractError(ValueError):
    """The recording cannot give a lesson, for example no usable frame near its end."""


def settle(frames: Iterable[GridState], extraction: Extraction,
           on_settle: Optional[Callable[[Settled], None]] = None,
           on_stable: Optional[Callable[[int], None]] = None) -> Iterable[Settled]:
    """Run frames, reduced to occupancy and top, through the existing stabilizer. At the end, one
    more settled board (final=True) from the latest frame the stabilizer would accept, even if
    nothing changed, so the final read never comes from a frame a hand was still over. Raises
    ExtractError when no frame is usable, or none in the last FINAL_GAP_S seconds."""
    state = StabilizerState()
    unusable: Counter[str] = Counter()
    latest: Optional[tuple[GridState, GridState, Board, float]] = None  # (frame, raw, board, run start)
    last_emitted: Optional[int] = None
    last_usable_t: Optional[float] = None
    last_t = 0.0
    for raw in frames:
        gs = occupancy_and_top(raw)
        extraction.frames += 1
        why = unusable_reason(gs)
        if why is not None:
            unusable[why] += 1
        else:
            last_usable_t = gs.t
        last_t = gs.t
        # Would the stabilizer accept a board on this frame if it had not emitted one yet?
        probe_state, probe = stabilizer_step(replace(state, emitted=None), gs)
        state, board = stabilizer_step(state, gs)
        if probe is not None:
            latest = (gs, raw, probe, probe_state.since)
            if on_stable is not None:
                on_stable(gs.frame_id)
        if board is not None:
            s = Settled(t=state.since, t_settled=gs.t, frame_id=gs.frame_id, board=board,
                        camera_board=copy_board(raw.board), camera_confidence=dict(raw.confidence))
            last_emitted = gs.frame_id
            if on_settle is not None:
                on_settle(s)
            yield s
    extraction.unusable = dict(sorted(unusable.items()))
    if last_usable_t is None:
        raise ExtractError("no usable frame in the recording: something is over the grid the whole time")
    if last_t - last_usable_t > FINAL_GAP_S:
        raise ExtractError(f"no usable frame after {last_usable_t:.1f} s ({last_t - last_usable_t:.1f} s before the "
                           "end): something stays over the grid or its edge until the recording ends")
    if latest is not None and latest[0].frame_id != last_emitted:
        gs, raw, board, since = latest
        s = Settled(t=since, t_settled=gs.t, frame_id=gs.frame_id, board=board, camera_board=copy_board(raw.board),
                    camera_confidence=dict(raw.confidence), final=True)
        if on_settle is not None:
            on_settle(s)
        yield s


def extract(frames: Iterable[GridState], on_settle: Optional[Callable[[Settled], None]] = None,
            robot_read: Optional[RobotRead] = None, robot_distance: Optional[RobotDistance] = None,
            on_stable: Optional[Callable[[int], None]] = None,
            clear_limits: tuple[float, float] = (CLEAR_MAX_DISTANCE, CLEAR_MIN_MARGIN)) -> Extraction:
    """Frames -> the existing stabilizer -> track()."""
    ex = Extraction(settled=[], steps=[], removed=[], baseline=[])
    return track(settle(frames, ex, on_settle, on_stable), ex, robot_read, robot_distance, clear_limits)


def name_new(dist: dict[str, float], taken: set[str],
             clear_limits: tuple[float, float] = (CLEAR_MAX_DISTANCE, CLEAR_MIN_MARGIN)) -> Optional[str]:
    """Name a new cube from robot-memory distances: only colors not yet on the board are candidates,
    and the clear-read limits (nearest within the distance, ahead of the next by the margin) are
    measured against those only. None when no candidate is clear."""
    cands = sorted((d, c) for c, d in dist.items() if c not in taken and c in COLORS)
    if not cands:
        return None
    d1, c1 = cands[0]
    d2 = cands[1][0] if len(cands) > 1 else float("inf")
    return c1 if d1 <= clear_limits[0] and d2 - d1 >= clear_limits[1] else None


def track(settled: Iterable[Settled], ex: Optional[Extraction] = None,
          robot_read: Optional[RobotRead] = None, robot_distance: Optional[RobotDistance] = None,
          clear_limits: tuple[float, float] = (CLEAR_MAX_DISTANCE, CLEAR_MIN_MARGIN)) -> Extraction:
    """Steps from boards that have already settled (oldest first), by history: see the module
    docstring. Only occupancy and the top color of each cell are read from a board."""
    ex = ex if ex is not None else Extraction(settled=[], steps=[], removed=[], baseline=[])
    ex.robot_used = robot_read is not None
    stacks: dict[str, list[Cube]] = {cell: [] for cell in CELLS}

    fresh_color: dict[int, str] = {}  # cubes added on the board being read: their color before any read is kept

    def colors_on_board(other: Optional[Cube] = None) -> set[str]:
        return {c.color() if c.seen else fresh_color.get(id(c), UNKNOWN)
                for st in stacks.values() for c in st if c is not other} - {UNKNOWN}

    def remove(cube: Cube, t: float, gone: list[Cube]) -> None:
        cube.t_removed = t
        gone.append(cube)
        if not cube.baseline:
            ex.steps.remove(cube)
            ex.removed.append(cube)

    last_occupied: Optional[set[str]] = None
    for s in settled:
        occupied = [cell for cell in CELLS if s.board[cell]]
        if last_occupied is not None:  # backstop: too many cells changed at once to be one move
            changed = sorted(set(occupied) ^ last_occupied)
            if len(changed) >= BACKSTOP_CELLS:
                ex.skipped.append({"t": round(s.t, 3), "frame_id": s.frame_id, "final": s.final, "cells": changed})
                continue
        last_occupied = set(occupied)
        first = not ex.settled
        ex.settled.append(s)
        camera = {cell: s.board[cell][-1] for cell in occupied}
        robot = {cell: robot_read(s.frame_id, cell) if robot_read is not None else None for cell in occupied}
        tops = {cell: robot[cell] if robot[cell] not in (None, UNKNOWN) else camera[cell] for cell in occupied}
        dist = {cell: d for cell in occupied if robot_distance is not None and (d := robot_distance(s.frame_id, cell))}
        ex.tops.append(tops)
        ex.distances.append(dist)
        seen = dict(tops)       # the read each top cube keeps from this board
        added: list[Cube] = []
        gone: list[Cube] = []
        skip: set[str] = set()  # cells whose read on this board was refused (not kept on the top cube)
        refused: set[str] = set()  # cells whose read conflicted: kept on the top cube as unknown
        named_now: dict[str, str] = {}  # cells named among colors not on the board (robot-memory distances)
        overridden: set[str] = set()    # cells whose read named a color on the board and was replaced by named_now
        fresh_color.clear()

        def add(cell: str, color: str) -> None:
            cube = Cube(cell, len(stacks[cell]), s.t, s.frame_id, seq=len(ex.steps) + len(ex.removed) + len(ex.baseline))
            fresh_color[id(cube)] = color
            stacks[cell].append(cube)
            ex.steps.append(cube)
            added.append(cube)
            seen[cell] = color

        if first:
            for cell in occupied:
                cube = Cube(cell, 0, s.t, s.frame_id, baseline=True)
                stacks[cell].append(cube)
                ex.baseline.append(cube)
        else:
            for cell in CELLS:  # 1. removals
                st = stacks[cell]
                if st and cell not in camera:
                    if len(st) > 1:  # judged in gate_history, once transients are known
                        ex.emptied.append((s.t, cell, list(st)))
                    for cube in reversed(st):
                        remove(cube, s.t, gone)
                    st.clear()
                elif len(st) >= 2 and tops[cell] != UNKNOWN and tops[cell] == st[-2].color() != st[-1].color():
                    remove(st.pop(), s.t, gone)
            fresh = [cell for cell in occupied if not stacks[cell]]
            for cell in fresh:  # 2. a cube on an empty cell: the strongest evidence
                read, taken = tops[cell], colors_on_board()
                if (read == UNKNOWN or read in taken) and (named := name_new(dist.get(cell, {}), taken, clear_limits)):
                    if read != UNKNOWN:
                        overridden.add(cell)
                    read = named_now[cell] = named  # only colors not yet on the board are candidates
                if read != UNKNOWN and read in colors_on_board():
                    ex.problems.append(("one_cube_per_color", f"{read} appeared on {cell} at {s.t:.1f} s while "
                                                              f"a {read} cube was still on the board"))
                add(cell, read)
            for cell in (c for c in occupied if c not in fresh):  # 3. what an occupied cell's top reads
                st, read = stacks[cell], tops[cell]
                top, color, d = st[-1], st[-1].color(), dist.get(cell, {})
                if read != UNKNOWN:
                    if read == color:
                        continue
                    if read in colors_on_board(top):  # a color already on the board: keep the old one
                        ex.conflicts.append({"t": round(s.t, 3), "cell": cell, "level": top.level, "kept": color,
                                             "read": read})
                        refused.add(cell)
                    elif color == UNKNOWN:
                        continue  # the top's first clear read: the same cube
                    elif len(st) > MAX_LEVEL:
                        ex.problems.append(("levels_match_stack_height", f"a fourth cube ({read}) on {cell} at "
                                                                         f"{s.t:.1f} s"))
                        skip.add(cell)
                    else:
                        add(cell, read)
                    continue
                named = name_new(d, colors_on_board(), clear_limits)
                if color == UNKNOWN:
                    if named:
                        seen[cell] = named_now[cell] = named  # the same unread cube, now named
                elif named is not None and named != color and (color not in d or d[named] < d[color]):
                    if len(st) > MAX_LEVEL:
                        ex.problems.append(("levels_match_stack_height", f"a fourth cube ({named}) on {cell} at "
                                                                         f"{s.t:.1f} s"))
                        skip.add(cell)
                    else:
                        add(cell, named)  # nearer a free color than its own: a new cube
                        named_now[cell] = named
                elif color in d and d[color] <= WEAK_READ_MAX_DISTANCE:
                    ex.weak_reads.append({"t": round(s.t, 3), "cell": cell, "level": top.level, "kept": color,
                                          "distance": round(d[color], 1)})  # a weaker read of the same cube
                elif len(st) <= MAX_LEVEL:
                    add(cell, UNKNOWN)  # unclear and not close: something else on it
            ex.changes.append((s.t, added, gone))
        # The top reads the gate sees: free-color names in, conflicting reads out (not clear reads of that cell).
        ex.tops[-1] = {**tops, **named_now, **{cell: UNKNOWN for cell in refused}}
        for cell in occupied:  # the read of each top cube on this board
            if cell in skip or not stacks[cell]:
                continue
            top = stacks[cell][-1]
            blank = cell in refused or cell in overridden  # that read is not evidence for this cube's color
            top.reads.append(UNKNOWN if blank else camera[cell])
            top.robot.append(UNKNOWN if blank and robot[cell] is not None else robot[cell])
            top.seen.append(UNKNOWN if cell in refused else seen[cell])
            if cell in named_now:
                top.named.append(named_now[cell])
            top.settles.append(s.frame_id)
    return ex


# ---------------------------------------------------------------- colors and the lesson (pure)


class ColorsFileError(ValueError):
    """A --colors-file that does not fit the extracted steps."""


@dataclass
class StepRow:
    number: int                  # 1-based, as printed and as the colors file uses
    cell: str
    level: int
    t_video: float
    color: str                   # a color from COLORS, or "unknown"
    source: str                  # "manual", "robot_memory", "camera", "gemini", or "none" (unknown)
    camera_color: str
    reads: dict[str, int]
    frame_id: int
    robot_color: Optional[str] = None             # None without a color memory
    robot_reads: Optional[dict[str, int]] = None
    manual_color: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        conflict = len([c for c in self.reads if c != UNKNOWN]) > 1
        return {"step": self.number, "cell": self.cell, "level": self.level, "t_video": round(self.t_video, 3),
                "color": self.color, "color_source": self.source,
                "camera_color": self.camera_color, "camera_reads": self.reads, "camera_reads_disagree": conflict,
                "robot_memory_color": self.robot_color, "robot_memory_reads": self.robot_reads,
                "manual_color": self.manual_color, "frame_id": self.frame_id}


def choose_color(camera: str, robot: Optional[str], manual: Optional[str]) -> tuple[str, str]:
    """The one place a step's color is decided: (color, source). Trust order: the manual colors
    file, then the robot's color memory, then the camera; the first known color wins. A source
    that does not know ("unknown", or None when not in use) passes to the next. Never a guess."""
    if manual is not None:
        return manual, "manual"
    if robot is not None and robot != UNKNOWN:
        return robot, "robot_memory"
    if camera != UNKNOWN:
        return camera, "camera"
    return UNKNOWN, "none"


def parse_colors_file(data: Any) -> list[dict[str, Any]]:
    """Shape check only (runs before extraction, so a typo fails fast)."""
    if not isinstance(data, list):
        raise ColorsFileError('colors file: expected a list like [{"step": 2, "cell": "B2", "color": "orange"}]')
    for i, e in enumerate(data):
        if not isinstance(e, dict) or not {"step", "cell", "color"} <= set(e):
            raise ColorsFileError(f"colors file entry {i + 1}: needs step, cell and color")
        if not isinstance(e["step"], int) or isinstance(e["step"], bool):
            raise ColorsFileError(f"colors file entry {i + 1}: step must be a whole number")
        if e["color"] in ("", None):
            raise ColorsFileError(f"colors file: step {e['step']} has a blank color; fill it in or remove the line")
        if e["color"] not in COLORS:
            raise ColorsFileError(f"colors file: step {e['step']} color {e['color']!r} is not one of {', '.join(COLORS)}")
    return data


def manual_colors(entries: list[dict[str, Any]], cubes: list[Cube]) -> dict[int, str]:
    """{step number: color} from colors-file entries, checked against the extracted steps."""
    out: dict[int, str] = {}
    for e in parse_colors_file(entries):
        n = e["step"]
        if not 1 <= n <= len(cubes):
            raise ColorsFileError(f"colors file: there is no step {n} (the lesson has {len(cubes)} steps)")
        cell = cubes[n - 1].cell
        if str(e["cell"]).upper() != cell:
            raise ColorsFileError(f"colors file: step {n} is on {cell}, but the file says {e['cell']}. "
                                  "Check the step table and fix the file.")
        if n in out:
            raise ColorsFileError(f"colors file: step {n} appears twice")
        out[n] = e["color"]
    return out


def split_transients(ex: Extraction) -> None:
    """A transient appeared in a slot, was removed before the end of the recording, and never had
    a known color from any source while it was there (a hand or a sleeve read as a block). It is
    not a cube: no step, no correction, and the gate does not count it (gate_history). Anything
    that ever had a known color, or is still on the board at the end, stays a cube."""
    kept = []
    for cube in ex.removed:
        known = [r for r in cube.reads + cube.robot_reads() + cube.named if r != UNKNOWN]
        (kept if known else ex.transients).append(cube)
    ex.removed = kept


def restore_hidden(ex: Extraction) -> None:
    """Undo removals an arm faked: a removed cube whose color is the color of the next cube to appear
    in the same slot (cell and level) was hidden, not removed. It keeps its step and time and takes on
    the later cube's reads; the later cube is dropped. Run after split_transients."""
    transient = {id(c) for c in ex.transients}
    for r in sorted(ex.removed, key=lambda c: c.seq):
        if r.t_removed is None or r.color() == UNKNOWN:
            continue
        later = [c for c in ex.steps + ex.removed if c is not r and id(c) not in transient
                 and (c.cell, c.level) == (r.cell, r.level) and c.t_video >= r.t_removed]
        back = min(later, key=lambda c: c.seq, default=None)
        if back is None or back.color() != r.color():
            continue
        ex.hidden.append({"cell": r.cell, "level": r.level, "color": r.color(), "t_gone": round(r.t_removed, 3),
                          "t_back": round(back.t_video, 3)})
        for field_name in ("reads", "robot", "seen", "named", "settles"):
            getattr(r, field_name).extend(getattr(back, field_name))
        ex.removed.remove(r)
        r.t_removed = back.t_removed
        if back in ex.steps:
            ex.steps[ex.steps.index(back)] = r
            ex.steps.sort(key=lambda c: c.seq)
        else:
            ex.removed[ex.removed.index(back)] = r
        for _, added, gone in ex.changes:  # neither the hiding nor the coming back is a change
            gone[:] = [r if c is back else c for c in gone if c is not r]  # a later removal of back is r's
            added[:] = [c for c in added if c is not back]


def gate_history(ex: Extraction) -> History:
    """What the gate checks, from the history map: transients left out of every change."""
    transient = {id(c) for c in ex.transients}

    def slots(cubes: list[Cube]) -> list[tuple[str, int]]:
        return [(c.cell, c.level) for c in cubes if id(c) not in transient]

    problems = list(ex.problems)
    for t, cell, cubes in ex.emptied:
        n = sum(1 for c in cubes if id(c) not in transient)
        if n > 1:
            problems.append(("one_cube_per_change", f"{cell} went from {n} cubes to empty in one change at {t:.1f} s"))
    if not ex.settled:
        return History(None, None, problems=problems)
    first, last, tops = ex.settled[0], ex.settled[-1], ex.tops[-1]
    return History(first_occupied=[cell for cell in CELLS if first.board[cell]],
                   final_tops={cell: tops.get(cell) if last.board[cell] else None for cell in CELLS},
                   final_distances=dict(ex.distances[-1]), weak_limit=WEAK_READ_MAX_DISTANCE,
                   changes=[Change(t, slots(added), slots(gone), slots([c for c in added if c.t_removed is not None]))
                            for t, added, gone in ex.changes],
                   problems=problems)


def step_rows(ex: Extraction, manual: Optional[dict[int, str]] = None) -> list[StepRow]:
    """One row per step. Robot-memory reads (taken while the cube was the top of its cell) follow
    the same agreement rule as the camera's; None when no color memory was in use."""
    manual = manual or {}
    rows = []
    for i, cube in enumerate(ex.steps):
        robot_color = robot_counts = None
        if ex.robot_used:  # the namer's reads, plus names among colors not on the board
            reads = cube.robot_reads()
            robot_color = agreed_color(reads + cube.named)
            robot_counts = dict(sorted(Counter(reads + [f"{n} (named among free colors)" for n in cube.named]).items()))
        color, source = choose_color(cube.camera_color(), robot_color, manual.get(i + 1))
        rows.append(StepRow(i + 1, cube.cell, cube.level, cube.t_video, color, source, cube.camera_color(),
                            cube.read_counts(), cube.frame_id, robot_color, robot_counts, manual.get(i + 1)))
    return rows


GeminiAsk = Callable[[bytes], dict]  # crop (jpeg) -> {"answer": a COLORS name or "none", "model", "raw"}


def clearest_frame(cube: Cube, ex: Extraction) -> Optional[int]:
    """The final read when the cube is on top there, else the settled board where it was on top and
    the camera was most sure of its cell."""
    if ex.settled and ex.settled[-1].final and ex.settled[-1].frame_id in cube.settles:
        return ex.settled[-1].frame_id
    conf = {s.frame_id: (s.camera_confidence or {}).get(cube.cell, 0.0) for s in ex.settled}
    return max(cube.settles, key=lambda fid: (conf.get(fid, 0.0), -fid)) if cube.settles else None


def gemini_fallback(rows: list[StepRow], ex: Extraction, crop: Callable[[int, str], Optional[bytes]],
                    ask: Optional[GeminiAsk], saved: dict[str, dict]) -> list[dict[str, Any]]:
    """The last color source, for steps still unknown (rows are ex.steps, so removed cubes and
    transients are never asked about). One record per step considered; accepted answers set the
    row's color with source "gemini". saved: earlier answers by "cell@frame", reused without a call."""
    records = []
    for row, cube in zip(rows, ex.steps):
        if row.color != UNKNOWN:
            continue
        fid = clearest_frame(cube, ex)
        if fid is None:
            continue
        key = f"{cube.cell}@{fid}"
        rec: dict[str, Any] = {"step": row.number, "cell": cube.cell, "frame_id": fid, "key": key}
        if key in saved:
            reply, rec["reused"] = saved[key], True
        else:
            jpeg = crop(fid, cube.cell)
            if ask is None or jpeg is None:
                records.append({**rec, "accepted": False, "reason": "not asked (no key or no crop)"})
                continue
            try:
                reply, rec["reused"] = ask(jpeg), False
            except GeminiUnavailable as e:
                records.append({**rec, "accepted": False, "reason": f"unavailable: {e}"})
                continue
        answer = reply.get("answer", "none")
        rec.update(answer=answer, model=reply.get("model"), raw=reply.get("raw"))
        taken = {r.color for r in rows if r is not row and r.color in COLORS}
        if answer not in COLORS:
            rec.update(accepted=False, reason="none")
        elif answer in taken:
            rec.update(accepted=False, reason=f"{answer} is already on the board")
        else:
            rec.update(accepted=True, reason="")
            row.color, row.source = answer, "gemini"
        records.append(rec)
    return records


def saved_gemini(lessons_dir: Path, source_sha256: Optional[str]) -> dict[str, dict]:
    """Gemini answers from earlier extract records of the same source file, by "cell@frame"."""
    saved: dict[str, dict] = {}
    folder = lessons_dir / "extract"
    if not source_sha256 or not folder.is_dir():
        return saved
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (record.get("source") or {}).get("sha256") != source_sha256:
            continue
        for g in record.get("gemini", []):
            if "answer" in g:
                saved[g["key"]] = {"answer": g["answer"], "model": g.get("model"), "raw": g.get("raw")}
    return saved


def build_lesson(rows: list[StepRow], lesson_id: str, title: str, source_video: Optional[str]) -> Lesson:
    """The lesson the steps describe. final_board is the replay of the steps (the board a student
    builds), which the gate compares with the recording's last settled board."""
    steps = [Step(r.color, r.cell, r.level, round(r.t_video, 3), None) for r in rows]
    final = empty_board()
    for s in steps:
        final[s.cell].append(s.color)
    return Lesson(lesson_id, title, source_video, steps, final)


def gate_lesson(lesson: Lesson, ex: Extraction, reference_empty: Optional[bool] = None,
                rows: Optional[list[StepRow]] = None) -> GateResult:
    """rows: the step rows, so a manual or Gemini color can stand in for an unreadable final read."""
    unverified = {r.number - 1: r.source for r in rows or [] if r.source in ("manual", "gemini")}
    result = run_gate(lesson.steps, gate_history(ex), reference_empty, unverified)
    lesson.gate = result.gate
    return result


def colors_template(rows: list[StepRow]) -> list[dict[str, Any]]:
    """A colors file listing every step without a known color, color left blank."""
    return [{"step": r.number, "cell": r.cell, "color": ""} for r in rows if r.color not in COLORS]


def template_text(template: list[dict[str, Any]]) -> str:
    return "[\n" + ",\n".join("  " + json.dumps(e) for e in template) + "\n]"


# ---------------------------------------------------------------- ids and files


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "lesson"


def taken_ids(lessons_dir: Path) -> set[str]:
    taken = set()
    for d in (lessons_dir, lessons_dir / "extract"):
        for path in d.glob("*.json") if d.is_dir() else []:
            taken.add(path.name.split(".")[0])
            try:
                taken.add(str(json.loads(path.read_text(encoding="utf-8")).get("lesson_id")))
            except (ValueError, OSError, AttributeError):
                pass
    return taken


def new_lesson_id(name: str, lessons_dir: Path, now: Optional[datetime] = None) -> str:
    """<name>-<YYYYMMDD-HHMMSS>, with -2, -3, ... if a lesson or extract file already uses it."""
    base = f"{slug(name)}-{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}"
    taken = taken_ids(lessons_dir)
    n = 1
    while (base if n == 1 else f"{base}-{n}") in taken:
        n += 1
    return base if n == 1 else f"{base}-{n}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def boards_json(ex: Extraction) -> list[dict[str, Any]]:
    """The settled boards the history used (occupancy and top, plus the clear top read), with the
    camera's own stacks as evidence only."""
    return [{"t": round(s.t, 3), "t_settled": round(s.t_settled, 3), "frame_id": s.frame_id,
             "board": copy_board(s.board), "tops": dict(tops),
             "camera_board": None if s.camera_board is None else copy_board(s.camera_board)}
            for s, tops in zip(ex.settled, ex.tops)]


def cube_dict(c: Cube, ex: Extraction) -> dict[str, Any]:
    """A removed or baseline cube, with what each source read on every board it was the top."""
    times = {s.frame_id: s.t for s in ex.settled}
    seen = [{"t": round(times[fid], 3), "frame_id": fid, "camera": cam, "robot_memory": rob}
            for fid, cam, rob in zip(c.settles, c.reads, c.robot)]
    return {"cell": c.cell, "level": c.level, "t_appeared": round(c.t_video, 3),
            "t_removed": None if c.t_removed is None else round(c.t_removed, 3),
            "color": c.color(), "camera_color": c.camera_color(), "seen": seen}


@dataclass
class Outcome:
    lesson: Lesson
    rows: list[StepRow]
    gate: GateResult
    lesson_path: Optional[Path]       # None when a color is unknown (draft only)
    extract_path: Path
    exit_code: int


def write_outputs(ex: Extraction, rows: list[StepRow], lesson: Lesson, gate: GateResult, out: Path,
                  info: dict[str, Any], boards_out: Optional[Path] = None,
                  pictures: Optional[Callable[[Path], list[str]]] = None,
                  gemini: Optional[list[dict[str, Any]]] = None) -> Outcome:
    """Write the lesson (only when every color is known), the extract file, and --boards-out."""
    extract_dir = out.parent / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_path = extract_dir / f"{lesson.lesson_id}.json"
    complete = all(r.color in COLORS for r in rows)
    lesson_path = None
    if complete:
        data = lesson.to_dict()
        Lesson.from_dict(data)  # a lesson that would not load is a bug here: fail loudly
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        lesson_path = out
    record: dict[str, Any] = {
        "lesson_id": lesson.lesson_id,
        "title": lesson.title,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "lesson" if complete else "draft: unknown colors",
        "lesson_file": None if lesson_path is None else repo_relative(lesson_path),
        **info,
        "git": git_info(),
        "frames": {"total": ex.frames, "unusable": ex.unusable, **info.get("frames", {})},
        "settled_boards": len(ex.settled),
        "settled_boards_with_unknown": ex.boards_with_unknown,
        "steps": [r.to_dict() for r in rows],
        "corrections": [cube_dict(c, ex) for c in ex.removed],
        "baseline_cubes": [cube_dict(c, ex) for c in ex.baseline],
        "transients": [cube_dict(c, ex) for c in ex.transients],
        "weak_reads": ex.weak_reads,
        "hidden": ex.hidden,
        "conflicting_reads": ex.conflicts,
        "skipped_boards": ex.skipped,
        "final_read": None if not ex.settled or not ex.settled[-1].final else
        {"t": round(ex.settled[-1].t_settled, 3), "frame_id": ex.settled[-1].frame_id},
        "gemini": gemini or [],
        "gate": {**gate.gate.to_dict(), "reasons": gate.reasons, "notes": gate.notes},
        "pictures": [repo_relative(Path(p)) for p in pictures(extract_dir / lesson.lesson_id)] if pictures else [],
    }
    if not complete:
        record["colors_file_template"] = colors_template(rows)
        record["draft_lesson"] = lesson.to_dict()
    with open(extract_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    if boards_out is not None:
        boards_out.parent.mkdir(parents=True, exist_ok=True)
        boards_out.write_text(json.dumps(boards_json(ex), indent=1) + "\n", encoding="utf-8")
    code = EXIT_DRAFT if not complete else EXIT_PASSED if gate.gate.passed else EXIT_GATE_FAILED
    return Outcome(lesson, rows, gate, lesson_path, extract_path, code)


# ---------------------------------------------------------------- sources


def script_frames(path: Path, seed: int = 0) -> list[GridState]:
    """The frames brain.fake_feed would publish for a script, with the script's own clock
    (seconds since the start) instead of the computer's."""
    import random

    from brain.fake_feed import HZ, make_frame, parse_script

    rng = random.Random(seed)
    frames: list[GridState] = []
    for seg in parse_script(json.loads(Path(path).read_text(encoding="utf-8"))):
        for _ in range(round(seg.duration_s * HZ)):
            n = len(frames)
            frames.append(make_frame(seg.board, seg.hands_clear, seg.arm_clear, seg.jitter, n, rng, t=n / HZ))
    return frames


def ws_frames(url: str, seconds: Optional[float], log: Callable[[str], None] = print) -> list[GridState]:
    """Every GridState from a live feed until Ctrl+C, the feed closes, or `seconds` pass.
    t becomes seconds since the first frame."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    frames: list[GridState] = []
    t0 = None
    started = time.monotonic()
    try:
        with connect(url) as ws:
            log(f"[extract] recording {url}; press Ctrl+C to end the take")
            while seconds is None or time.monotonic() - started < seconds:
                try:
                    msg = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                gs = GridState.from_dict(json.loads(msg))
                t0 = gs.t if t0 is None else t0
                gs.t -= t0
                frames.append(gs)
    except KeyboardInterrupt:
        log("[extract] take ended")
    except ConnectionClosed:
        log("[extract] feed closed")
    return frames


# ---------------------------------------------------------------- command line


def print_report(out: Outcome, ex: Extraction) -> None:
    print(f"\n{'step':>4}  {'cell':<4}  {'level':>5}  {'color':<8}  {'source':<12}  {'video s':>7}")
    for r in out.rows:
        print(f"{r.number:>4}  {r.cell:<4}  {r.level:>5}  {r.color:<8}  {r.source:<12}  {r.t_video:>7.2f}")
    if not out.rows:
        print("   (no steps)")
    for c in ex.removed:
        print(f"  correction: {c.color()} cube on {c.cell} level {c.level}, appeared {c.t_video:.2f} s, "
              f"removed {c.t_removed:.2f} s")
    for h in ex.hidden:
        print(f"  hidden: {h['color']} cube on {h['cell']} level {h['level']} gone at {h['t_gone']:.2f} s, back at "
              f"{h['t_back']:.2f} s (an arm hid it; step kept)")
    for c in ex.conflicts:
        print(f"  conflicting read: {c['cell']} level {c['level']} at {c['t']:.2f} s read {c['read']}, already on the "
              f"board; kept {c['kept']}")
    for k in ex.skipped:
        print(f"  skipped board at {k['t']:.2f} s{' (final read)' if k['final'] else ''}: "
              f"{len(k['cells'])} cells changed at once ({', '.join(k['cells'])})")
    for w in ex.weak_reads:
        print(f"  weak read: {w['cell']} level {w['level']} at {w['t']:.2f} s kept {w['kept']} "
              f"(robot-memory distance {w['distance']}, limit {WEAK_READ_MAX_DISTANCE:g})")
    for c in ex.transients:
        print(f"  WARNING transient (not a cube, no known color from any source): {c.cell} level {c.level}, "
              f"{c.t_video:.2f} s to {c.t_removed:.2f} s")
    disagree = [r for r in out.rows if len([c for c in r.reads if c != UNKNOWN]) > 1]
    for r in disagree:
        print(f"  step {r.number}: camera reads disagree {r.reads}")
    usable = ex.frames - sum(ex.unusable.values())
    final = " (last one is the final read)" if ex.settled and ex.settled[-1].final else ""
    print(f"\nframes: {ex.frames} read, {usable} usable; settled boards: {len(ex.settled)}{final}, "
          f"{len(ex.skipped)} skipped ({ex.boards_with_unknown} with an unknown cell)")
    for why, n in ex.unusable.items():
        print(f"  unusable: {n} frames, {why}")
    g = out.gate
    print(f"\ngate: {'PASSED' if g.gate.passed else 'FAILED'}")
    for c in g.gate.checks:
        print(f"  {'pass' if c.passed else 'FAIL'}  {c.name}" + (f": {g.reasons[c.name]}" if c.name in g.reasons else ""))
        for note in g.notes.get(c.name, []):
            print(f"        note: {note}")
    print(f"\nextract file: {repo_relative(out.extract_path)}")
    if out.lesson_path is not None:
        print(f"lesson file:  {repo_relative(out.lesson_path)}")
        if not g.gate.passed:
            print("\n!!! GATE FAILED: the lesson is written but marked failed (gate.passed = false). "
                  "Do not teach from it until the reasons above are fixed. !!!")
    else:
        print("\n!!! NO LESSON WRITTEN: some steps have no known color. The draft is in the extract file. !!!")
        print("Save this as a .json file, fill in each blank color "
              f"({', '.join(COLORS)}), then rerun with --colors-file <file>:")
        print(template_text(colors_template(out.rows)))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="recorded video; perception runs on it in this process")
    src.add_argument("--script", type=Path, help="a tests/scripts/*.json fake feed script")
    src.add_argument("--grid", metavar="URL", help=f"a live GridState feed, e.g. ws://localhost:{GRID_WS_PORT}")
    p.add_argument("--out", type=Path, help="lesson file (default lessons/<lesson_id>.json); never overwritten")
    p.add_argument("--title", help="lesson title (default from the source file name)")
    p.add_argument("--colors-file", type=Path, help='per-step colors: [{"step": 2, "cell": "B2", "color": "orange"}]')
    p.add_argument("--boards-out", type=Path, help="write the settled boards with their times (what the gate checks)")
    p.add_argument("--source-video", help="--script/--grid: video path to store in the lesson (for show-me clips)")
    p.add_argument("--seed", type=int, default=0, help="--script: jitter seed (default 0)")
    p.add_argument("--seconds", type=float, help="--grid: stop after this many seconds (default: Ctrl+C)")
    p.add_argument("--no-gemini", action="store_true", help="--video: never ask Gemini for a color")
    v = p.add_argument_group("--video only")
    v.add_argument("--calib", type=Path, help="folder with grid_lock.json, empty_board.png, colors.json "
                                              "(default: grid lock and empty board from the video's first second)")
    v.add_argument("--colors", help=f"comma-separated colors in play (default: all shared colors, {','.join(COLORS)})")
    v.add_argument("--block-mm", type=float, help=f"block edge in mm (default {DEFAULT_BLOCK_MM:g})")
    v.add_argument("--color-memory", type=Path, help="robot color memory: name colors as the robot does "
                                                     "(default robot/outputs/color_memory.json if it exists)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    started = time.monotonic()
    try:
        return run(argv)
    finally:
        print(f"[extract] run time {time.monotonic() - started:.1f} s")


def run(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    if not args.video and (args.calib or args.colors or args.block_mm or args.color_memory):
        print("[extract] --calib, --colors, --block-mm and --color-memory only apply to --video", file=sys.stderr)
        return EXIT_BAD_INPUT
    try:
        entries = parse_colors_file(json.loads(args.colors_file.read_text(encoding="utf-8"))) \
            if args.colors_file else []
    except (OSError, ValueError) as e:
        print(f"[extract] {e}", file=sys.stderr)
        return EXIT_BAD_INPUT

    info: dict[str, Any] = {}
    video = None
    source_errors: tuple[type[Exception], ...] = ()
    source_video: Optional[str] = args.source_video
    if args.video:
        from brain.video_source import VideoSource, VideoSourceError  # loads perception, --video only
        source_errors = (VideoSourceError,)
        try:
            video = VideoSource(args.video, args.colors or ",".join(COLORS), args.block_mm or DEFAULT_BLOCK_MM,
                                args.calib, args.color_memory)
        except (VideoSourceError, ValueError) as e:
            print(f"[extract] {e}", file=sys.stderr)
            return EXIT_BAD_INPUT
        name, source_video = args.video.stem, repo_relative(args.video)
        if not source_video.startswith("videos/"):
            print(f"[extract] warning: {source_video} is not under videos/, so the UI cannot play show-me clips from it")
        frames: Iterable[GridState] = video.frames()
    elif args.script:
        name, frames = args.script.stem, script_frames(args.script, args.seed)
        info["source"] = {"kind": "script", "path": repo_relative(args.script), "sha256": file_sha256(args.script),
                          "seed": args.seed}
    else:
        print("[extract] t_video is seconds since the first frame received. That matches a live camera; "
              "with perception.publish --video the times are wrong (it plays slower than real time).")
        name, frames = "grid", ws_frames(args.grid, args.seconds)
        info["source"] = {"kind": "grid", "url": args.grid}
    if not args.video:
        info["calibration"] = {"mode": None, "colors_in_play": None,
                               "note": "colors and calibration were decided by the feed, not recorded here"}

    try:
        return finish(args, entries, name, frames, info, source_video, video)
    except source_errors + (ExtractError,) as e:
        print(f"[extract] {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    finally:
        if video is not None:
            video.close()


def ask_gemini(args: argparse.Namespace, rows: list[StepRow], ex: Extraction, video: Any,
               lessons_dir: Path) -> list[dict[str, Any]]:
    """The Gemini fallback for --video runs; prints how many calls it made."""
    unknown = [r for r in rows if r.color == UNKNOWN]
    if video is None or args.no_gemini or not unknown:
        if video is not None:
            print(f"[extract] gemini: 0 calls ({'off' if args.no_gemini else 'no step needs it'})")
        return []
    from brain import gemini_colors

    key = gemini_colors.api_key()
    calls = 0

    def ask(jpeg: bytes) -> dict:
        nonlocal calls
        calls += 1
        return gemini_colors.ask_color(jpeg, key)

    if key is None:
        print("[extract] WARNING: no GEMINI_API_KEY set, so the Gemini color fallback only reuses saved answers")
    saved = saved_gemini(lessons_dir, (video.info.get("source") or {}).get("sha256"))
    records = gemini_fallback(rows, ex, video.crop_jpeg, ask if key else None, saved)
    print(f"[extract] gemini: {calls} call(s), {sum(1 for r in records if r.get('reused'))} saved answer(s) reused")
    models = sorted({r["model"] for r in records if r.get("model")})
    if models:
        print(f"[extract] gemini model: {', '.join(models)}")
    for r in records:
        print(f"  gemini step {r['step']} {r['cell']}: {r.get('answer', '-')} "
              + ("accepted (unverified)" if r["accepted"] else f"rejected: {r['reason']}"))
    return records


def finish(args: argparse.Namespace, entries: list[dict[str, Any]], name: str, frames: Iterable[GridState],
           info: dict[str, Any], source_video: Optional[str], video: Any) -> int:
    robot_read = video.robot_read if video is not None and video.namer is not None else None
    ex = extract(frames, video.on_settle if video is not None else None, robot_read,
                 video.robot_distance if robot_read is not None else None,
                 video.on_stable if video is not None else None,
                 video.clear_limits if video is not None else (CLEAR_MAX_DISTANCE, CLEAR_MIN_MARGIN))
    if video is not None:
        print(f"[extract] hand band: {video.hand_frames} frame(s) perception called clear were thrown out "
              f"(band changed more than {video.band_max:g})")
    if video is not None:
        info.update(video.info)
    split_transients(ex)
    restore_hidden(ex)
    try:
        manual = manual_colors(entries, ex.steps)
    except ColorsFileError as e:
        print(f"[extract] {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    lessons_dir = args.out.parent if args.out else LESSONS_DIR
    lesson_id = new_lesson_id(name, lessons_dir)
    out = args.out or lessons_dir / f"{lesson_id}.json"
    if out.exists():
        print(f"[extract] {out} already exists; pick another --out (lessons are never overwritten)", file=sys.stderr)
        return EXIT_BAD_INPUT
    rows = step_rows(ex, manual)
    gemini = ask_gemini(args, rows, ex, video, lessons_dir)
    lesson = build_lesson(rows, lesson_id, args.title or name.replace("_", " ").replace("-", " ").title(), source_video)
    gate = gate_lesson(lesson, ex, video.reference_empty if video is not None else None, rows)
    pictures = None
    if video is not None:
        pictures = lambda folder: video.save_pictures(folder, [(r.number, r.cell, r.frame_id) for r in rows])  # noqa: E731
    outcome = write_outputs(ex, rows, lesson, gate, out, info, args.boards_out, pictures, gemini)
    print_report(outcome, ex)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
