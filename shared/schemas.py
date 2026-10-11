"""Contract types shared by perception, brain, robot, and ui (README section 5).

Changing anything here requires the whole team to agree (see AGENTS.md).
Stdlib only. Every type has to_dict() / from_dict(); from_dict raises
SchemaError on contract violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from shared.config import ALL_COLORS, CELLS, COLORS, LEVELS, MAX_LEVEL

# Cell name -> stack of colors, bottom to top. All 9 cells always present.
Board = dict[str, list[str]]


class SchemaError(ValueError):
    """Raised when data violates a contract in this module."""


class Verdict(str, Enum):
    PASS = "PASS"
    WRONG_CELL = "WRONG_CELL"
    WRONG_COLOR = "WRONG_COLOR"
    WRONG_LEVEL = "WRONG_LEVEL"
    EXTRA = "EXTRA"
    MISSING = "MISSING"
    WAITING = "WAITING"


# ---------------------------------------------------------------- helpers


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected an object, got {type(d).__name__}")
    if key not in d:
        raise SchemaError(f"{where}: missing field '{key}'")
    return d[key]


def _check_cell(cell: Any, where: str) -> str:
    if cell not in CELLS:
        raise SchemaError(f"{where}: invalid cell {cell!r}")
    return cell


def validate_board(board: Any, allow_unknown: bool = True) -> Board:
    """Raise SchemaError unless board has exactly the 9 cells, valid colors,
    and no stack taller than MAX_LEVEL + 1. Returns the board unchanged."""
    if not isinstance(board, dict):
        raise SchemaError(f"board: expected an object, got {type(board).__name__}")
    missing = [c for c in CELLS if c not in board]
    if missing:
        raise SchemaError(f"board: missing cells {missing}")
    extra = [c for c in board if c not in CELLS]
    if extra:
        raise SchemaError(f"board: unknown cells {extra}")
    allowed = ALL_COLORS if allow_unknown else COLORS
    for cell in CELLS:
        stack = board[cell]
        if not isinstance(stack, list):
            raise SchemaError(f"board[{cell}]: expected a list, got {type(stack).__name__}")
        if len(stack) > MAX_LEVEL + 1:
            raise SchemaError(f"board[{cell}]: stack of {len(stack)} exceeds {MAX_LEVEL + 1}")
        for color in stack:
            if color not in allowed:
                raise SchemaError(f"board[{cell}]: invalid color {color!r}")
    return board


def empty_board() -> Board:
    return {cell: [] for cell in CELLS}


def copy_board(board: Board) -> Board:
    return {cell: list(board[cell]) for cell in CELLS}


def boards_equal(a: Board, b: Board) -> bool:
    return all(list(a.get(cell, [])) == list(b.get(cell, [])) for cell in CELLS)


def apply_step(board: Board, step: Step) -> Board:
    """Return a new board with step's cube placed. Does not mutate board.

    Raises SchemaError if step.level is not the current height of the stack
    (a floating cube or a cube placed into an occupied level).
    """
    height = len(board[step.cell])
    if step.level != height:
        raise SchemaError(
            f"apply_step: {step.cell} has height {height}, cannot place at level {step.level}"
        )
    new = copy_board(board)
    new[step.cell].append(step.color)
    return new


# ---------------------------------------------------------------- 5.1 GridState


@dataclass
class GridState:
    t: float
    board: Board
    arm_clear: bool
    hands_clear: bool
    confidence: dict[str, float]
    frame_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "board": copy_board(self.board),
            "arm_clear": self.arm_clear,
            "hands_clear": self.hands_clear,
            "confidence": dict(self.confidence),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridState:
        w = "GridState"
        board = validate_board(_req(d, "board", w))
        confidence = _req(d, "confidence", w)
        if not isinstance(confidence, dict):
            raise SchemaError(f"{w}.confidence: expected an object")
        for cell, value in confidence.items():
            _check_cell(cell, f"{w}.confidence")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SchemaError(f"{w}.confidence[{cell}]: expected a number")
        return cls(
            t=float(_req(d, "t", w)),
            board=copy_board(board),
            arm_clear=bool(_req(d, "arm_clear", w)),
            hands_clear=bool(_req(d, "hands_clear", w)),
            confidence={c: float(v) for c, v in confidence.items()},
            frame_id=int(_req(d, "frame_id", w)),
        )


# ---------------------------------------------------------------- 5.2 Lesson


@dataclass
class Step:
    color: str
    cell: str
    level: int
    t_video: Optional[float] = None
    narration: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "cell": self.cell,
            "level": self.level,
            "t_video": self.t_video,
            "narration": self.narration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        w = "Step"
        color = _req(d, "color", w)
        if color not in COLORS:
            raise SchemaError(f"{w}: invalid color {color!r}")
        cell = _check_cell(_req(d, "cell", w), w)
        level = _req(d, "level", w)
        if level not in LEVELS or isinstance(level, bool):
            raise SchemaError(f"{w}: invalid level {level!r}")
        t_video = d.get("t_video")
        return cls(
            color=color,
            cell=cell,
            level=int(level),
            t_video=None if t_video is None else float(t_video),
            narration=d.get("narration"),
        )


@dataclass
class GateCheck:
    name: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateCheck:
        return cls(name=str(_req(d, "name", "GateCheck")), passed=bool(_req(d, "passed", "GateCheck")))


@dataclass
class Gate:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gate:
        return cls(
            passed=bool(_req(d, "passed", "Gate")),
            checks=[GateCheck.from_dict(c) for c in d.get("checks", [])],
        )


@dataclass
class Lesson:
    lesson_id: str
    title: str
    source_video: Optional[str]
    steps: list[Step]
    final_board: Board
    gate: Optional[Gate] = None  # None until the gate has run

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "title": self.title,
            "source_video": self.source_video,
            "steps": [s.to_dict() for s in self.steps],
            "final_board": copy_board(self.final_board),
            "gate": None if self.gate is None else self.gate.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lesson:
        w = "Lesson"
        final_board = validate_board(_req(d, "final_board", w), allow_unknown=False)
        gate = d.get("gate")
        return cls(
            lesson_id=str(_req(d, "lesson_id", w)),
            title=str(_req(d, "title", w)),
            source_video=d.get("source_video"),
            steps=[Step.from_dict(s) for s in _req(d, "steps", w)],
            final_board=copy_board(final_board),
            gate=None if gate is None else Gate.from_dict(gate),
        )


# ---------------------------------------------------------------- 5.4 SessionState


@dataclass
class ShowMe:
    video: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"video": self.video, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShowMe:
        w = "ShowMe"
        return cls(
            video=str(_req(d, "video", w)),
            start=float(_req(d, "start", w)),
            end=float(_req(d, "end", w)),
        )


@dataclass
class Progress:
    # One entry per lesson step, e.g. "pass", "retry", "active", "pending".
    human: list[str] = field(default_factory=list)
    robot: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"human": list(self.human), "robot": list(self.robot)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Progress:
        return cls(human=list(d.get("human", [])), robot=list(d.get("robot", [])))


MODES = ["human", "robot"]
PHASES = ["ready", "running", "complete", "stopped"]
ROBOT_STATUSES = ["idle", "placing", "checking", "fixing", "homing", "stopped", "pointing"]  # pointing: human-mode hover


@dataclass
class SessionState:
    mode: str
    lesson_id: str
    step_index: int
    board: Board
    expected_step: Optional[Step] = None  # None once the lesson is complete
    verdict: Verdict = Verdict.WAITING
    fault_cell: Optional[str] = None
    hint_level: int = 0
    hint_text: Optional[str] = None
    show_me: Optional[ShowMe] = None
    progress: Progress = field(default_factory=Progress)
    # Added for the UI. All optional with defaults, so older producers and consumers still work.
    lesson_title: Optional[str] = None
    final_board: Optional[Board] = None
    human_side: Optional[str] = None       # "row3" (C-row edge, opposite the arm) or "row1", brain/settings.py
    phase: str = "running"                 # one of PHASES
    expected_cell: Optional[str] = None    # slot that needs a cube, from the latest check
    expected_level: Optional[int] = None
    robot_status: str = "idle"             # one of ROBOT_STATUSES
    board_seen: bool = False               # brain has seen at least one stable board (camera is up)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lesson_id": self.lesson_id,
            "step_index": self.step_index,
            "expected_step": None if self.expected_step is None else self.expected_step.to_dict(),
            "verdict": self.verdict.value,
            "fault_cell": self.fault_cell,
            "hint_level": self.hint_level,
            "hint_text": self.hint_text,
            "show_me": None if self.show_me is None else self.show_me.to_dict(),
            "progress": self.progress.to_dict(),
            "board": copy_board(self.board),
            "lesson_title": self.lesson_title,
            "final_board": None if self.final_board is None else copy_board(self.final_board),
            "human_side": self.human_side,
            "phase": self.phase,
            "expected_cell": self.expected_cell,
            "expected_level": self.expected_level,
            "robot_status": self.robot_status,
            "board_seen": self.board_seen,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionState:
        w = "SessionState"
        mode = _req(d, "mode", w)
        if mode not in MODES:
            raise SchemaError(f"{w}: invalid mode {mode!r}")
        try:
            verdict = Verdict(d.get("verdict", Verdict.WAITING.value))
        except ValueError as e:
            raise SchemaError(f"{w}: {e}") from None
        fault_cell = d.get("fault_cell")
        if fault_cell is not None:
            _check_cell(fault_cell, f"{w}.fault_cell")
        expected = d.get("expected_step")
        show_me = d.get("show_me")
        phase = d.get("phase", "running")
        if phase not in PHASES:
            raise SchemaError(f"{w}: invalid phase {phase!r}")
        robot_status = d.get("robot_status", "idle")
        if robot_status not in ROBOT_STATUSES:
            raise SchemaError(f"{w}: invalid robot_status {robot_status!r}")
        expected_cell = d.get("expected_cell")
        if expected_cell is not None:
            _check_cell(expected_cell, f"{w}.expected_cell")
        final_board = d.get("final_board")
        expected_level = d.get("expected_level")
        return cls(
            mode=mode,
            lesson_id=str(_req(d, "lesson_id", w)),
            step_index=int(_req(d, "step_index", w)),
            board=copy_board(validate_board(_req(d, "board", w))),
            expected_step=None if expected is None else Step.from_dict(expected),
            verdict=verdict,
            fault_cell=fault_cell,
            hint_level=int(d.get("hint_level", 0)),
            hint_text=d.get("hint_text"),
            show_me=None if show_me is None else ShowMe.from_dict(show_me),
            progress=Progress.from_dict(d.get("progress") or {}),
            lesson_title=d.get("lesson_title"),
            final_board=None if final_board is None else copy_board(validate_board(final_board)),
            human_side=d.get("human_side"),
            phase=phase,
            expected_cell=expected_cell,
            expected_level=None if expected_level is None else int(expected_level),
            robot_status=robot_status,
            board_seen=bool(d.get("board_seen", False)),
        )
