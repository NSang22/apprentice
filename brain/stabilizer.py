"""Stabilizer: turns the ~10 Hz GridState feed into stable boards (README section 6.1).

A board counts as stable once it has been identical for STABLE_FRAMES consecutive usable
frames. A frame is usable only if hands_clear and arm_clear are true and every EMPTY cell has
confidence >= MIN_CONFIDENCE (a cell with no confidence entry counts as 0). An occupied cell
below MIN_CONFIDENCE becomes "unknown" in the frame's board instead of discarding the frame,
so an unreadable block still settles into a board the session can explain ("I can't tell the
color of the block on B2") rather than freezing it; the checker never passes "unknown". Any
unusable or different frame resets the count.

A stable board is emitted once and not repeated while it stays stable. It is emitted
again when it settles after an interruption (a frame with hands_clear or arm_clear
false), even if it is identical to the last emitted board: a hand that reached in and
pulled back, or a robot move that changed nothing, still counts as a new settle. Frames
that differ (jitter, a block flickering to "unknown") reset the count but are not
interruptions, so jitter alone never repeats a board.

The logic lives in pure functions over an immutable StabilizerState; Stabilizer is a thin
stateful wrapper for callers that want feed(gs) -> board | None.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from shared.config import CELLS, MIN_CONFIDENCE, STABLE_FRAMES, UNKNOWN
from shared.schemas import Board, GridState, boards_equal, copy_board


@dataclass(frozen=True)
class StabilizerState:
    candidate: Optional[Board] = None  # board seen on the current run of usable frames
    count: int = 0                     # length of that run
    emitted: Optional[Board] = None    # last board returned as stable
    interrupted: bool = False          # hands or arm crossed the grid since the last emit


def frame_board(gs: GridState, min_confidence: float = MIN_CONFIDENCE) -> Optional[Board]:
    """The board this frame vouches for (low-confidence blocks as "unknown"), or None if the
    frame is unusable."""
    if not (gs.hands_clear and gs.arm_clear):
        return None
    board = copy_board(gs.board)
    for cell in CELLS:
        if gs.confidence.get(cell, 0.0) >= min_confidence:
            continue
        if not board[cell]:
            return None  # can't vouch that this cell is empty
        board[cell] = [UNKNOWN] * len(board[cell])
    return board


def frame_usable(gs: GridState, min_confidence: float = MIN_CONFIDENCE) -> bool:
    return frame_board(gs, min_confidence) is not None


def step(
    state: StabilizerState,
    gs: GridState,
    stable_frames: int = STABLE_FRAMES,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[StabilizerState, Optional[Board]]:
    """Advance the state by one frame. Returns (new state, newly stable board or None)."""
    board = frame_board(gs, min_confidence)
    if board is None:
        interrupted = state.interrupted or not (gs.hands_clear and gs.arm_clear)
        return replace(state, candidate=None, count=0, interrupted=interrupted), None
    if state.candidate is not None and boards_equal(state.candidate, board):
        state = replace(state, count=state.count + 1)
    else:
        state = replace(state, candidate=board, count=1)
    if state.count >= stable_frames and (
        state.emitted is None or state.interrupted or not boards_equal(state.emitted, state.candidate)
    ):
        state = replace(state, emitted=state.candidate, interrupted=False)
        return state, copy_board(state.candidate)
    return state, None


class Stabilizer:
    def __init__(self, stable_frames: int = STABLE_FRAMES, min_confidence: float = MIN_CONFIDENCE):
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self.state = StabilizerState()

    def feed(self, gs: GridState) -> Optional[Board]:
        self.state, board = step(self.state, gs, self.stable_frames, self.min_confidence)
        return board

    def reset(self) -> None:
        """Forget everything, so the current board is emitted again once it settles."""
        self.state = StabilizerState()
