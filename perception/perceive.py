"""Per-frame perception with a locked grid: frame in, GridState out (see grid_lock and change).

Grid states, as the --debug status band shows them:
  GRID NOT LOCKED     no saved lock and empty-board reference yet: clear the board, press E
                      (or run perception.calibrate). Nothing is published as usable.
  GRID NOT CONFIRMED  a saved lock, but fewer than MIN_CONFIRM_TAGS tags seen where it predicts
                      them since start-up. Nothing usable yet.
  GRID LOCKED         confirmed. Stays locked while blocks or hands cover tags (taped-down camera
                      and sheet); any visible tag off by more than TAG_TOLERANCE_PX loses it.
  GRID LOST: moved    a visible tag moved. hands_clear is false until all 4 tags are visible on
                      an empty board for RELOCK_FRAMES frames, then the lock is refit and saved.

While locked, a board that reads empty with no hands for REFRESH_EMPTY_S seconds becomes the new
empty-board reference (slow light drift), once per empty spell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from perception.blocks import Analysis, board_from, detect_blocks
from perception.change import warp_to_sheet
from perception.colors import Calibration, active_refs
from perception.robot_bridge import default_active_colors
from perception.stacks import Reading, StackMemory, unknown_stack
from robot.perception import segment
from perception.grid_lock import (CALIB_DIR, LOCK_PATH, MIN_CONFIRM_TAGS, TAG_IDS, GridLock, TagCheck, check_tags,
                                  load_lock, lock_from_frame, save_lock, tag_centers)
from robot.perception.sheet import SheetLocalizer
from shared.config import CELLS
from shared.schemas import GridState, empty_board

REF_PATH = CALIB_DIR / "empty_board.png"
REFRESH_EMPTY_S = 3.0
RELOCK_FRAMES = 5

NOT_LOCKED, NOT_CONFIRMED, LOCKED, LOST = "not_locked", "not_confirmed", "locked", "lost"


@dataclass
class FrameResult:
    state: GridState
    grid: str                       # NOT_LOCKED, NOT_CONFIRMED, LOCKED, LOST
    status: str                     # one line for the status band
    check: Optional[TagCheck] = None
    analysis: Optional[Analysis] = None
    reason: str = ""                # why hands_clear is false, if it is
    stacks: dict = field(default_factory=dict)  # cell -> (stack bottom to top, or None, why unknown)

    @property
    def ok(self) -> bool:
        return self.grid == LOCKED


def hands_clear(analysis: Analysis, crowded: bool) -> bool:
    """No hand or unknown object on the sheet and no cell with two blocks. Hidden tags play no part."""
    return not analysis.others and not crowded


class Perceiver:
    def __init__(self, calib: Calibration, localizer: Optional[SheetLocalizer] = None,
                 lock_path: Path = LOCK_PATH, ref_path: Path = REF_PATH,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print,
                 active: Optional[list[str]] = None, block_mm: Optional[float] = None):
        """active: colors blocks may be named (default robot ACTIVE_COLORS); block_mm: block edge
        (default whatever robot_bridge.set_block_mm put in use)."""
        self.calib = calib
        self.active = active or default_active_colors()
        self.refs = active_refs(calib, self.active)
        self.block_mm = float(block_mm or segment.CUBE_SIZE_MM)
        self.localizer = localizer or SheetLocalizer(min_tags=MIN_CONFIRM_TAGS)
        self.lock_path, self.ref_path = lock_path, ref_path
        self.clock, self.log = clock, log
        self.lock: Optional[GridLock] = load_lock(lock_path)
        self.reference: Optional[np.ndarray] = cv2.imread(str(ref_path)) if ref_path.exists() else None
        if self.lock is None or self.reference is None:
            self.lock, self.reference = None, None
        self.confirmed = False
        self.lost = False
        self.relock_count = 0
        self.empty_since: Optional[float] = None
        self.memory = StackMemory()   # per-cell stack history (perception/stacks.py)
        self.refreshed = False

    # ------------------------------------------------------------ capture

    def capture(self, frame_bgr: np.ndarray) -> tuple[bool, str]:
        """Lock the grid and take the empty-board reference from this frame (all 4 tags, empty board)."""
        lock, why = lock_from_frame(frame_bgr, self.localizer)
        if lock is None:
            return False, why
        self._adopt(lock, warp_to_sheet(frame_bgr, lock.H))
        return True, f"grid locked and empty board saved ({self.lock_path.name}, {self.ref_path.name})"

    def _adopt(self, lock: GridLock, reference: np.ndarray) -> None:
        self.lock, self.reference = lock, reference
        save_lock(lock, self.lock_path)
        self._save_reference()
        self.confirmed, self.lost, self.relock_count = True, False, 0
        self.empty_since, self.refreshed = None, False

    def _save_reference(self) -> None:
        self.ref_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.ref_path), self.reference)

    # ------------------------------------------------------------ per frame

    def step(self, frame_bgr: np.ndarray, frame_id: int, timestamp: Optional[float] = None) -> FrameResult:
        t = time.time() if timestamp is None else timestamp
        visible = tag_centers(self.localizer.detect_tags(frame_bgr))
        if self.lock is None:
            return self._unusable(t, frame_id, NOT_LOCKED, "GRID NOT LOCKED: clear the board and press E", "grid not locked")
        h, w = frame_bgr.shape[:2]
        if tuple(self.lock.frame_size) not in ((w, h), (0, 0)):
            return self._unusable(t, frame_id, NOT_LOCKED,
                                  f"GRID NOT LOCKED: lock is {self.lock.frame_size[0]}x{self.lock.frame_size[1]}, "
                                  f"camera gives {w}x{h}; clear the board and press E", "grid not locked")
        check = check_tags(self.lock.tag_centers, visible)
        if check.status == "moved" and not self.lost:
            self.lost, self.confirmed, self.relock_count = True, False, 0
            self.log(f"[perception] grid moved: tags {list(check.moved)} off by up to {check.max_error_px:.0f} px")
        if self.lost:
            return self._try_relock(frame_bgr, t, frame_id, check, visible)
        if not self.confirmed:
            if check.status != "ok":
                return self._unusable(t, frame_id, NOT_CONFIRMED,
                                      f"GRID NOT CONFIRMED: {check.visible}/4 tags seen, need {MIN_CONFIRM_TAGS} "
                                      "where the lock expects them", "grid not confirmed", check)
            self.confirmed = True
            self.log(f"[perception] saved grid lock confirmed by {check.visible} tags "
                     f"(off by at most {check.max_error_px:.1f} px)")

        analysis = detect_blocks(frame_bgr, self.lock.pose(), self.reference, self.refs, self.block_mm)
        live = analysis.live_sheet
        board, confidence, crowded = board_from(analysis)
        clear = hands_clear(analysis, crowded)
        reason = "" if clear else ("two blocks in one cell" if crowded and not analysis.others
                                   else f"{analysis.others[0].kind} on the sheet")
        stacks = self._stacks(analysis, board, confidence, clear)
        self._maybe_refresh(live, analysis, clear)
        state = GridState(t=t, board=board, arm_clear=True, hands_clear=clear, confidence=confidence, frame_id=frame_id)
        return FrameResult(state, LOCKED, f"GRID LOCKED ({check.visible}/4 tags)", check, analysis, reason, stacks)

    def _stacks(self, analysis: Analysis, board: dict, confidence: dict, usable: bool) -> dict:
        """Replace each occupied cell's top color with its full stack (bottom to top), from the
        stack height, side bands, and memory; "unknown" (confidence 0) when they disagree."""
        best = {}
        for b in sorted(analysis.blocks, key=lambda b: b.score, reverse=True):
            if b.cell is not None and b.cell not in best:
                best[b.cell] = b
        readings = {cell: Reading(b.level, b.color, b.sides) for cell, b in best.items()}
        self.memory.update(readings, usable)
        out = {}
        for cell, r in readings.items():
            stack, why = self.memory.resolve(cell, r)
            if stack is None:
                board[cell], confidence[cell] = unknown_stack(r.level), 0.0
            else:
                board[cell] = stack
            out[cell] = (stack, best[cell].level_reason if r.level is None else why)
        return out

    def _try_relock(self, frame_bgr, t, frame_id, check, visible) -> FrameResult:
        status = "GRID LOST: moved (re-locks when all 4 tags show on an empty board)"
        if len(visible) == len(TAG_IDS):
            lock, _ = lock_from_frame(frame_bgr, self.localizer)
            empty = lock is not None and detect_blocks(frame_bgr, lock.pose(), self.reference, self.refs,
                                                       self.block_mm).empty
            self.relock_count = self.relock_count + 1 if empty else 0
            if empty and self.relock_count >= RELOCK_FRAMES:
                self._adopt(lock, warp_to_sheet(frame_bgr, lock.H))
                self.log("[perception] grid re-locked on an empty board")
                return self._unusable(t, frame_id, LOCKED, "GRID LOCKED (4/4 tags), re-locked", "re-locking", check)
        else:
            self.relock_count = 0
        return self._unusable(t, frame_id, LOST, status, "grid moved", check)

    def _maybe_refresh(self, live: np.ndarray, analysis: Analysis, clear: bool) -> None:
        if not (analysis.empty and clear):
            self.empty_since, self.refreshed = None, False
            return
        now = self.clock()
        if self.empty_since is None:
            self.empty_since = now
        elif not self.refreshed and now - self.empty_since >= REFRESH_EMPTY_S:
            self.reference, self.refreshed = live, True
            self._save_reference()

    def _unusable(self, t, frame_id, grid, status, reason, check=None) -> FrameResult:
        self.memory.update({}, usable=False)
        state = GridState(t=t, board=empty_board(), arm_clear=True, hands_clear=False,
                          confidence={c: 0.0 for c in CELLS}, frame_id=frame_id)
        return FrameResult(state, grid, status, check, None, reason)
