"""Session state machine: one lesson, one student (README section 6.5).

A session begins in phase "ready": it tracks stable boards but checks nothing. start()
begins the lesson, and refuses while the last stable board seen still has cubes on it
(hint_text asks to clear the board). If a stable board has already been seen, start()
checks it at once, so the first verdict is MISSING ("Place the red cube on A1") rather than
WAITING until the student touches the board. board_seen in the state tells the UI whether
the camera has produced any stable board yet. Then feed it stable boards with on_board().

Progress is a function of the board, not a forward-only counter. On every stable board
step_index becomes the first step whose slot does not hold its color (checker.progress_index),
so taking an earlier block off moves the lesson, its instruction, and the progress marks back
to that step, and putting it back moves them forward again. Mistakes, robot retries, and hint
levels are kept per step, so going back and forth never loses them. While any cell reads
"unknown" the step holds still (an unreadable block must not move the lesson either way) and
the verdict is WAITING with a nudge hint. The lesson is complete when the board is exactly the
final board; every block in place plus an extra one stays on the last step, asking for the
extra to come off.

Each new board is checked against the current step and drives hints (human mode) or the robot
(robot mode). All I/O goes through the injected journal,
robot client, and on_state callback; sleep is injectable so tests run instantly.

Human mode
  Each wrong settle raises the step's hint_level by one, up to 4 (a new step starts at 0):
    1 text   2 ghost cube on the target   3 show-me clip (needs t_video)   4 robot hovers
  "Still placing" (the board is exactly as it was before this step) is not a mistake and
  leaves the hint level alone. Level 4 needs a robot; without one the ladder stops at 3.
  After a hover the arm crosses the grid, so the stabilizer re-emits the same board; that
  one resettle is ignored instead of counted as another mistake.

Robot mode
  place(step) -> home -> next stable board -> check. PASS places the next step right away.
  Anything else is a failed attempt: pick the wrong cube (fault_cell, fault_level), place
  the missing one (expected_cell, expected_level), home, check again. The arm only picks
  cubes resting on the table, so a wrong cube above level 0 stops the session at once
  (home, error state, a message asking for a human) instead of attempting the pick. After
  MAX_ROBOT_RETRIES failed retries on one step it homes, stops, and reports the error in
  hint_text and progress ("error"). {"ok": false} from the robot is journaled and counts
  as a failed attempt; a failed home() stops the session at once, since the camera can't
  see the board past the arm.

robot_moves counts robot commands. The server resets the stabilizer whenever it changes,
so a move that changed nothing still produces a fresh stable board to check.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from brain.checker import CheckResult, check, expected_board, progress_index, unknown_cells
from brain.journal import Journal
from brain.robot_client import RobotClient, RobotReply, fail
from brain.settings import HOVER_HOLD_S, HUMAN_SIDE, MAX_HINT_LEVEL, SHOW_ME_POST_S, SHOW_ME_PRE_S
from shared.config import MAX_ROBOT_RETRIES
from shared.schemas import (
    MODES,
    Board,
    Lesson,
    Progress,
    SessionState,
    ShowMe,
    Verdict,
    boards_equal,
    copy_board,
    empty_board,
)

READY, RUNNING, COMPLETE, STOPPED = "ready", "running", "complete", "stopped"

CLEAR_BOARD = "Please clear the board to start."
READY_TO_START = "Ready to start."
WAITING_FOR_CAMERA = "Waiting for the camera."  # ready, but no stable board seen yet

ROBOT_NEEDS_HAND = "The robot needs a hand. Please check the board, then restart the lesson."


def robot_stack_help(color: str, cell: str) -> str:
    return (f"The robot needs a hand. It can't lift the {color} cube off the stack on {cell}. "
            "Please take it off, then restart the lesson.")


def is_mistake(r: CheckResult) -> bool:
    """A wrong settle: anything but PASS, WAITING, or still placing the current step."""
    if r.verdict in (Verdict.PASS, Verdict.WAITING):
        return False
    return not (r.verdict is Verdict.MISSING and r.fault_cell is None)


class Session:
    def __init__(
        self,
        lesson: Lesson,
        mode: str,
        journal: Journal,
        robot: Optional[RobotClient] = None,
        on_state: Optional[Callable[[SessionState], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
        human_side: str = HUMAN_SIDE,
        max_retries: int = MAX_ROBOT_RETRIES,
        provenance: Optional[Callable[[], dict[str, Any]]] = None,
    ):
        """provenance: returns grid_source, robot, git_commit, git_dirty for session_start
        (the server supplies it; without one those fields read "unknown")."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "robot" and robot is None:
            raise ValueError("robot mode needs a robot client")
        if not lesson.steps:
            raise ValueError(f"lesson {lesson.lesson_id} has no steps")
        self.lesson = lesson
        self.mode = mode
        self.journal = journal
        self.robot = robot
        self.on_state = on_state
        self.sleep = sleep
        self.human_side = human_side
        self.max_retries = max_retries
        self.provenance = provenance
        self.max_hint_level = MAX_HINT_LEVEL if robot is not None else MAX_HINT_LEVEL - 1

        self.phase = READY
        self.error: Optional[str] = None
        self.step_index = 0
        self.board: Board = empty_board()
        self.verdict = Verdict.WAITING
        self.fault_cell: Optional[str] = None
        self.hint_level = 0
        self.hint_text: Optional[str] = None
        self.show_me: Optional[ShowMe] = None
        self.marks = ["active"] + ["pending"] * (len(lesson.steps) - 1)
        n = len(lesson.steps)
        self.step_mistakes = [0] * n   # human: wrong settles, per step
        self.step_retries = [0] * n    # robot: failed retries, per step
        self.step_hints = [0] * n      # human: hint level reached, per step (restored on going back)
        self.placed_current = False  # robot: first place of the current step already issued
        self.robot_moves = 0
        self.expected_cell: Optional[str] = lesson.steps[0].cell
        self.expected_level: Optional[int] = lesson.steps[0].level
        self.hint_text = WAITING_FOR_CAMERA
        self._activity: Optional[str] = None  # robot_status while a robot command is in flight
        self._fixing = False
        self._hover_board: Optional[Board] = None
        self._started = False
        self._ended = False
        self.board_seen = False

    # ------------------------------------------------------------ per-step history

    @property
    def mistakes(self) -> int:
        return self.step_mistakes[self.step_index] if self.step_index < len(self.step_mistakes) else 0

    @mistakes.setter
    def mistakes(self, value: int) -> None:
        if self.step_index < len(self.step_mistakes):
            self.step_mistakes[self.step_index] = value

    @property
    def retries(self) -> int:
        return self.step_retries[self.step_index] if self.step_index < len(self.step_retries) else 0

    @retries.setter
    def retries(self, value: int) -> None:
        if self.step_index < len(self.step_retries):
            self.step_retries[self.step_index] = value

    def _had_trouble(self, k: int) -> bool:
        return self.step_mistakes[k] > 0 or self.step_retries[k] > 0

    # ------------------------------------------------------------ public

    def state(self) -> SessionState:
        n = len(self.lesson.steps)
        marks = list(self.marks)
        return SessionState(
            mode=self.mode,
            lesson_id=self.lesson.lesson_id,
            step_index=self.step_index,
            board=copy_board(self.board),
            expected_step=self.lesson.steps[self.step_index] if self.step_index < n else None,
            verdict=self.verdict,
            fault_cell=self.fault_cell,
            hint_level=self.hint_level,
            hint_text=self.hint_text,
            show_me=self.show_me,
            progress=Progress(
                human=marks if self.mode == "human" else [],
                robot=marks if self.mode == "robot" else [],
            ),
            lesson_title=self.lesson.title,
            final_board=copy_board(self.lesson.final_board),
            human_side=self.human_side,
            phase=self.phase,
            expected_cell=self.expected_cell,
            expected_level=self.expected_level,
            robot_status=self.robot_status(),
            board_seen=self.board_seen,
        )

    def robot_status(self) -> str:
        if self.phase == STOPPED:
            return "stopped"
        if self._activity is not None:
            return self._activity
        return "checking" if self.mode == "robot" and self.phase == RUNNING else "idle"

    def start(self) -> bool:
        """Begin the lesson. Returns False (and stays ready) if the board still has cubes on it."""
        if self.phase != READY:
            return False
        if any(self.board.values()):
            self.hint_text = CLEAR_BOARD
            self._publish()
            return False
        self.phase = RUNNING
        self._started = True
        first = self.lesson.steps[0]
        self.hint_text = (
            f"Place a {first.color} cube on {first.cell}." if self.mode == "human" else "The robot is getting ready."
        )
        self._event("session_start", {
            "lesson_id": self.lesson.lesson_id,
            "title": self.lesson.title,
            "steps": len(self.lesson.steps),
            "mode": self.mode,
            "human_side": self.human_side,
            "max_retries": self.max_retries,
            "grid_source": "unknown",
            "robot": "unknown" if self.robot is not None else "none",
            "git_commit": "unknown",
            "git_dirty": "unknown",
            **(self.provenance() if self.provenance is not None else {}),
        })
        gate = self.lesson.gate
        self._event("gate", {"gate": None if gate is None else gate.to_dict()})
        if gate is None or not gate.passed:
            self._event("note", {"warning": "lesson gate missing or failed"})
        if self.mode == "robot":
            self._home_or_stop()
        if self.board_seen and self.phase == RUNNING:
            self._check_current()   # the board is already stable: grade it now, not on the next change
        self._settle()
        return True

    def _check_current(self) -> None:
        """Run the mode's board handler on the last stable board (no new board event)."""
        board = copy_board(self.board)
        if self.mode == "human":
            self._human_on_board(board)
        else:
            self._robot_on_board(board)

    def on_board(self, board: Board) -> None:
        self.board = copy_board(board)
        self.board_seen = True
        if self.phase == READY:
            self.hint_text = CLEAR_BOARD if any(board.values()) else READY_TO_START
        self._event("board", {"step": self.step_index, "board": copy_board(board)})
        if self.phase != RUNNING:
            return
        try:
            if self.mode == "human":
                self._human_on_board(board)
            else:
                self._robot_on_board(board)
        finally:
            self._settle()

    def end(self, reason: str = "stopped") -> None:
        """Close the session from outside (server shutdown, reset). Safe to call more than once.
        A session that never started writes nothing."""
        self._finish(reason)

    # ------------------------------------------------------------ shared

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self.journal.write(kind, data)
        self._publish()

    def _publish(self) -> None:
        if self.on_state is not None:
            self.on_state(self.state())

    def _settle(self) -> None:
        """The robot commands for this board are done: robot_status falls back to idle/checking."""
        if self._activity is not None:
            self._activity = None
            self._publish()

    def _check(self, board: Board) -> CheckResult:
        r = check(board, self.lesson, self.step_index, self.human_side)
        self.verdict, self.fault_cell = r.verdict, r.fault_cell
        self.expected_cell, self.expected_level = r.expected_cell, r.expected_level
        self.hint_text = r.hint_text if self.mode == "human" else self._robot_check_text(r)
        data = {
            "step": self.step_index,
            "verdict": r.verdict.value,
            "fault_cell": r.fault_cell,
            "fault_level": r.fault_level,
            "expected_cell": r.expected_cell,
            "expected_level": r.expected_level,
            "detail": r.detail,
        }
        data.update({"retry": self.retries} if self.mode == "robot" else {"hint_level": self.hint_level})
        self._event("verdict", data)
        return r

    def _sync_progress(self, board: Board) -> None:
        """Move step_index to where the board says the lesson is (see module docstring)."""
        if unknown_cells(board):
            return  # hold still: the check explains the unreadable block
        n = len(self.lesson.steps)
        k = progress_index(board, self.lesson)
        if k == n and not boards_equal(board, self.lesson.final_board):
            k = n - 1  # every block in place plus extras: stay on the last step until they come off
        old = self.step_index
        if k == old:
            return
        self.step_hints[old] = self.hint_level
        if k > old:
            for j in range(old, k):  # one PASS and one step_advance per step, even when blocks jump several
                s = self.lesson.steps[j]
                self.verdict, self.fault_cell = Verdict.PASS, None
                self.expected_cell, self.expected_level = s.cell, s.level
                self.hint_text = "Nice! That's right." if self.mode == "human" else f"Step {j + 1} checks out."
                data = {"step": j, "verdict": Verdict.PASS.value, "fault_cell": None, "fault_level": None,
                        "expected_cell": s.cell, "expected_level": s.level, "detail": f"step {j} matches"}
                data.update({"retry": self.step_retries[j]} if self.mode == "robot" else {"hint_level": self.step_hints[j]})
                self._event("verdict", data)
                self.step_index = j + 1
                self.verdict = Verdict.WAITING  # the new step has not been checked yet
                self._recompute_marks()
                self._event("step_advance", {"from_step": j, "to_step": j + 1, "first_try": not self._had_trouble(j)})
        else:
            self.step_index = k
            self._recompute_marks()
            self.verdict = Verdict.WAITING  # checked right after this
            self._event("step_back", {"from_step": old, "to_step": k,
                                      "info": f"the block for step {k + 1} is no longer in place"})
        self.placed_current = False
        if k == n:
            self.phase = COMPLETE
            self.verdict = Verdict.PASS
            self.hint_level, self.show_me = 0, None
            self.expected_cell = self.expected_level = None
            self.hint_text = "All done! You built the whole thing." if self.mode == "human" else "The robot finished the lesson."
            self._finish("complete")
            return
        self.hint_level = self.step_hints[k]
        self.show_me = self._show_me_clip() if self.hint_level >= 3 else None
        self.fault_cell = None
        self.expected_cell, self.expected_level = self.lesson.steps[k].cell, self.lesson.steps[k].level

    def _recompute_marks(self) -> None:
        """Progress marks from the current step: done steps pass (or retry, if they ever had a
        mistake), the current one is active, the rest pending."""
        k = self.step_index
        self.marks = [("retry" if self._had_trouble(i) else "pass") if i < k else "active" if i == k else "pending"
                      for i in range(len(self.lesson.steps))]

    def _finish(self, outcome: str, **data: Any) -> None:
        if self._ended or not self._started:
            return
        self._ended = True
        self._event("session_end", {"outcome": outcome, "step": self.step_index, **data})

    # ------------------------------------------------------------ human mode

    def _human_on_board(self, board: Board) -> None:
        if self._hover_board is not None:
            resettle = boards_equal(board, self._hover_board)
            self._hover_board = None
            if resettle:
                self._event("note", {"step": self.step_index, "info": "board resettled after robot hover, not a new attempt"})
                return
        self._sync_progress(board)
        if self.phase != RUNNING:
            return
        r = self._check(board)
        if is_mistake(r):
            self.mistakes += 1
            self.hint_level = min(self.hint_level + 1, self.max_hint_level)
            self.step_hints[self.step_index] = self.hint_level
            self._give_hint(r, board)

    def _give_hint(self, r: CheckResult, board: Board) -> None:
        if self.hint_level >= 3:
            self.show_me = self._show_me_clip()
        self._event("hint", {
            "step": self.step_index,
            "hint_level": self.hint_level,
            "hint_text": self.hint_text,
            "ghost_cell": r.expected_cell,
            "show_me": None if self.show_me is None else self.show_me.to_dict(),
        })
        target = r.expected_cell or r.fault_cell
        if self.hint_level >= 4 and self.robot is not None and target is not None:
            self._robot_cmd("hover", cell=target)
            self.sleep(HOVER_HOLD_S)
            self._robot_cmd("home")
            self._hover_board = copy_board(board)

    def _show_me_clip(self) -> Optional[ShowMe]:
        t = self.lesson.steps[self.step_index].t_video
        if self.lesson.source_video is None or t is None:
            return None
        return ShowMe(video=self.lesson.source_video, start=max(0.0, t - SHOW_ME_PRE_S), end=t + SHOW_ME_POST_S)

    # ------------------------------------------------------------ robot mode

    def _robot_on_board(self, board: Board) -> None:
        before = self.step_index
        self._sync_progress(board)
        if self.phase != RUNNING:
            return
        if self.step_index > before:  # passed: place the next step right away
            self._robot_place_current()
            return
        r = self._check(board)
        if r.verdict is Verdict.WAITING:
            return  # an unreadable block: wait for the next board instead of moving the arm
        still_placing = r.verdict is Verdict.MISSING and r.fault_cell is None
        if still_placing and not self.placed_current:
            self._robot_place_current()
            return
        if self.retries >= self.max_retries:
            self._robot_stop(f"step {self.step_index + 1} still {r.verdict.value} after {self.retries} retries")
            return
        self.retries += 1
        self._robot_fix(r, board)

    def _robot_check_text(self, r: CheckResult) -> str:
        """Robot mode talks about the robot, never tells the watcher to move a cube."""
        if r.verdict is Verdict.PASS:
            return f"Step {self.step_index + 1} checks out."
        if r.verdict is Verdict.MISSING and r.fault_cell is None:
            return "The camera doesn't see the new cube yet."
        if r.verdict is Verdict.WAITING and r.fault_cell is not None:
            return f"The camera can't tell the color of the block on {r.fault_cell}."
        where = f" at {r.fault_cell}" if r.fault_cell else ""
        return f"The camera caught a problem{where}. The robot will fix it."

    def _robot_place_current(self) -> None:
        s = self.lesson.steps[self.step_index]
        self.placed_current = True
        self.hint_text = f"The robot is placing a {s.color} cube on {s.cell}."
        self._robot_cmd("place", color=s.color, cell=s.cell, level=s.level)
        self._home_or_stop()

    def _robot_fix(self, r: CheckResult, board: Board) -> None:
        self._fixing = True
        try:
            self._robot_fix_steps(r, board)
        finally:
            self._fixing = False

    def _robot_fix_steps(self, r: CheckResult, board: Board) -> None:
        picked_from = None
        if r.fault_level is not None and r.fault_cell is not None:
            stack = board[r.fault_cell]
            if r.fault_level != len(stack) - 1:
                self._robot_stop(f"wrong cube on {r.fault_cell} level {r.fault_level} is under another cube")
                return
            if r.fault_level > 0:
                # The arm only grasps cubes resting on the table, so it can't lift one off a stack.
                self._robot_stop(
                    f"wrong cube on {r.fault_cell} level {r.fault_level} is stacked; the robot only picks at level 0",
                    hint=robot_stack_help(stack[r.fault_level], r.fault_cell),
                )
                return
            self.hint_text = f"The robot is taking the {stack[r.fault_level]} cube off {r.fault_cell}."
            reply = self._robot_cmd("pick", cell=r.fault_cell, level=r.fault_level)
            if not reply["ok"]:
                self._home_or_stop()
                return
            picked_from = r.fault_cell
        if r.expected_cell is not None and r.expected_level is not None:
            height = len(board[r.expected_cell]) - (1 if picked_from == r.expected_cell else 0)
            if r.expected_level == height:
                color = expected_board(self.lesson, self.step_index)[r.expected_cell][r.expected_level]
                self.placed_current = True
                self.hint_text = f"The robot is placing a {color} cube on {r.expected_cell}."
                self._robot_cmd("place", color=color, cell=r.expected_cell, level=r.expected_level)
            else:
                self._event("note", {
                    "step": self.step_index,
                    "info": f"cannot reach {r.expected_cell} level {r.expected_level} yet (height {height})",
                })
        self._home_or_stop()

    def _robot_cmd(self, cmd: str, **args: Any) -> RobotReply:
        self.robot_moves += 1
        self._activity = {"home": "homing", "pick": "fixing", "hover": "pointing"}.get(
            cmd, "fixing" if self._fixing else "placing")
        self._event("robot_cmd", {"step": self.step_index, "cmd": cmd, **args})
        try:
            reply = getattr(self.robot, cmd)(**args)
        except Exception as e:  # a client bug must not crash the demo; journal it as a failed call
            reply = fail(f"{type(e).__name__}: {e}")
        if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
            reply = fail(f"bad reply {reply!r}")
        self._event("robot_result", {"step": self.step_index, "cmd": cmd, "ok": reply["ok"], "error": reply.get("error")})
        return reply

    def _home_or_stop(self) -> None:
        if not self._robot_cmd("home")["ok"]:
            self._robot_stop("robot could not return home", home=False)

    def _robot_stop(self, reason: str, home: bool = True, hint: str = ROBOT_NEEDS_HAND) -> None:
        if home:
            self._robot_cmd("home")
        self.phase = STOPPED
        self.error = reason
        if self.step_index < len(self.marks):
            self.marks[self.step_index] = "error"
        self.hint_text = hint
        self._event("note", {"step": self.step_index, "error": reason})
        self._finish("robot_stopped", reason=reason)
