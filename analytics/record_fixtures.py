"""Re-record tests/fixtures/brain_journal_{human,robot}.jsonl from the real brain.

    python -m analytics.record_fixtures

The fixtures are the brain's own journals, so the analytics tests check the loader against what
the brain really writes. Never edit them by hand: re-run this whenever the brain's journal
changes, and commit the result.

Both sessions run through brain.server's BrainServer the way `python -m brain.server ...
--autostart` does: frames into its queue, the stabilizer, the Session on a worker thread, the
journal. The only difference from the CLI is where the frames come from, so a judge can move a
cube mid-run. Defaults match the CLI (fake robot motion time, sim feed rate, hover hold).

  human  lessons/sample-flat-v1.json over fake_feed-style frames: a pass, four mistakes on one
         step (hint levels 1 to 4, so the fake robot hovers and the board resettles), red lifted
         off A1 and put back (step_back, then a re-pass), a block the camera can't read
         (WAITING with fault_cell), then the rest of the lesson to complete.
  robot  the same lesson over the sim feed with a fake robot: the 2nd place lands on the wrong
         cell and the retry passes, a judge slides red from A1 to A2 (step_back, the robot puts
         it back), and every place on A3 misses, so the session stops at the retry cap.

Each run is checked for those events before anything is written; a run that misses one fails
loudly and leaves the fixtures as they were. Takes about a minute.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from brain.fake_robot import FakeRobot
from brain.journal import Journal
from brain.server import BrainServer
from brain.session import Session
from brain.settings import DEMO_LESSON, FAKE_MOTION_S
from shared.config import CELLS, GRID_WS_PORT, MIN_CONFIDENCE, REPO_ROOT
from shared.schemas import Board, GridState, Lesson, empty_board

FIXTURES = REPO_ROOT / "tests" / "fixtures"
HUMAN_FIXTURE = FIXTURES / "brain_journal_human.jsonl"
ROBOT_FIXTURE = FIXTURES / "brain_journal_robot.jsonl"

FEED_HZ = 10.0                     # brain.fake_feed's rate
HANDS_FRAMES = 3                   # a hand over the grid between boards, so every board is a new settle
CONF_OK, CONF_UNSURE = 0.95, 0.5   # fake_feed's confidences; below MIN_CONFIDENCE reads as unknown
SETTLE_TIMEOUT_S = 30.0
RUN_TIMEOUT_S = 180.0

# Human run: (cells that change, number of stable boards to wait for, a cell the camera can't read).
# Lesson: red A1, teal C1, purple B2, yellow A3, skyblue C3, orange C2.
HUMAN_SCRIPT: list[tuple[dict[str, list[str]], int, Optional[str]]] = [
    ({}, 1, None),                                  # empty board: autostart, "Place a red cube on A1"
    ({"A1": ["red"]}, 1, None),                     # step 0 passes
    ({"B1": ["teal"]}, 1, None),                    # WRONG_CELL, hint 1
    ({"B1": [], "C2": ["teal"]}, 1, None),          # WRONG_CELL, hint 2
    ({"C2": [], "C1": ["purple"]}, 1, None),        # WRONG_COLOR, hint 3
    ({"C1": [], "B1": ["teal"]}, 2, None),          # WRONG_CELL, hint 4: hover, then the same board resettles
    ({"B1": [], "C1": ["teal"]}, 1, None),          # step 1 passes
    ({"B2": ["purple"]}, 1, None),                  # step 2 passes
    ({"A1": []}, 1, None),                          # red lifted off: step_back to 0
    ({"A1": ["red"]}, 1, None),                     # back on: re-pass 0 to 2
    ({"A3": ["yellow"]}, 1, "A3"),                  # unreadable: WAITING, fault_cell A3
    ({}, 1, None),                                  # readable now: step 3 passes
    ({"C3": ["skyblue"]}, 1, None),
    ({"C2": ["orange"]}, 1, None),                  # complete
]

ROBOT_WRONG_ON = {2}           # the 2nd place lands on the nearest other cell; the retry passes
ROBOT_MISS_CELL = "A3"         # every place here lands on the nearest other cell: retry cap
JUDGE_FROM, JUDGE_TO = "A1", "A2"
JUDGE_AFTER_CUBES = 3          # the judge acts once the robot is home with this many cubes down


class CountingJournal(Journal):
    """The brain's Journal, counting what it writes (the file is unchanged)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.counts: Counter = Counter()

    def write(self, kind: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        line = super().write(kind, data)
        self.counts[kind] += 1
        return line


class JudgeRobot(FakeRobot):
    """The CLI's fake robot, plus a judge who slides one cube to another cell while the arm is
    home, and a cell where every place misses."""

    def __init__(self, **kwargs: Any):
        super().__init__(wrong_cell_on=ROBOT_WRONG_ON, motion_s=FAKE_MOTION_S, **kwargs)
        self.judged = False

    def place(self, color: str, cell: str, level: int):
        if cell == ROBOT_MISS_CELL:
            self.wrong_cell_on.add(self.place_count + 1)
        return super().place(color, cell, level)

    def home(self):
        reply = super().home()
        with self._lock:
            if not self.judged and sum(map(len, self.board.values())) >= JUDGE_AFTER_CUBES and self.board[JUDGE_FROM]:
                self.board[JUDGE_TO].append(self.board[JUDGE_FROM].pop())
                self.judged = True
        return reply


def _frame(board: Board, hands_clear: bool, unsure: Optional[str], frame_id: int) -> str:
    confidence = {c: CONF_UNSURE if c == unsure else CONF_OK for c in CELLS}
    gs = GridState(t=time.time(), board=board, arm_clear=True, hands_clear=hands_clear,
                   confidence=confidence, frame_id=frame_id)
    return json.dumps({**gs.to_dict(), "source": "fake_feed"})  # as brain.fake_feed sends it


async def _feed(server: BrainServer, journal: CountingJournal) -> None:
    """Play HUMAN_SCRIPT into the server's frame queue, one board at a time, each held until the
    brain has settled it the expected number of times."""
    board, frame_id = empty_board(), 0

    async def send(hands_clear: bool, unsure: Optional[str]) -> None:
        nonlocal frame_id
        server.frames.put_nowait((time.monotonic(), _frame(board, hands_clear, unsure, frame_id)))
        frame_id += 1
        await asyncio.sleep(1.0 / FEED_HZ)

    for changes, settles, unsure in HUMAN_SCRIPT:
        board = {**board, **{c: list(v) for c, v in changes.items()}}
        for _ in range(HANDS_FRAMES):
            await send(False, None)
        want = journal.counts["board"] + settles
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while journal.counts["board"] < want:
            if time.monotonic() > deadline:
                raise RuntimeError(f"board {changes} never settled {settles} time(s)")
            await send(True, unsure)
    while True:  # keep the last board coming until the session completes
        await send(True, None)


async def _record(mode: str, tmp: Path) -> Path:
    lesson = Lesson.from_dict(json.loads(DEMO_LESSON.read_text(encoding="utf-8")))
    journal = CountingJournal.open_new(mode, directory=tmp)
    if mode == "human":
        robot: FakeRobot = FakeRobot(motion_s=FAKE_MOTION_S)
        grid = f"ws://localhost:{GRID_WS_PORT}"
    else:
        robot, grid = JudgeRobot(), "sim"
    server = BrainServer(Session(lesson, mode, journal, robot=robot), journal, grid=grid, robot_spec="fake")
    server.loop = asyncio.get_running_loop()
    server.autostart_pending = True
    if mode == "human":
        source = asyncio.create_task(_feed(server, journal))
    else:
        server.sim_robot = robot
        source = asyncio.create_task(server.sim_feed(robot))
    try:
        process = asyncio.create_task(server.process(until_done=True))
        done, _ = await asyncio.wait({process, source}, timeout=RUN_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        if process not in done:
            if source in done:
                source.result()  # the feed failed: raise its error
            raise RuntimeError(f"{mode} session did not finish in {RUN_TIMEOUT_S:g}s")
    finally:
        source.cancel()
        server.loop = None
        server.session.end("server_stopped")  # as brain.server main does on exit
        journal.close()
    return journal.path


def check(mode: str, events: list[dict]) -> list[str]:
    """What the recorded run is missing, from the scenario in the module docstring."""
    kinds = Counter(e["kind"] for e in events)
    verdicts = [e["data"] for e in events if e["kind"] == "verdict"]
    end = next((e["data"] for e in events if e["kind"] == "session_end"), {})
    want = {
        "step_back": kinds["step_back"] >= 1,
        "a mistake": any(v["verdict"] not in ("PASS", "WAITING", "MISSING") for v in verdicts),
    }
    if mode == "human":
        want["hint level 4"] = any(e["kind"] == "hint" and e["data"]["hint_level"] == 4 for e in events)
        want["hover"] = any(e["kind"] == "robot_cmd" and e["data"]["cmd"] == "hover" for e in events)
        want["resettle note"] = any(e["kind"] == "note" and "resettled" in str(e["data"].get("info")) for e in events)
        want["unreadable block"] = any(v["verdict"] == "WAITING" and v["fault_cell"] for v in verdicts)
        want["grid_source note"] = any(e["kind"] == "note" and "grid_source" in e["data"] for e in events)
        want["outcome complete"] = end.get("outcome") == "complete"
    else:
        want["retry that passes"] = any(v["verdict"] == "PASS" and v.get("retry") for v in verdicts)
        want["pick"] = any(e["kind"] == "robot_cmd" and e["data"]["cmd"] == "pick" for e in events)
        want["outcome robot_stopped"] = end.get("outcome") == "robot_stopped"
    return [name for name, ok in want.items() if not ok]


def main() -> None:
    from brain.journal import read_journal  # the brain's own reader: strict, a bad line raises

    texts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode, out in (("human", HUMAN_FIXTURE), ("robot", ROBOT_FIXTURE)):
            print(f"recording {mode} ...", flush=True)
            path = asyncio.run(_record(mode, Path(tmp)))
            gaps = check(mode, read_journal(path))
            if gaps:
                raise SystemExit(f"{mode} run is missing: {', '.join(gaps)}. Fixtures left unchanged.")
            texts[out] = path.read_text(encoding="utf-8")
    for out, text in texts.items():
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {out.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
