"""Brain server: grid feed -> stabilizer -> session -> journal, SessionState out to the UI.

    python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human
    python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode robot --robot fake --grid sim

Reads GridState from --grid (perception or brain.fake_feed on ws :8765) and broadcasts
SessionState JSON on ws :8766 to every connected UI. A UI that connects late gets the
current state right away. The journal goes to data/sessions/<session_id>.jsonl.

--robot fake uses brain/fake_robot.py. Its moves only show up in a feed that reads the
fake board, so pair it with --grid sim (in-process feed of the fake board; arm_clear is
false while the fake arm is away from home). --robot http://host:8770 talks to the real
robot API. Without --robot, human mode works but its hint ladder stops at level 3.

After any robot command the server resets the stabilizer and drops frames that arrived
while the robot was moving, so the session always checks a board seen after the move.
The demo never crashes on bad input: bad frames and session errors go to the journal
and the console, and the loop keeps running. If the feed drops it reconnects once a second.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Union

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.exceptions import ConnectionClosed

from brain.fake_robot import FakeRobot
from brain.journal import Journal
from brain.robot_client import HttpRobotClient, RobotClient
from brain.session import RUNNING, Session
from brain.settings import FAKE_MOTION_S, SIM_FEED_HZ
from brain.stabilizer import Stabilizer
from shared.config import BRAIN_WS_PORT, CELLS, GRID_WS_PORT
from shared.schemas import GridState, Lesson, SchemaError, SessionState

RETRY_S = 1.0
BAD_FRAME_JOURNAL_EVERY = 100  # journal the 1st, 101st, ... bad frame so a broken feed can't flood the log

Frame = Union[str, GridState]  # raw JSON from a ws feed, or a GridState from the sim feed


class BrainServer:
    def __init__(self, session: Session, journal: Journal, stabilizer: Optional[Stabilizer] = None):
        self.session = session
        self.journal = journal
        self.stabilizer = stabilizer or Stabilizer()
        self.clients: set[ServerConnection] = set()
        self.latest: Optional[str] = None
        self.frames: asyncio.Queue[tuple[float, Frame]] = asyncio.Queue()
        self.discard_before = 0.0
        self.bad_frames = 0
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        session.on_state = self.publish

    # ------------------------------------------------------------ UI side

    def publish(self, state: SessionState) -> None:
        """Called by the session after every event, possibly from a worker thread."""
        msg = json.dumps(state.to_dict())
        if self.loop is None:
            self.latest = msg
        else:
            self.loop.call_soon_threadsafe(self._broadcast, msg)

    def _broadcast(self, msg: str) -> None:
        self.latest = msg
        broadcast(self.clients, msg)

    async def ui_handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        print(f"[brain] ui connected ({len(self.clients)} total)", flush=True)
        try:
            if self.latest is not None:
                await ws.send(self.latest)
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)
            print(f"[brain] ui disconnected ({len(self.clients)} total)", flush=True)

    # ------------------------------------------------------------ feeds

    async def ws_feed(self, url: str) -> None:
        while True:
            try:
                async with connect(url) as ws:
                    print(f"[brain] grid feed connected: {url}", flush=True)
                    async for msg in ws:
                        self.frames.put_nowait((time.monotonic(), msg))
            except (OSError, ConnectionClosed) as e:
                print(f"[brain] grid feed not available ({e.__class__.__name__}); retrying in {RETRY_S:g}s", flush=True)
                await asyncio.sleep(RETRY_S)

    async def sim_feed(self, robot: FakeRobot, hz: float = SIM_FEED_HZ) -> None:
        frame_id = 0
        while True:
            board, at_home = robot.snapshot()
            gs = GridState(t=time.time(), board=board, arm_clear=at_home, hands_clear=True,
                           confidence={c: 0.99 for c in CELLS}, frame_id=frame_id)
            self.frames.put_nowait((time.monotonic(), gs))
            frame_id += 1
            await asyncio.sleep(1.0 / hz)

    # ------------------------------------------------------------ pipeline

    def parse(self, item: Frame) -> Optional[GridState]:
        if isinstance(item, GridState):
            return item
        try:
            return GridState.from_dict(json.loads(item))
        except (SchemaError, ValueError, TypeError) as e:
            self.bad_frames += 1
            if self.bad_frames % BAD_FRAME_JOURNAL_EVERY == 1:
                print(f"[brain] skipping bad frame ({self.bad_frames} so far): {e}", flush=True)
                self.journal.write("note", {"warning": "bad grid frame", "error": str(e), "count": self.bad_frames})
            return None

    async def process(self, until_done: bool = False) -> None:
        """Consume frames forever (or, with until_done, until the session stops running)."""
        while not (until_done and self.session.phase != RUNNING):
            arrived, item = await self.frames.get()
            if arrived < self.discard_before:
                continue  # seen while the robot was moving
            gs = self.parse(item)
            board = None if gs is None else self.stabilizer.feed(gs)
            if board is None:
                continue
            moves = self.session.robot_moves
            try:
                await asyncio.to_thread(self.session.on_board, board)
            except Exception as e:  # fail gracefully in the demo: journal, print, keep going
                traceback.print_exc()
                self.journal.write("note", {"error": f"session crashed on a board: {type(e).__name__}: {e}"})
            if self.session.robot_moves != moves:
                self.stabilizer.reset()
                self.discard_before = time.monotonic()

    async def run(self, grid: str, robot: Optional[RobotClient], ui_host: str, ui_port: int) -> None:
        self.loop = asyncio.get_running_loop()
        async with serve(self.ui_handler, ui_host, ui_port):
            print(f"[brain] SessionState on ws://{ui_host}:{ui_port}", flush=True)
            await asyncio.to_thread(self.session.start)
            if grid == "sim":
                assert isinstance(robot, FakeRobot)
                feed = asyncio.create_task(self.sim_feed(robot))
            else:
                feed = asyncio.create_task(self.ws_feed(grid))
            try:
                await self.process()
            finally:
                feed.cancel()
                self.loop = None  # later events (session.end at shutdown) only update self.latest


# ---------------------------------------------------------------- CLI


def make_robot(spec: Optional[str], wrong_on: list[int], fail_on: list[int]) -> Optional[RobotClient]:
    if spec is None:
        return None
    if spec == "fake":
        return FakeRobot(wrong_cell_on=wrong_on, fail_on=fail_on, motion_s=FAKE_MOTION_S)
    return HttpRobotClient(spec)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lesson", required=True, help="lesson JSON, e.g. tests/fixtures/staircase-v1.json")
    p.add_argument("--mode", choices=["human", "robot"], default="human")
    p.add_argument("--grid", default=f"ws://localhost:{GRID_WS_PORT}",
                   help="GridState feed URL, or 'sim' for an in-process feed of the fake robot's board")
    p.add_argument("--robot", default=None, help="'fake', or the robot API base URL (http://localhost:8770)")
    p.add_argument("--fake-wrong-on", type=int, action="append", default=[], metavar="N",
                   help="fake robot: the Nth place lands on the wrong cell (repeatable)")
    p.add_argument("--fake-fail-on", type=int, action="append", default=[], metavar="N",
                   help="fake robot: the Nth place returns ok=false (repeatable)")
    p.add_argument("--ui-host", default="localhost")
    p.add_argument("--ui-port", type=int, default=BRAIN_WS_PORT)
    args = p.parse_args(argv)

    if args.mode == "robot" and args.robot is None:
        p.error("--mode robot needs --robot fake or --robot <url>")
    if args.grid == "sim" and args.robot != "fake":
        p.error("--grid sim only works with --robot fake")
    if (args.fake_wrong_on or args.fake_fail_on) and args.robot != "fake":
        p.error("--fake-wrong-on / --fake-fail-on only apply to --robot fake")

    lesson = Lesson.from_dict(json.loads(Path(args.lesson).read_text()))
    robot = make_robot(args.robot, args.fake_wrong_on, args.fake_fail_on)
    journal = Journal.open_new(args.mode)
    session = Session(lesson, args.mode, journal, robot=robot)
    server = BrainServer(session, journal)
    print(f"[brain] session {journal.session_id}: lesson {lesson.lesson_id}, mode {args.mode}, "
          f"robot {args.robot or 'none'}, journal {journal.path}", flush=True)
    try:
        asyncio.run(server.run(args.grid, robot, args.ui_host, args.ui_port))
    except KeyboardInterrupt:
        pass
    finally:
        session.end("server_stopped")
        journal.close()
        print(f"[brain] journal written to {journal.path}", flush=True)


if __name__ == "__main__":
    main()
