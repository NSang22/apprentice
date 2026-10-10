import asyncio
import json
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from brain.fake_robot import FakeRobot
from brain.journal import Journal, read_journal
from brain.server import BrainServer
from brain.session import COMPLETE, STOPPED, Session
from shared.config import CELLS
from shared.schemas import GridState, Lesson, empty_board

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LESSON = Lesson.from_dict(json.loads((FIXTURES_DIR / "staircase-v1.json").read_text()))


def make_server(tmp_path, mode="robot", robot=None):
    journal = Journal("s-test", mode, directory=tmp_path)
    session = Session(LESSON, mode, journal, robot=robot)
    return BrainServer(session, journal), session, journal


async def run_sim(server, session, robot, timeout_s=20.0):
    server.loop = asyncio.get_running_loop()
    await asyncio.to_thread(session.start)
    feed = asyncio.create_task(server.sim_feed(robot, hz=200.0))
    try:
        await asyncio.wait_for(server.process(until_done=True), timeout_s)
    finally:
        feed.cancel()


def test_robot_mode_end_to_end_over_sim_feed_with_a_retry(tmp_path):
    robot = FakeRobot(wrong_cell_on={2}, motion_s=0.03)
    server, session, journal = make_server(tmp_path, robot=robot)
    asyncio.run(run_sim(server, session, robot))
    journal.close()
    assert session.phase == COMPLETE
    assert robot.board == LESSON.final_board
    assert session.marks[1] == "retry"
    # every verdict was taken on a board seen after the robot was home again
    verdicts = [l["data"]["verdict"] for l in read_journal(journal.path) if l["kind"] == "verdict"]
    assert verdicts.count("PASS") == len(LESSON.steps) and "WRONG_CELL" in verdicts


def test_robot_mode_stops_at_retry_cap_over_sim_feed(tmp_path):
    robot = FakeRobot(wrong_cell_on={1, 2, 3, 4}, motion_s=0.0)
    server, session, journal = make_server(tmp_path, robot=robot)
    asyncio.run(run_sim(server, session, robot))
    journal.close()
    assert session.phase == STOPPED and robot.place_count == 3


def test_ui_gets_latest_state_on_connect_and_every_update(tmp_path):
    server, session, journal = make_server(tmp_path, mode="human")

    async def scenario():
        server.loop = asyncio.get_running_loop()
        session.start()  # publishes before any UI is connected
        async with serve(server.ui_handler, "127.0.0.1", 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ui:
                first = json.loads(await asyncio.wait_for(ui.recv(), 2))
                b = empty_board()
                b["A1"] = ["red"]
                await asyncio.to_thread(session.on_board, b)  # from a worker thread, like process()
                updates = []
                while not updates or updates[-1]["step_index"] != 1:
                    updates.append(json.loads(await asyncio.wait_for(ui.recv(), 2)))
                return first, updates

    first, updates = asyncio.run(scenario())
    journal.close()
    assert first["step_index"] == 0 and first["hint_text"] == "Place a red cube on A1."
    assert updates[-1]["expected_step"]["color"] == "green"
    assert updates[-1]["progress"]["human"][:2] == ["pass", "active"]


def test_bad_frames_are_skipped_and_journaled_sparingly(tmp_path):
    server, session, journal = make_server(tmp_path, mode="human")
    for _ in range(150):
        assert server.parse("not json") is None
    assert server.parse('{"t": 0}') is None
    good = GridState(t=0.0, board=empty_board(), arm_clear=True, hands_clear=True,
                     confidence={c: 1.0 for c in CELLS}, frame_id=1)
    assert server.parse(json.dumps(good.to_dict())) == good
    journal.close()
    notes = [l for l in read_journal(journal.path) if l["kind"] == "note"]
    assert [n["data"]["count"] for n in notes] == [1, 101]
