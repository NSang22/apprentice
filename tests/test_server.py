import asyncio
import json
import time
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from brain.fake_robot import FakeRobot
from brain.journal import Journal, read_journal
from brain.server import BrainServer
from brain.session import COMPLETE, STOPPED, Session
from brain.settings import ROBOT_STATUS_MAX_AGE_S
from shared.config import CELLS, STABLE_FRAMES
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
    assert updates[-1]["expected_step"]["color"] == "teal"
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


# ---------------------------------------------------------------- robot /status gates arm_clear


class StatusOnly:
    """Just the status() half of a robot client, with a reply the test can change."""

    def __init__(self, reply):
        self.reply = reply

    def status(self):
        return dict(self.reply)


def frame(arm_clear=True, frame_id=0):
    b = empty_board()
    b["A1"] = ["red"]
    return GridState(t=0.0, board=b, arm_clear=arm_clear, hands_clear=True,
                     confidence={c: 1.0 for c in CELLS}, frame_id=frame_id)


def gated_server(tmp_path, at_home, age_s=0.0):
    journal = Journal("s-test", "human", directory=tmp_path)
    server = BrainServer(Session(LESSON, "human", journal), journal, robot_status=StatusOnly({}))
    server.at_home, server.at_home_t = at_home, time.monotonic() - age_s
    return server


def test_without_robot_status_arm_clear_passes_through(tmp_path):
    server, _, _ = make_server(tmp_path, mode="human")
    assert server.gate_arm(frame()).arm_clear is True
    assert server.gate_arm(frame(arm_clear=False)).arm_clear is False


def test_robot_at_home_and_fresh_keeps_arm_clear(tmp_path):
    assert gated_server(tmp_path, at_home=True).gate_arm(frame()).arm_clear is True


def test_robot_away_from_home_clears_arm_clear(tmp_path):
    assert gated_server(tmp_path, at_home=False).gate_arm(frame()).arm_clear is False


def test_stale_robot_status_counts_as_not_home(tmp_path):
    server = gated_server(tmp_path, at_home=True, age_s=ROBOT_STATUS_MAX_AGE_S + 0.5)
    assert server.gate_arm(frame()).arm_clear is False


def test_perception_arm_not_clear_wins_even_when_robot_is_home(tmp_path):
    assert gated_server(tmp_path, at_home=True).gate_arm(frame(arm_clear=False)).arm_clear is False


def test_never_any_status_reply_means_not_home(tmp_path):
    journal = Journal("s-test", "human", directory=tmp_path)
    server = BrainServer(Session(LESSON, "human", journal), journal, robot_status=StatusOnly({}))
    assert server.gate_arm(frame()).arm_clear is False


def test_stabilizer_only_settles_once_the_robot_is_home(tmp_path):
    server = gated_server(tmp_path, at_home=False)
    assert [server.accept(frame(frame_id=i)) for i in range(STABLE_FRAMES * 3)] == [None] * (STABLE_FRAMES * 3)
    server.at_home, server.at_home_t = True, time.monotonic()
    out = [server.accept(frame(frame_id=i)) for i in range(STABLE_FRAMES)]
    assert out[:-1] == [None] * (STABLE_FRAMES - 1) and out[-1]["A1"] == ["red"]


@pytest.mark.parametrize(
    "reply,expected",
    [
        ({"ok": True, "busy": False, "at_home": True, "last_error": None}, True),
        ({"ok": True, "busy": True, "at_home": False, "last_error": None}, False),
        ({"ok": False, "error": "GET /status: timed out"}, False),
        ({"ok": True}, False),  # no at_home field: unknown is never home
    ],
)
def test_poll_robot_status_reads_at_home(tmp_path, reply, expected):
    server = gated_server(tmp_path, at_home=not expected, age_s=100.0)

    async def poll_briefly():
        task = asyncio.create_task(server.poll_robot_status(StatusOnly(reply), hz=200.0))
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(poll_briefly())
    assert server.at_home is expected
    assert time.monotonic() - server.at_home_t < ROBOT_STATUS_MAX_AGE_S


# ---------------------------------------------------------------- UI commands: start and reset


def test_start_and_reset_commands(tmp_path):
    from brain.server import Command

    made = []

    def new_session():
        j = Journal(f"s-new{len(made)}", "human", directory=tmp_path)
        s = Session(LESSON, "human", j)
        made.append(j)
        return s, j

    old_journal = Journal("s-old", "human", directory=tmp_path)
    server = BrainServer(Session(LESSON, "human", old_journal), old_journal, new_session=new_session)
    server.command(Command(cmd="start"))
    server.session.on_board(server.session.board)  # empty board: step 0 still to place
    b = empty_board()
    b["A1"] = ["red"]
    server.session.on_board(b)
    assert server.session.step_index == 1

    server.command(Command(cmd="reset"))
    ends = [l for l in read_journal(old_journal.path) if l["kind"] == "session_end"]
    assert [e["data"]["outcome"] for e in ends] == ["reset"]
    assert server.session.phase == "ready" and server.session.step_index == 0
    assert server.journal is made[0] and json.loads(server.latest)["phase"] == "ready"
    assert server.parse_command("not json") is None and server.parse_command('{"cmd": "fly"}') is None


# ---------------------------------------------------------------- provenance in session_start


def test_session_start_records_provenance(tmp_path):
    journal = Journal("s-prov", "human", directory=tmp_path)
    server = BrainServer(Session(LESSON, "human", journal), journal,
                         grid="ws://localhost:8765", robot_spec="http://localhost:8770")
    frame_msg = json.dumps({**frame().to_dict(), "source": "fake_feed"})
    assert server.parse(frame_msg) is not None  # extra "source" key parses fine
    server.session.start()
    journal.close()
    start = [l for l in read_journal(journal.path) if l["kind"] == "session_start"][0]["data"]
    assert {"grid_source", "robot", "mode", "lesson_id", "human_side", "git_commit", "git_dirty"} <= start.keys()
    assert start["grid_source"] == "fake_feed" and start["robot"] == "http:http://localhost:8770"
    assert start["mode"] == "human" and start["git_dirty"] in (True, False, "unknown")


def test_autostart_waits_for_the_first_frame_and_records_its_source(tmp_path):
    journal = Journal("s-auto", "human", directory=tmp_path)
    server = BrainServer(Session(LESSON, "human", journal), journal, grid="ws://localhost:8765")
    server.autostart_pending = True
    assert server.frames.empty()
    server.accept(json.dumps({**frame().to_dict(), "source": "perception"}))
    _, cmd = server.frames.get_nowait()
    assert cmd == {"cmd": "start"} and not server.autostart_pending
    server.command(cmd)
    journal.close()
    start = [l for l in read_journal(journal.path) if l["kind"] == "session_start"][0]["data"]
    assert start["grid_source"] == "perception:ws://localhost:8765"
