import json
from dataclasses import replace
from pathlib import Path

import pytest

from brain.checker import expected_board
from brain.fake_robot import FakeRobot
from brain.journal import Journal, read_journal
from brain.session import COMPLETE, ROBOT_NEEDS_HAND, RUNNING, STOPPED, Session
from shared.config import MAX_ROBOT_RETRIES
from shared.schemas import Lesson, SessionState, Verdict, copy_board, empty_board

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# staircase-v1 steps:
#   0 red A1 L0 | 1 green B1 L0 | 2 blue C1 L0 | 3 blue A1 L1 | 4 red B1 L1 | 5 yellow A1 L2
LESSON = Lesson.from_dict(json.loads((FIXTURES_DIR / "staircase-v1.json").read_text()))
N = len(LESSON.steps)


def board(**cells):
    return {**empty_board(), **cells}


def after(k):
    return expected_board(LESSON, k)


def with_video(lesson=LESSON):
    steps = [replace(s, t_video=4.0 + 5 * i) for i, s in enumerate(lesson.steps)]
    return replace(lesson, source_video="videos/expert_staircase.mp4", steps=steps)


class Harness:
    """A session wired to a tmp journal, recording every published SessionState."""

    def __init__(self, tmp_path, mode="human", robot=None, lesson=LESSON, **kw):
        self.journal = Journal("s-test", mode, directory=tmp_path, clock=lambda: 0.0)
        self.states: list[SessionState] = []
        self.sleeps: list[float] = []
        self.session = Session(lesson, mode, self.journal, robot=robot, on_state=self.states.append,
                               sleep=self.sleeps.append, **kw)
        self.session.start()

    def feed(self, *boards):
        for b in boards:
            self.session.on_board(copy_board(b))
        return self.session

    def events(self, kind=None):
        lines = read_journal(self.journal.path)
        return [l for l in lines if kind is None or l["kind"] == kind]

    def verdicts(self):
        return [(e["data"]["step"], e["data"]["verdict"]) for e in self.events("verdict")]


def run_robot(h: Harness, robot: FakeRobot, max_boards=60):
    """Close the loop like the camera would: after each session action, show it the fake board."""
    for _ in range(max_boards):
        if h.session.phase != RUNNING:
            return
        h.feed(robot.board)
    raise AssertionError("robot session never finished")


# ================================================================ human mode


def test_human_clean_run_completes(tmp_path):
    h = Harness(tmp_path)
    s = h.feed(empty_board(), *(after(k) for k in range(N)))
    assert s.phase == COMPLETE
    assert s.marks == ["pass"] * N
    st = s.state()
    assert st.expected_step is None and st.step_index == N and st.verdict is Verdict.PASS
    assert st.progress.human == ["pass"] * N and st.progress.robot == []
    assert [e["data"]["to_step"] for e in h.events("step_advance")] == list(range(1, N + 1))
    assert h.events("session_end")[-1]["data"]["outcome"] == "complete"
    assert h.events("hint") == []


def test_human_start_journals_session_and_gate(tmp_path):
    h = Harness(tmp_path)
    kinds = [e["kind"] for e in h.events()]
    assert kinds[:2] == ["session_start", "gate"]
    assert h.events("session_start")[0]["data"]["lesson_id"] == "staircase-v1"
    assert any("gate" in e["data"].get("warning", "") for e in h.events("note"))  # fixture has no gate yet
    assert h.session.state().hint_text == "Place a red cube on A1."


def test_human_pass_rechecks_same_board_for_next_step(tmp_path):
    h = Harness(tmp_path)
    h.feed(after(0))
    assert h.verdicts() == [(0, "PASS"), (1, "MISSING")]
    st = h.session.state()
    assert st.step_index == 1 and st.verdict is Verdict.MISSING and "green" in st.hint_text
    assert st.progress.human[:3] == ["pass", "active", "pending"]


def test_human_hints_escalate_on_each_wrong_settle_and_reset_on_pass(tmp_path):
    robot = FakeRobot()
    h = Harness(tmp_path, robot=robot, lesson=with_video())
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1))
    levels = []
    for _ in range(3):
        h.feed(wrong)  # the stabilizer re-emits this after every hand interruption
        levels.append(h.session.hint_level)
    assert levels == [1, 2, 3]
    st = h.session.state()
    assert st.verdict is Verdict.WRONG_CELL and st.fault_cell == "C2"
    assert st.show_me is not None and (st.show_me.start, st.show_me.end) == (11.0, 15.0)  # t_video 14.0
    assert robot.calls == []  # no hover below level 4

    h.feed(after(2))
    st = h.session.state()
    assert st.hint_level == 0 and st.show_me is None and st.step_index == 3
    assert st.progress.human[:4] == ["pass", "pass", "retry", "active"]
    assert [e["data"]["hint_level"] for e in h.events("hint")] == [1, 2, 3]


def test_human_level_4_hovers_robot_over_expected_cell(tmp_path):
    robot = FakeRobot()
    h = Harness(tmp_path, robot=robot)
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1), wrong, wrong, wrong, wrong)
    assert h.session.hint_level == 4
    assert robot.calls == [("hover", "C1"), ("home",)]
    assert h.sleeps == [3.0]
    assert [e["data"]["cmd"] for e in h.events("robot_cmd")] == ["hover", "home"]


def test_human_resettle_after_hover_is_not_another_mistake(tmp_path):
    robot = FakeRobot()
    h = Harness(tmp_path, robot=robot)
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1), wrong, wrong, wrong, wrong)
    hints_before = len(h.events("hint"))
    h.feed(wrong)  # re-emitted because the arm crossed the grid
    assert len(h.events("hint")) == hints_before
    assert robot.calls == [("hover", "C1"), ("home",)]
    h.feed(wrong)  # the human reached in again without fixing it: a real new settle
    assert robot.calls == [("hover", "C1"), ("home",)] * 2
    assert h.session.hint_level == 4


def test_human_without_robot_caps_hints_at_3(tmp_path):
    h = Harness(tmp_path)
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1), *[wrong] * 6)
    assert h.session.hint_level == 3
    assert not h.events("robot_cmd")


def test_human_show_me_is_none_without_t_video(tmp_path):
    h = Harness(tmp_path)
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1), wrong, wrong, wrong)
    assert h.session.hint_level == 3 and h.session.state().show_me is None


def test_human_still_placing_does_not_escalate(tmp_path):
    h = Harness(tmp_path)
    h.feed(after(0), after(1), after(1), after(1))
    assert h.session.hint_level == 0
    assert h.session.state().verdict is Verdict.MISSING


def test_human_removing_a_wrong_cube_keeps_hint_level(tmp_path):
    h = Harness(tmp_path)
    h.feed(after(0), after(1), board(A1=["red"], B1=["green"], C2=["blue"]), after(1))
    st = h.session.state()
    assert st.hint_level == 1 and st.verdict is Verdict.MISSING and "Place a blue cube on C1" in st.hint_text


@pytest.mark.parametrize(
    "wrong,verdict",
    [
        (board(A1=["red"], B1=["green"], C1=["yellow"]), Verdict.WRONG_COLOR),
        (board(A1=["red"], B1=["green"], C1=["blue"], B3=["yellow"]), Verdict.EXTRA),
        (board(A1=["red"], C1=["blue"]), Verdict.MISSING),  # green knocked off the grid
        (board(A1=["red"], B2=["green"], C1=["blue"]), Verdict.WRONG_CELL),  # green knocked to B2
    ],
)
def test_human_each_kind_of_mistake_escalates(tmp_path, wrong, verdict):
    h = Harness(tmp_path)
    h.feed(after(0), after(1), wrong)
    assert h.session.state().verdict is verdict
    assert h.session.hint_level == 1


def test_human_expert_correction_cube_placed_then_moved(tmp_path):
    h = Harness(tmp_path)
    h.feed(after(0), after(1), board(A1=["red"], B1=["green"], C2=["blue"]), after(2))
    assert h.session.step_index == 3


def test_human_boards_after_complete_are_journaled_only(tmp_path):
    h = Harness(tmp_path)
    h.feed(*(after(k) for k in range(N)))
    n_verdicts = len(h.events("verdict"))
    h.feed(board(A1=["red"]))
    assert len(h.events("verdict")) == n_verdicts
    assert h.events()[-1]["kind"] == "board"


def test_a_state_is_published_after_every_event(tmp_path):
    h = Harness(tmp_path, robot=FakeRobot())
    wrong = board(A1=["red"], B1=["green"], C2=["blue"])
    h.feed(after(0), after(1), wrong, wrong, wrong, wrong, after(2))
    assert len(h.states) == len(h.events())
    for st in h.states:
        assert SessionState.from_dict(json.loads(json.dumps(st.to_dict()))) == st


def test_end_is_idempotent(tmp_path):
    h = Harness(tmp_path)
    h.session.end("server_stopped")
    h.session.end("server_stopped")
    ends = h.events("session_end")
    assert len(ends) == 1 and ends[0]["data"]["outcome"] == "server_stopped"


def test_bad_construction_raises(tmp_path):
    j = Journal("s-x", "robot", directory=tmp_path)
    with pytest.raises(ValueError):
        Session(LESSON, "robot", j)  # no robot client
    with pytest.raises(ValueError):
        Session(LESSON, "teacher", j, robot=FakeRobot())


# ================================================================ robot mode


def robot_harness(tmp_path, **robot_kw):
    robot = FakeRobot(**robot_kw)
    return Harness(tmp_path, mode="robot", robot=robot), robot


def test_robot_clean_run_completes(tmp_path):
    h, robot = robot_harness(tmp_path)
    run_robot(h, robot)
    s = h.session
    assert s.phase == COMPLETE and robot.board == LESSON.final_board
    assert s.state().progress.robot == ["pass"] * N and s.state().progress.human == []
    places = [c for c in robot.calls if c[0] == "place"]
    assert places == [("place", st.color, st.cell, st.level) for st in LESSON.steps]
    assert robot.calls[0] == ("home",)  # homes at start
    # every place is followed by home before the next check
    for i, c in enumerate(robot.calls):
        if c[0] == "place":
            assert robot.calls[i + 1] == ("home",)
    assert h.events("session_end")[-1]["data"]["outcome"] == "complete"


def test_robot_journals_every_command_and_result(tmp_path):
    h, robot = robot_harness(tmp_path)
    run_robot(h, robot)
    cmds, results = h.events("robot_cmd"), h.events("robot_result")
    assert len(cmds) == len(results) == len(robot.calls)
    assert all(r["data"]["ok"] for r in results)


def test_robot_wrong_cell_is_picked_and_re_placed(tmp_path):
    h, robot = robot_harness(tmp_path, wrong_cell_on={2})  # green lands next to B1
    run_robot(h, robot)
    assert h.session.phase == COMPLETE and robot.board == LESSON.final_board
    assert h.session.marks[1] == "retry" and h.session.marks.count("retry") == 1
    wrong_verdict = [e["data"] for e in h.events("verdict") if e["data"]["verdict"] == "WRONG_CELL"][0]
    assert wrong_verdict["step"] == 1 and wrong_verdict["retry"] == 0
    picks = [c for c in robot.calls if c[0] == "pick"]
    assert picks == [("pick", wrong_verdict["fault_cell"], 0)]
    assert [c for c in robot.calls if c[0] == "place"].count(("place", "green", "B1", 0)) == 2


def test_robot_retry_cap_stops_homes_and_flags(tmp_path):
    h, robot = robot_harness(tmp_path, wrong_cell_on={2, 3, 4, 5, 6})
    run_robot(h, robot)
    s = h.session
    assert s.phase == STOPPED and s.step_index == 1
    assert robot.place_count == 1 + 1 + MAX_ROBOT_RETRIES  # step 0, first try at step 1, then the retries
    assert robot.calls[-1] == ("home",)
    st = s.state()
    assert st.hint_text == ROBOT_NEEDS_HAND
    assert st.progress.robot[:3] == ["pass", "error", "pending"]
    assert [e["data"]["retry"] for e in h.events("verdict") if e["data"]["step"] == 1] == [0, 1, 2]
    end = h.events("session_end")[-1]["data"]
    assert end["outcome"] == "robot_stopped" and "after 2 retries" in end["reason"]
    # stopped means stopped: more boards never move the robot again
    n_calls = len(robot.calls)
    h.feed(robot.board, robot.board)
    assert len(robot.calls) == n_calls


def test_robot_ok_false_is_journaled_and_counts_as_a_failed_attempt(tmp_path):
    h, robot = robot_harness(tmp_path, fail_on={1})
    run_robot(h, robot)
    assert h.session.phase == COMPLETE
    failed = [e["data"] for e in h.events("robot_result") if not e["data"]["ok"]]
    assert failed == [{"step": 0, "cmd": "place", "ok": False, "error": "injected failure on place 1"}]
    assert h.session.marks[0] == "retry"
    assert [e["data"]["retry"] for e in h.events("verdict") if e["data"]["step"] == 0] == [0, 0, 1]


def test_robot_repeated_ok_false_hits_the_cap(tmp_path):
    h, robot = robot_harness(tmp_path, fail_on={1, 2, 3, 4})
    run_robot(h, robot)
    assert h.session.phase == STOPPED and robot.place_count == 1 + MAX_ROBOT_RETRIES
    assert h.session.marks[0] == "error"


def test_robot_fixes_a_cube_a_judge_moved(tmp_path):
    h, robot = robot_harness(tmp_path)
    while h.session.step_index < 3:
        h.feed(robot.board)
    # The robot just placed step 3 (blue on A1). A judge slides the green cube from B1 to B2.
    robot.board["B1"].pop()
    robot.board["B2"].append("green")
    run_robot(h, robot)
    assert h.session.phase == COMPLETE and robot.board == LESSON.final_board
    assert ("pick", "B2", 0) in robot.calls and robot.calls.count(("place", "green", "B1", 0)) == 2
    assert h.session.marks[3] == "retry"


def test_robot_removes_an_extra_cube_then_places(tmp_path):
    h, robot = robot_harness(tmp_path)
    robot.board["C3"].append("yellow")  # stray cube on the board before the robot starts
    run_robot(h, robot)
    assert h.session.phase == COMPLETE and robot.board == LESSON.final_board
    assert robot.calls[:4] == [("home",), ("pick", "C3", 0), ("place", "red", "A1", 0), ("home",)]


def test_robot_stops_when_wrong_cube_is_buried(tmp_path):
    h, robot = robot_harness(tmp_path)
    while h.session.step_index < 3:
        h.feed(robot.board)
    h.feed(robot.board)  # step 3 placed and verified, step 4 placed
    robot.board["A1"][0] = "yellow"  # the bottom cube under blue is now the wrong color
    h.feed(robot.board)
    assert h.session.phase == STOPPED
    assert "under another cube" in h.session.error
    assert not any(c[0] == "pick" for c in robot.calls)


class HomeFails(FakeRobot):
    def home(self):
        self.calls.append(("home",))
        return {"ok": False, "error": "servo fault"}


def test_robot_failed_home_stops_at_once(tmp_path):
    robot = HomeFails()
    h = Harness(tmp_path, mode="robot", robot=robot)
    assert h.session.phase == STOPPED and h.session.error == "robot could not return home"
    assert robot.calls == [("home",)]  # does not retry home in a loop


class Explodes(FakeRobot):
    def place(self, color, cell, level):
        raise RuntimeError("socket closed")


def test_robot_client_exception_becomes_a_failed_call(tmp_path):
    robot = Explodes()
    h = Harness(tmp_path, mode="robot", robot=robot)
    run_robot(h, robot)
    assert h.session.phase == STOPPED
    errors = [e["data"]["error"] for e in h.events("robot_result") if not e["data"]["ok"]]
    assert errors and all("RuntimeError: socket closed" in e for e in errors)


def test_robot_moves_counter_tracks_commands(tmp_path):
    h, robot = robot_harness(tmp_path)
    before = h.session.robot_moves
    h.feed(robot.board)
    assert h.session.robot_moves == before + 2  # place + home


def test_robot_new_step_reads_waiting_until_checked(tmp_path):
    h, robot = robot_harness(tmp_path)
    h.feed(robot.board)  # empty board: robot places step 0
    h.feed(robot.board)  # step 0 passes, robot places step 1
    advance = [st for st in h.states if st.step_index == 1][0]
    assert advance.verdict is Verdict.WAITING and advance.progress.robot[:2] == ["pass", "active"]


def test_robot_text_never_tells_the_watcher_to_move_cubes(tmp_path):
    h, robot = robot_harness(tmp_path, wrong_cell_on={2})
    run_robot(h, robot)
    texts = {st.hint_text for st in h.states}
    assert "The camera caught a problem at B2. The robot will fix it." in texts
    assert not any(t and ("Move the" in t or "Swap the" in t or "Take the" in t) for t in texts)
