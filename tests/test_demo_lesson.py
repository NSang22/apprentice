import json
from collections import Counter

from brain.fake_robot import FakeRobot
from brain.journal import Journal
from brain.session import COMPLETE, RUNNING, Session
from brain.settings import DEMO_LESSON
from perception.robot_bridge import default_active_colors
from shared.config import CELLS, COLORS
from shared.schemas import Lesson, apply_step, boards_equal, empty_board, validate_board

RAW = json.loads(DEMO_LESSON.read_text())
LESSON = Lesson.from_dict(RAW)


def test_demo_lesson_is_the_flat_sample():
    assert DEMO_LESSON.name == "sample-flat-v1.json"
    assert LESSON.lesson_id == "sample-flat-v1"


def test_demo_lesson_validates():
    validate_board(RAW["final_board"], allow_unknown=False)
    assert list(LESSON.final_board) == CELLS


def test_demo_lesson_replays_to_its_final_board():
    board = empty_board()
    for step in LESSON.steps:
        board = apply_step(board, step)
    assert boards_equal(board, LESSON.final_board)


def test_demo_lesson_fits_the_robot_today():
    steps = LESSON.steps
    assert 5 <= len(steps) <= 6
    assert all(s.level == 0 for s in steps)
    colors = Counter(s.color for s in steps)
    assert set(colors) <= set(COLORS)
    assert max(colors.values()) <= 2  # the kit has 2 blocks of each color
    assert set(colors) <= set(default_active_colors())  # perception separates only these
    for s in steps:  # same colors never touch: they would merge into one blob
        for t in steps:
            dx = abs(ord(s.cell[0]) - ord(t.cell[0]))
            dy = abs(int(s.cell[1]) - int(t.cell[1]))
            assert s is t or s.color != t.color or max(dx, dy) > 1, (s, t)
    assert len({s.cell for s in steps}) == len(steps)  # one cube per cell, spread out


def test_demo_lesson_is_the_default_for_server_and_check_live():
    from brain import check_live, server
    assert server.parser().parse_args([]).lesson == str(DEMO_LESSON)
    assert check_live.parser().parse_args([]).lesson == str(DEMO_LESSON)


def test_robot_mode_builds_the_demo_lesson_with_one_retry(tmp_path):
    robot = FakeRobot(wrong_cell_on={3})
    session = Session(LESSON, "robot", Journal("s-test", "robot", directory=tmp_path), robot=robot)
    session.start()
    for _ in range(40):
        if session.phase != RUNNING:
            break
        session.on_board(robot.board)
    assert session.phase == COMPLETE and robot.board == LESSON.final_board
    assert session.marks.count("retry") == 1
    assert not any(c[0] == "pick" and c[2] > 0 for c in robot.calls)
