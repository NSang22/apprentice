import json

from analytics.fake_journal import (
    LESSON_FILE,
    SAMPLE_PREFIX,
    build_lesson,
    coverage,
    generate,
    missing_coverage,
    write_samples,
)
from shared.config import MAX_ROBOT_RETRIES
from shared.schemas import Lesson, apply_step, boards_equal, empty_board, validate_board


def complete_events(text):
    events = []
    for line in text.split("\n"):
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def test_same_seed_same_output_and_different_seed_differs():
    assert generate(7) == generate(7)
    assert generate(7)[1] != generate(8)[1]


def test_default_run_covers_every_case():
    _, files = generate(7)
    cov = coverage(files)
    assert missing_coverage(cov) == []
    assert cov["sessions"] == {"human": 10, "robot": 6}


def test_every_session_is_marked_sample():
    _, files = generate(7)
    for name, text in files.items():
        assert name.startswith(SAMPLE_PREFIX)
        first = json.loads(text.split("\n")[0])
        assert first["kind"] == "session_start"
        assert first["data"]["sample"] is True
        assert first["session"] + ".jsonl" == name


def test_boards_are_valid_and_robot_retries_are_capped():
    _, files = generate(7)
    for text in files.values():
        for event in complete_events(text):
            if event["kind"] == "board":
                validate_board(event["data"]["board"], allow_unknown=False)
            if event["kind"] == "verdict" and event["mode"] == "robot":
                assert 0 <= event["data"]["retry"] <= MAX_ROBOT_RETRIES


def test_exactly_one_file_ends_mid_line():
    _, files = generate(7)
    cut = [name for name, text in files.items() if not text.endswith("\n")]
    assert len(cut) == 1


def test_lesson_round_trips_and_replays_to_final_board():
    lesson_dict, _ = generate(7)
    lesson = Lesson.from_dict(json.loads(json.dumps(lesson_dict)))
    board = empty_board()
    for step in lesson.steps:
        board = apply_step(board, step)
    assert boards_equal(board, lesson.final_board)
    assert lesson == build_lesson()


def test_write_samples_only_writes_sample_files(tmp_path):
    sessions, lessons = tmp_path / "sessions", tmp_path / "lessons"
    written = write_samples(3, 5, 2, sessions, lessons)
    assert (lessons / LESSON_FILE).exists()
    assert all(p.name.startswith(SAMPLE_PREFIX) for p in written)
    assert len(list(sessions.glob("*.jsonl"))) == 7
