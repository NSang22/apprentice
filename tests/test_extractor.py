import json
from pathlib import Path

import pytest

from brain.extractor import (ColorsFileError, FlatLog, build_lesson, extract, flat_frames, gate_lesson, manual_colors,
                             script_frames, step_rows, write_outputs)
from brain.fake_feed import HZ, make_frame, parse_script
from brain.journal import Journal, read_journal
from brain.session import COMPLETE, Session
from brain.stabilizer import Stabilizer
from shared.schemas import Lesson

SCRIPTS = Path(__file__).parent / "scripts"
STAIRCASE = Lesson.from_dict(json.loads((Path(__file__).parent / "fixtures" / "staircase-v1.json").read_text()))


def frames_of(segments, seed=0):
    """In-memory frames for inline script segments (same format as tests/scripts)."""
    import random

    rng, out = random.Random(seed), []
    for seg in parse_script(segments):
        for _ in range(round(seg.duration_s * HZ)):
            out.append(make_frame(seg.board, seg.hands_clear, seg.arm_clear, seg.jitter, len(out), rng, t=len(out) / HZ))
    return out


def seg(board, s=1.5, **flags):
    return {"board": board, "duration_s": s, **flags}


def lesson_from(frames, manual=None):
    ex = extract(frames)
    rows = step_rows(ex, manual_colors(manual or [], ex.steps))
    lesson = build_lesson(rows, "test-lesson", "Test", None)
    return ex, rows, lesson, gate_lesson(lesson, ex)


def placed(lesson):
    return [(s.color, s.cell, s.level) for s in lesson.steps]


def failed(gate):
    return [c.name for c in gate.gate.checks if not c.passed]


def test_clean_build_gives_steps_in_order():
    _, _, lesson, gate = lesson_from(script_frames(SCRIPTS / "staircase_clean.json"))
    assert placed(lesson) == placed(STAIRCASE)
    times = [s.t_video for s in lesson.steps]
    assert times == sorted(times) and times[0] > 0
    assert gate.gate.passed, gate.reasons


def test_expert_correction_leaves_one_step():
    # skyblue goes on C2 first, then moves to C1
    ex, _, lesson, gate = lesson_from(script_frames(SCRIPTS / "staircase_with_mistake.json"))
    assert placed(lesson) == placed(STAIRCASE)
    assert [(c.cell, c.level) for c in ex.removed] == [("C2", 0)]
    assert gate.gate.passed, gate.reasons


def test_two_cubes_at_once_fails_one_cube_check():
    _, _, lesson, gate = lesson_from(frames_of([seg({}), seg({"A1": ["red"], "B2": ["teal"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("teal", "B2", 0)]
    assert failed(gate) == ["one_cube_per_change"]


def test_recording_not_starting_empty_fails_gate():
    _, _, lesson, gate = lesson_from(frames_of([seg({"A1": ["red"]}), seg({"A1": ["red"], "B1": ["teal"]})]))
    assert placed(lesson) == [("teal", "B1", 0)]
    assert "starts_empty" in failed(gate) and "replay_matches_final_frame" in failed(gate)


def test_hands_and_jitter_produce_no_steps():
    frames = frames_of([
        seg({}),
        seg({"B2": ["purple"]}, 1.0, hands_clear=False),   # a cube held over the grid, taken away again
        seg({}),
        seg({"A1": ["red"]}),
        seg({"A1": ["red"]}, 3.0, jitter=True),
        seg({"A1": ["red"]}, 1.0, arm_clear=False),
        seg({"A1": ["red"]}),
    ], seed=1)
    _, _, lesson, gate = lesson_from(frames)
    assert placed(lesson) == [("red", "A1", 0)]
    assert gate.gate.passed, gate.reasons


def test_round_trip_lesson_teaches_what_was_shown(tmp_path):
    frames = script_frames(SCRIPTS / "staircase_clean.json")
    _, _, lesson, gate = lesson_from(frames)
    assert gate.gate.passed
    lesson = Lesson.from_dict(lesson.to_dict())
    journal = Journal("s-roundtrip", "human", directory=tmp_path, clock=lambda: 0.0)
    session = Session(lesson, "human", journal, sleep=lambda s: None)
    session.start()
    stab = Stabilizer()
    for gs in frames:
        board = stab.feed(gs)
        if board is not None:
            session.on_board(board)
    assert session.phase == COMPLETE
    assert session.marks == ["pass"] * len(lesson.steps)
    verdicts = [e["data"]["verdict"] for e in read_journal(journal.path) if e["kind"] == "verdict"]
    assert "PASS" in verdicts and not set(verdicts) - {"PASS", "MISSING"}


UNREADABLE = [seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"]}),
              seg({"A1": ["red"], "B2": ["purple"]}, s=1.0, hands_clear=False),
              seg({"A1": ["red"], "B2": ["unknown"]})]


def test_unknown_color_still_makes_its_step_and_only_fails_colors_known(tmp_path):
    ex, rows, lesson, gate = lesson_from(frames_of(UNREADABLE))
    assert [(r.cell, r.level, r.color) for r in rows] == [("A1", 0, "red"), ("B2", 0, "unknown")]
    assert failed(gate) == ["colors_known"]
    out = write_outputs(ex, rows, lesson, gate, tmp_path / "test-lesson.json", {})
    assert out.lesson_path is None and not (tmp_path / "test-lesson.json").exists()
    record = json.loads(out.extract_path.read_text())
    assert record["colors_file_template"] == [{"step": 2, "cell": "B2", "color": ""}]
    assert record["draft_lesson"]["steps"][1]["color"] == "unknown"


def test_colors_file_fills_unknown_and_gate_passes(tmp_path):
    ex, rows, lesson, gate = lesson_from(frames_of(UNREADABLE), [{"step": 2, "cell": "B2", "color": "orange"}])
    assert [(r.color, r.source) for r in rows] == [("red", "camera"), ("orange", "manual")]
    assert gate.gate.passed, gate.reasons
    out = write_outputs(ex, rows, lesson, gate, tmp_path / "test-lesson.json", {})
    assert Lesson.from_dict(json.loads(out.lesson_path.read_text())).steps[1].color == "orange"


def test_trust_order_manual_then_robot_memory_then_camera():
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["yellow"]})]))
    robot = {"A1": "unknown", "B2": "orange", "C3": "purple"}   # the robot memory read per cell
    rows = step_rows(ex, {3: "teal"}, lambda frame_id, cell: robot[cell])
    assert [(r.color, r.source) for r in rows] == [("red", "camera"), ("orange", "robot_memory"), ("teal", "manual")]
    assert [(r.camera_color, r.robot_color, r.manual_color) for r in rows] == [
        ("red", "unknown", None), ("unknown", "orange", None), ("yellow", "purple", "teal")]


def test_flat_mode_reads_a_tall_cell_as_one_block():
    segments = [seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "C1": ["unknown", "yellow"]})]  # C1 misread 2-high
    _, _, lesson, gate = lesson_from(frames_of(segments))
    assert placed(lesson) == [("red", "A1", 0), ("unknown", "C1", 0), ("yellow", "C1", 1)]   # as before, no --flat
    log = FlatLog()
    _, _, lesson, gate = lesson_from(flat_frames(frames_of(segments), log))
    assert placed(lesson) == [("red", "A1", 0), ("yellow", "C1", 0)]
    assert gate.gate.passed, gate.reasons
    assert [(r["cell"], r["perception_said"], r["t_first"]) for r in log.runs] == [("C1", ["unknown", "yellow"], 3.0)]


def test_colors_file_with_wrong_cell_is_refused():
    with pytest.raises(ColorsFileError, match="step 2 is on B2"):
        lesson_from(frames_of(UNREADABLE), [{"step": 2, "cell": "C3", "color": "orange"}])
