import json
from pathlib import Path

import pytest

from brain.extractor import (ColorsFileError, build_lesson, extract, gate_lesson, gemini_fallback, manual_colors,
                             script_frames, split_transients, step_rows, write_outputs)
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
    return ex, rows, lesson, gate_lesson(lesson, ex, rows=rows)


def placed(lesson):
    return [(s.color, s.cell, s.level) for s in lesson.steps]


def failed(gate):
    return [c.name for c in gate.gate.checks if not c.passed]


def test_staircase_gives_stacked_steps_from_history():
    _, _, lesson, gate = lesson_from(script_frames(SCRIPTS / "staircase_clean.json"))
    assert placed(lesson) == placed(STAIRCASE)   # levels 0, 1 and 2, no flag
    times = [s.t_video for s in lesson.steps]
    assert times == sorted(times) and times[0] > 0
    assert failed(gate) == ["one_cube_per_color"]   # the staircase uses two skyblue and two red


def test_expert_correction_leaves_one_step():
    # skyblue goes on C2 first, then moves to C1
    ex, _, lesson, gate = lesson_from(script_frames(SCRIPTS / "staircase_with_mistake.json"))
    assert placed(lesson) == placed(STAIRCASE)
    assert [(c.cell, c.level) for c in ex.removed] == [("C2", 0)]
    assert failed(gate) == ["one_cube_per_color"]


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
    assert failed(gate) == ["one_cube_per_color"]
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


def test_unknown_color_still_makes_its_step(tmp_path):
    ex, rows, lesson, gate = lesson_from(frames_of(UNREADABLE))
    assert [(r.cell, r.level, r.color) for r in rows] == [("A1", 0, "red"), ("B2", 0, "unknown")]
    assert failed(gate) == ["replay_matches_final_frame", "colors_known"]   # the last board can't read B2 either
    out = write_outputs(ex, rows, lesson, gate, tmp_path / "test-lesson.json", {})
    assert out.lesson_path is None and not (tmp_path / "test-lesson.json").exists()
    record = json.loads(out.extract_path.read_text())
    assert record["colors_file_template"] == [{"step": 2, "cell": "B2", "color": ""}]
    assert record["draft_lesson"]["steps"][1]["color"] == "unknown"


def test_colors_file_fills_unknown_and_gate_passes(tmp_path):
    ex, rows, lesson, gate = lesson_from(frames_of(UNREADABLE), [{"step": 2, "cell": "B2", "color": "orange"}])
    assert [(r.color, r.source) for r in rows] == [("red", "camera"), ("orange", "manual")]
    assert gate.gate.passed, gate.reasons   # the manual color stands in for the unreadable final read
    assert gate.notes["replay_matches_final_frame"] == ["B2: manual, unverified (orange)"]
    out = write_outputs(ex, rows, lesson, gate, tmp_path / "test-lesson.json", {})
    assert Lesson.from_dict(json.loads(out.lesson_path.read_text())).steps[1].color == "orange"


def test_trust_order_manual_then_robot_memory_then_camera():
    robot = {"A1": "unknown", "B2": "orange", "C3": "purple"}   # the robot memory read per cell
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["yellow"]})]),
                 robot_read=lambda frame_id, cell: robot[cell])
    rows = step_rows(ex, {3: "teal"})
    assert [(r.color, r.source) for r in rows] == [("red", "camera"), ("orange", "robot_memory"), ("teal", "manual")]
    assert [(r.camera_color, r.robot_color, r.manual_color) for r in rows] == [
        ("red", "unknown", None), ("unknown", "orange", None), ("yellow", "purple", "teal")]


def test_single_cube_the_camera_reads_two_high_stays_level_0():
    # the C1 case from expert_take1: perception says two high, sometimes one; one cube was placed
    _, _, lesson, gate = lesson_from(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "C1": ["unknown", "yellow"]}),
                                                seg({"A1": ["red"], "C1": ["yellow"]}),
                                                seg({"A1": ["red"], "C1": ["yellow", "yellow"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("yellow", "C1", 0)]
    assert gate.gate.passed, gate.reasons


def test_cube_put_on_a_stack_and_taken_off_is_a_correction():
    ex, _, lesson, gate = lesson_from(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red", "purple"]}),
                                                 seg({"A1": ["red", "purple", "teal"]}), seg({"A1": ["red", "purple"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("purple", "A1", 1)]
    assert [(c.color(), c.cell, c.level) for c in ex.removed] == [("teal", "A1", 2)]
    assert gate.gate.passed, gate.reasons


def test_two_cubes_of_one_color_fail_one_cube_per_color():
    _, _, lesson, gate = lesson_from(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "C3": ["red"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("red", "C3", 0)]
    assert failed(gate) == ["one_cube_per_color"]


def test_unread_cube_stacked_on_a_known_top_is_a_step():
    # A1 red, then a cube on it no source can read: a draft step at level 1, which a colors file can name
    stacked = [seg({}), seg({"A1": ["red"]}), seg({"A1": ["red", "unknown"]}), seg({"A1": ["red", "unknown"]})]
    ex, rows, lesson, gate = lesson_from(frames_of(stacked))
    assert placed(lesson) == [("red", "A1", 0), ("unknown", "A1", 1)]
    assert "colors_known" in failed(gate)
    _, rows, _, _ = lesson_from(frames_of(stacked), [{"step": 2, "cell": "A1", "color": "teal"}])
    assert [(r.color, r.source) for r in rows] == [("red", "camera"), ("teal", "manual")]
    # the top goes unknown and the old top comes back: that unknown cube was a transient
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red", "unknown"]}), seg({"A1": ["red"]})]))
    split_transients(ex)
    assert [(c.cell, c.level) for c in ex.steps] == [("A1", 0)]
    assert [(c.cell, c.level) for c in ex.transients] == [("A1", 1)] and ex.removed == []


def test_unclear_read_close_to_the_top_keeps_its_color():
    # A1 red; then nothing reads it clearly, but it is 20 from red in the robot memory: a weak read
    frames = frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["unknown"]})])
    robot_read = lambda frame_id, cell: "red" if frames[frame_id].board[cell] == ["red"] else "unknown"
    ex = extract(frames, robot_read=robot_read, robot_distance=lambda frame_id, cell: {"red": 20.0, "orange": 30.0})
    split_transients(ex)
    rows = step_rows(ex)
    lesson = build_lesson(rows, "test-lesson", "Test", None)
    gate = gate_lesson(lesson, ex, rows=rows)
    assert placed(lesson) == [("red", "A1", 0)] and [w["kept"] for w in ex.weak_reads] == ["red"]
    assert gate.gate.passed, gate.reasons
    assert gate.notes["replay_matches_final_frame"] == ["A1: weak read, 20.0 from red (limit 26)"]
    # the same read 40 from red is not close: something else is on A1, an unknown step, and the gate fails
    ex = extract(frames, robot_read=robot_read, robot_distance=lambda frame_id, cell: {"red": 40.0})
    rows = step_rows(ex)
    gate = gate_lesson(build_lesson(rows, "test-lesson", "Test", None), ex, rows=rows)
    assert [(r.color, r.cell, r.level) for r in rows] == [("red", "A1", 0), ("unknown", "A1", 1)]
    assert set(failed(gate)) == {"replay_matches_final_frame", "colors_known"}


def test_object_with_no_known_color_that_leaves_is_a_transient():
    # B2 never reads a color and goes away again (a hand); it appears together with the real teal cube
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"], "C1": ["teal"]}),
                            seg({"A1": ["red"], "C1": ["teal"]})]))
    split_transients(ex)
    assert [(c.cell, c.t_video, c.t_removed) for c in ex.transients] == [("B2", 3.0, 4.5)]
    assert ex.removed == []
    rows = step_rows(ex)
    lesson = build_lesson(rows, "test-lesson", "Test", None)
    gate = gate_lesson(lesson, ex)
    assert placed(lesson) == [("red", "A1", 0), ("teal", "C1", 0)]
    assert gate.gate.passed, gate.reasons


def test_gemini_fallback_accepts_and_rejects():
    # A1 red from the camera; B2, C3, B1 never read. A fake Gemini: purple (taken), red (already on A1), none.
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["unknown"], "B1": ["unknown"]})]))
    replies = iter(["purple", "red", "none"])
    rows = step_rows(ex)
    records = gemini_fallback(rows, ex, lambda frame_id, cell: b"jpeg", lambda jpeg: {"answer": next(replies)}, {})
    assert [(r.cell, r.color, r.source) for r in rows] == [
        ("A1", "red", "camera"), ("B2", "purple", "gemini"), ("C3", "unknown", "none"), ("B1", "unknown", "none")]
    assert [(g["answer"], g["accepted"]) for g in records] == [("purple", True), ("red", False), ("none", False)]
    gate = gate_lesson(build_lesson(rows, "test-lesson", "Test", None), ex, rows=rows)
    assert gate.notes["colors_known"] == ["step 2: gemini, unverified (purple)"]
    assert "B2: gemini, unverified (purple)" in gate.notes["replay_matches_final_frame"]
    assert set(failed(gate)) == {"replay_matches_final_frame", "colors_known"}   # C3 and B1 stay unknown
    # a rerun reuses the saved answers and never calls Gemini
    saved = {g["key"]: {"answer": g["answer"]} for g in records}
    rows = step_rows(ex)
    gemini_fallback(rows, ex, lambda frame_id, cell: b"jpeg", lambda jpeg: pytest.fail("called Gemini"), saved)
    assert [r.color for r in rows] == ["red", "purple", "unknown", "unknown"]


def test_colors_file_with_wrong_cell_is_refused():
    with pytest.raises(ColorsFileError, match="step 2 is on B2"):
        lesson_from(frames_of(UNREADABLE), [{"step": 2, "cell": "C3", "color": "orange"}])
