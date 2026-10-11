import json
from pathlib import Path

import pytest

from brain.extractor import (ColorsFileError, build_lesson, extract, gate_lesson, model_fallback, manual_colors,
                             restore_hidden, script_frames, split_transients, step_rows, write_outputs)
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


def stairs(boards):
    """Each board held, then a hand over the grid, like tests/scripts/staircase_clean.json."""
    out = [seg({}, 2.0), seg({}, 1.0, hands_clear=False)]
    for b in boards:
        out += [seg(b), seg(b, 1.0, hands_clear=False)]
    return out + [seg(boards[-1], 3.0)]


# The staircase with one cube of each color (the fixture's repeats now read as conflicting).
DISTINCT = stairs([{"A1": ["red"]}, {"A1": ["red"], "B1": ["teal"]}, {"A1": ["red"], "B1": ["teal"], "C1": ["skyblue"]},
                   {"A1": ["red", "purple"], "B1": ["teal"], "C1": ["skyblue"]},
                   {"A1": ["red", "purple"], "B1": ["teal", "orange"], "C1": ["skyblue"]},
                   {"A1": ["red", "purple", "yellow"], "B1": ["teal", "orange"], "C1": ["skyblue"]}])
DISTINCT_STEPS = [("red", "A1", 0), ("teal", "B1", 0), ("skyblue", "C1", 0), ("purple", "A1", 1),
                  ("orange", "B1", 1), ("yellow", "A1", 2)]


def test_staircase_gives_stacked_steps_from_history():
    _, _, lesson, gate = lesson_from(frames_of(DISTINCT))
    assert placed(lesson) == DISTINCT_STEPS   # levels 0, 1 and 2, no flag
    times = [s.t_video for s in lesson.steps]
    assert times == sorted(times) and times[0] > 0
    assert gate.gate.passed, gate.reasons


def test_expert_correction_leaves_one_step():
    # skyblue goes on C2 first, then moves to C1
    mistake = stairs([{"A1": ["red"]}, {"A1": ["red"], "B1": ["teal"]}, {"A1": ["red"], "B1": ["teal"], "C2": ["skyblue"]},
                      {"A1": ["red"], "B1": ["teal"], "C1": ["skyblue"]},
                      {"A1": ["red", "purple"], "B1": ["teal"], "C1": ["skyblue"]}])
    ex, _, lesson, gate = lesson_from(frames_of(mistake))
    assert placed(lesson) == DISTINCT_STEPS[:4]
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
    frames = frames_of(DISTINCT)
    _, _, lesson, gate = lesson_from(frames)
    assert gate.gate.passed, gate.reasons
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
    assert placed(lesson) == [("red", "A1", 0)] and {w["kept"] for w in ex.weak_reads} == {"red"}
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


def test_model_fallback_accepts_and_rejects():
    # A1 red from the camera; B2, C3, B1 never read. A fake vision model: purple (taken), red (already on A1), none.
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["unknown"]}),
                            seg({"A1": ["red"], "B2": ["unknown"], "C3": ["unknown"], "B1": ["unknown"]})]))
    replies = iter(["purple", "red", "none"])
    rows = step_rows(ex)
    records = model_fallback(rows, ex, lambda frame_id, cell: b"jpeg", lambda jpeg: {"answer": next(replies)}, {})
    assert [(r.cell, r.color, r.source) for r in rows] == [
        ("A1", "red", "camera"), ("B2", "purple", "model"), ("C3", "unknown", "none"), ("B1", "unknown", "none")]
    assert [(g["answer"], g["accepted"]) for g in records] == [("purple", True), ("red", False), ("none", False)]
    gate = gate_lesson(build_lesson(rows, "test-lesson", "Test", None), ex, rows=rows)
    assert gate.notes["colors_known"] == ["step 2: model, unverified (purple)"]
    assert "B2: model, unverified (purple)" in gate.notes["replay_matches_final_frame"]
    assert set(failed(gate)) == {"replay_matches_final_frame", "colors_known"}   # C3 and B1 stay unknown
    # a rerun reuses the saved answers and never calls vision model
    saved = {g["key"]: {"answer": g["answer"]} for g in records}
    rows = step_rows(ex)
    model_fallback(rows, ex, lambda frame_id, cell: b"jpeg", lambda jpeg: pytest.fail("called the vision model"), saved)
    assert [r.color for r in rows] == ["red", "purple", "unknown", "unknown"]


def test_hand_band_ignores_stacks_and_sees_a_hand():
    import numpy as np
    from brain.video_source import HAND_BAND_MAX, band_change, hand_band
    from perception.change import SHEET_PX_PER_MM, sheet_size_px
    from perception.cells import cell_center_mm, grid_geometry
    from robot.perception.sheet import SheetPose
    # camera 430 mm over the sheet, 2 image px per mm, as a straight-down view
    band = hand_band(SheetPose(np.diag([0.5, 0.5, 1.0]), [0, 1, 2, 3], 0.0, np.array([108.0, 220.0, 430.0])), 48.0)
    _, _, size = grid_geometry()
    for cell in ("A1", "B2", "C3"):   # a cell and the block on it are never in the band
        x, y = (np.array(cell_center_mm(cell)) * SHEET_PX_PER_MM).astype(int)
        assert not band[y, x]
    w, h = sheet_size_px()
    empty = np.full((h, w, 3), 235, np.uint8)
    arm = empty.copy()
    arm[band.any(axis=1).nonzero()[0].min():, : w // 2] = (40, 40, 40)   # a dark sleeve over the band's left half
    assert band_change(empty, empty, band) == 0.0
    assert band_change(arm, empty, band) > HAND_BAND_MAX


def test_backstop_skips_a_board_that_changes_three_cells():
    ex, _, lesson, gate = lesson_from(frames_of([
        seg({}), seg({"A1": ["red"]}),
        seg({"A1": ["red"], "A2": ["unknown"], "A3": ["unknown"], "B1": ["unknown"]}),   # a sleeve read as blocks
        seg({"A1": ["red"], "B2": ["teal"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("teal", "B2", 0)]
    assert [k["cells"] for k in ex.skipped] == [["A2", "A3", "B1"]]
    assert gate.gate.passed, gate.reasons


def test_new_cube_on_an_empty_cell_beats_a_color_change():
    # B2 yellow appears while A1's top also reads yellow: B2 is the cube, A1 keeps red (a conflicting read)
    ex, _, lesson, gate = lesson_from(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["yellow"], "B2": ["yellow"]}),
                                                 seg({"A1": ["red"], "B2": ["yellow"]})]))
    assert placed(lesson) == [("red", "A1", 0), ("yellow", "B2", 0)]
    assert [(c["cell"], c["kept"], c["read"]) for c in ex.conflicts] == [("A1", "red", "yellow")]
    assert gate.gate.passed, gate.reasons


def test_new_cube_is_named_among_colors_not_on_the_board():
    # B2 sits 8 from skyblue and 12 from teal: unclear among all colors, but skyblue is already on A1
    robot = {"A1": "skyblue", "B2": "unknown"}
    dist = {"A1": {"skyblue": 2.0, "teal": 20.0, "yellow": 100.0}, "B2": {"skyblue": 8.0, "teal": 12.0, "yellow": 100.0}}
    ex = extract(frames_of([seg({}), seg({"A1": ["unknown"]}), seg({"A1": ["unknown"], "B2": ["unknown"]})]),
                 robot_read=lambda f, cell: robot[cell], robot_distance=lambda f, cell: dist[cell])
    rows = step_rows(ex)
    assert [(r.cell, r.color) for r in rows] == [("A1", "skyblue"), ("B2", "teal")]


def test_cube_hidden_by_an_arm_keeps_its_step():
    # red on A1, teal on B2; an arm hides A1 (it reads empty), then red is back on A1
    ex = extract(frames_of([seg({}), seg({"A1": ["red"]}), seg({"A1": ["red"], "B2": ["teal"]}),
                            seg({"B2": ["teal"]}), seg({"A1": ["red"], "B2": ["teal"]})]))
    split_transients(ex)
    restore_hidden(ex)
    rows = step_rows(ex)
    lesson = build_lesson(rows, "test-lesson", "Test", None)
    gate = gate_lesson(lesson, ex, rows=rows)
    assert [(s.color, s.cell, s.level, s.t_video) for s in lesson.steps] == [("red", "A1", 0, 1.5), ("teal", "B2", 0, 3.0)]
    assert ex.removed == [] and [(h["cell"], h["color"]) for h in ex.hidden] == [("A1", "red")]
    assert gate.gate.passed, gate.reasons


def test_two_cubes_at_once_pass_when_one_is_later_removed():
    ex = extract(frames_of([seg({}), seg({"A1": ["red"], "B2": ["teal"]}), seg({"A1": ["red"]})]))
    split_transients(ex)
    rows = step_rows(ex)
    lesson = build_lesson(rows, "test-lesson", "Test", None)
    gate = gate_lesson(lesson, ex, rows=rows)
    assert placed(lesson) == [("red", "A1", 0)] and [(c.cell, c.color()) for c in ex.removed] == [("B2", "teal")]
    assert gate.gate.passed, gate.reasons
    assert "B2 L0 was later removed" in gate.notes["one_cube_per_change"][0]


def test_colors_file_with_wrong_cell_is_refused():
    with pytest.raises(ColorsFileError, match="step 2 is on B2"):
        lesson_from(frames_of(UNREADABLE), [{"step": 2, "cell": "C3", "color": "orange"}])
