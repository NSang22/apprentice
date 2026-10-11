import json
import shutil
from pathlib import Path

from analytics.fake_journal import generate, write_journals
from analytics.load import DEFAULT_LESSON_DIRS, build_attempts, build_session, load_all, provenance, read_journal
from brain.journal import Journal
from brain.session import Session
from brain.settings import DEMO_LESSON
from shared.schemas import Lesson, empty_board

FIXTURES = Path(__file__).parent / "fixtures"
HUMAN, ROBOT = "s-20261011-003036", "s-20261011-003021"  # session ids inside the brain fixtures


def test_step_back_counts_one_pass_per_step_and_restarts_the_step_clock(tmp_path):
    lesson = Lesson.from_dict(json.loads(DEMO_LESSON.read_text()))  # red A1, teal C1, purple B2, ...
    now = [0.0]
    journal = Journal("s-back", "human", tmp_path, clock=lambda: now[0])
    session = Session(lesson, "human", journal)
    board = empty_board()

    def settle(t, cell, stack):
        now[0] = t
        board[cell] = stack
        session.on_board({c: list(v) for c, v in board.items()})

    session.on_board(empty_board())
    session.start()
    settle(10, "A1", ["red"])
    settle(20, "C1", ["teal"])
    settle(30, "A1", [])            # red comes off: the brain steps back to step 0
    settle(35, "A2", ["red"])       # put back on the wrong cell: a mistake on the revisit
    settle(40, "A2", [])
    settle(40, "A1", ["red"])       # back in place: steps 0 and 1 re-pass
    for i, s in enumerate(lesson.steps[2:], 5):
        settle(10 * i, s.cell, [s.color])
    journal.close()

    events = read_journal(journal.path)[0]
    assert sum(e["kind"] == "step_advance" for e in events) > len(lesson.steps)  # the brain re-advanced
    assert build_session(events, {}).steps_completed == len(lesson.steps)
    attempts = build_attempts(events, {})
    assert [(a.verdict, a.attempt_no, a.seconds_since_step_start) for a in attempts if a.step == 0] == [
        ("PASS", 1, 10.0), ("WRONG_CELL", 2, 5.0)]  # the revisit is timed from the step_back at 30
    assert sorted(a.step for a in attempts if a.passed) == list(range(len(lesson.steps)))


def test_loader_matches_the_brain_on_its_own_journals(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for p in FIXTURES.glob("brain_journal_*.jsonl"):
        shutil.copy(p, sessions / p.name)
    # A crash mid-write, and a session that never left the "ready" phase (board lines only).
    with open(sessions / "brain_journal_human.jsonl", "a", encoding="utf-8") as f:
        f.write('{"t": 1791650000.0, "session": "s-20261011-003036", "ki')
    (sessions / "s-ready.jsonl").write_text(
        '{"t":1.0,"session":"s-ready","mode":"human","kind":"board","data":{"step":0,"board":{}}}\n')
    # Synthetic journals are written by the real brain too, and add a robot place that fails.
    lesson = Lesson.from_dict(json.loads(DEMO_LESSON.read_text()))
    write_journals(generate(lesson, seed=7), sessions)

    t = load_all(sessions, DEFAULT_LESSON_DIRS)

    assert t.never_started == 1
    assert t.skipped_lines == 2  # our cut line plus the generator's one cut-off session
    assert t.unknown_lessons == []
    prov = {s.session: s.provenance for s in t.sessions}
    assert prov[HUMAN] == prov[ROBOT] == "simulation"  # grid_source fake_feed / sim, robot fake
    assert sum(p == "synthetic" for p in prov.values()) == 16

    # The invariant behind the "still placing" skip rule: for every step the brain advanced,
    # the loader's first attempt passed exactly when the brain says first_try.
    first = {(a.session, a.step): a.passed for a in t.attempts if a.attempt_no == 1}
    # Only a step's first advance: after a step_back (a cube moved off an earlier step) the brain
    # advances that step again, and first_try then also reflects the fix made on the revisit.
    checked, seen = 0, set()
    for path in sessions.glob("*.jsonl"):
        for e in read_journal(path)[0]:
            key = (e["session"], e["data"].get("from_step"))
            if e["kind"] == "step_advance" and key not in seen:
                seen.add(key)
                assert first[key] == e["data"]["first_try"], (path.name, e)
                checked += 1
    assert checked > 50
    assert any(e["kind"] == "step_back" and e["mode"] == "human"
               for p in sessions.glob("synthetic_*.jsonl") for e in read_journal(p)[0])  # the invariant sees revisits
    # A re-pass after a step_back is not an attempt, and steps completed never exceed the lesson.
    passes = [(a.session, a.step) for a in t.attempts if a.passed]
    assert len(passes) == len(set(passes))
    for s in t.sessions:
        assert 0 <= s.steps_completed <= s.n_steps
        assert s.outcome != "complete" or s.steps_completed == s.n_steps

    # Hand-checked against the fixture lines.
    human = [a for a in t.attempts if a.session == HUMAN]
    assert len(human) == 7  # 6 passes + 1 mistake; the 6 "still placing" checks are skipped
    assert [(a.verdict, a.attempt_no, a.hint_level_before, a.fault_cell) for a in human if a.step == 2] == [
        ("WRONG_CELL", 1, 0, "C2"), ("PASS", 2, 1, None)]
    assert (human[0].step_color, human[0].step_cell, human[0].step_level) == ("red", "A1", 0)  # tests/fixtures lesson
    robot = [a for a in t.attempts if a.session == ROBOT]
    assert len(robot) == 7  # 6 passes + 1 wrong cell; the first "still placing" check comes before any place, so it is skipped
    assert [(a.verdict, a.retry, a.fault_cell) for a in robot if a.step == 1] == [
        ("WRONG_CELL", 0, "B1"), ("PASS", 1, None)]
    assert robot[0].step_color == "red"  # lessons/sample-flat-v1.json
    # After a failed place, "still placing" is a failed robot attempt, as in the brain.
    assert any(a.mode == "robot" and a.verdict == "MISSING" and a.fault_cell is None for a in t.attempts)

    # Provenance from the brain's grid_source: a session started before the first frame says
    # "unknown:<url>", and the server's later note says what it was. Mixed or unknown is unverified.
    def ev(kind, **data):
        return {"kind": kind, "t": 0.0, "data": data}
    start = ev("session_start", grid_source="unknown:ws://jetson:8765", robot="none")
    assert provenance(start) == "unverified"
    assert provenance(start, [ev("note", grid_source="perception:ws://jetson:8765"), start]) == "real"
    assert provenance(start, [ev("note", grid_source="perception:ws://jetson:8765"), ev("note", grid_source="fake_feed")]) == "unverified"
    assert provenance(ev("session_start", grid_source="perception:ws://jetson:8765", robot="fake")) == "simulation"
