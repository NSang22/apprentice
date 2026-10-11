import json
import shutil
from pathlib import Path

from analytics.fake_journal import generate, write_journals
from analytics.load import DEFAULT_LESSON_DIRS, load_all, provenance, read_journal
from brain.settings import DEMO_LESSON
from shared.schemas import Lesson

FIXTURES = Path(__file__).parent / "fixtures"
HUMAN, ROBOT = "s-20261010-223829", "s-20261010-223808"  # session ids inside the brain fixtures


def test_loader_matches_the_brain_on_its_own_journals(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for p in FIXTURES.glob("brain_journal_*.jsonl"):
        shutil.copy(p, sessions / p.name)
    # A crash mid-write, and a session that never left the "ready" phase (board lines only).
    with open(sessions / "brain_journal_human.jsonl", "a", encoding="utf-8") as f:
        f.write('{"t": 1791650000.0, "session": "s-20261010-223829", "ki')
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
    checked = 0
    for path in sessions.glob("*.jsonl"):
        for e in read_journal(path)[0]:
            if e["kind"] == "step_advance":
                assert first[(e["session"], e["data"]["from_step"])] == e["data"]["first_try"], (path.name, e)
                checked += 1
    assert checked > 50

    # Hand-checked against the fixture lines.
    human = [a for a in t.attempts if a.session == HUMAN]
    assert len(human) == 7  # 6 passes + 1 mistake; the 6 "still placing" checks are skipped
    assert [(a.verdict, a.attempt_no, a.hint_level_before, a.fault_cell) for a in human if a.step == 2] == [
        ("WRONG_CELL", 1, 0, "C2"), ("PASS", 2, 1, None)]
    assert (human[0].step_color, human[0].step_cell, human[0].step_level) == ("red", "A1", 0)  # tests/fixtures lesson
    robot = [a for a in t.attempts if a.session == ROBOT]
    assert len(robot) == 6  # the first "still placing" check comes before any place, so it is skipped
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
