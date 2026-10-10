import json

import pytest

from brain.journal import KINDS, Journal, read_journal


def make(tmp_path, session_id="s-test", mode="human"):
    return Journal(session_id, mode, directory=tmp_path, clock=lambda: 123.5)


def test_line_shape_matches_readme(tmp_path):
    j = make(tmp_path, mode="robot")
    j.write("verdict", {"step": 3, "verdict": "WRONG_CELL", "fault_cell": "B3", "retry": 1})
    j.close()
    assert read_journal(tmp_path / "s-test.jsonl") == [
        {"t": 123.5, "session": "s-test", "mode": "robot", "kind": "verdict",
         "data": {"step": 3, "verdict": "WRONG_CELL", "fault_cell": "B3", "retry": 1}}
    ]


def test_every_line_is_flushed_before_close(tmp_path):
    j = make(tmp_path)
    j.write("session_start", {"lesson_id": "x"})
    j.write("note")
    lines = (tmp_path / "s-test.jsonl").read_text().splitlines()
    assert [json.loads(l)["kind"] for l in lines] == ["session_start", "note"]
    j.close()


def test_appends_never_rewrite_earlier_lines(tmp_path):
    j = make(tmp_path)
    j.write("note", {"n": 1})
    first = (tmp_path / "s-test.jsonl").read_text()
    j.write("note", {"n": 2})
    j.close()
    assert (tmp_path / "s-test.jsonl").read_text().startswith(first)


def test_all_readme_kinds_accepted(tmp_path):
    with make(tmp_path) as j:
        for kind in KINDS:
            j.write(kind, {})
    assert [l["kind"] for l in read_journal(j.path)] == list(KINDS)


def test_unknown_kind_raises(tmp_path):
    with make(tmp_path) as j, pytest.raises(ValueError):
        j.write("verdicts", {})


def test_bad_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        Journal("s-x", "teacher", directory=tmp_path)


def test_existing_session_id_is_refused(tmp_path):
    make(tmp_path).close()
    with pytest.raises(FileExistsError):
        make(tmp_path)


def test_write_after_close_raises(tmp_path):
    j = make(tmp_path)
    j.close()
    with pytest.raises(ValueError):
        j.write("note")


def test_open_new_picks_a_free_id(tmp_path):
    a = Journal.open_new("human", directory=tmp_path)
    b = Journal.open_new("human", directory=tmp_path)
    assert a.path != b.path and a.path.exists() and b.path.exists()
    a.close()
    b.close()


def test_creates_missing_directory(tmp_path):
    with Journal("s-1", "human", directory=tmp_path / "data" / "sessions") as j:
        j.write("note")
    assert j.path.exists()
