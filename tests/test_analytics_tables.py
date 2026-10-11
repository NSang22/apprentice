import json
import shutil
from pathlib import Path

from analytics.databricks.tables import LAYERS, PY_TYPES, SCHEMAS, as_tuples, columns, gold, hint_policy_rows, journal_raw_rows, silver
from analytics.fake_journal import generate, write_journals
from analytics.load import DEFAULT_LESSON_DIRS, load_all
from brain.settings import DEMO_LESSON
from shared.schemas import Lesson

FIXTURES = Path(__file__).parent / "fixtures"


def test_every_row_matches_its_declared_schema(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for p in FIXTURES.glob("brain_journal_*.jsonl"):
        shutil.copy(p, sessions / p.name)
    write_journals(generate(Lesson.from_dict(json.loads(DEMO_LESSON.read_text())), seed=7), sessions)
    t = load_all(sessions, DEFAULT_LESSON_DIRS)

    rows = {"journal_raw": [r for p in sorted(sessions.glob("*.jsonl")) for r in journal_raw_rows(p.name, p.read_text())]}
    rows.update(silver(t))
    rows.update(gold(t, include_synthetic=True))
    rows["hint_policy"] = hint_policy_rows(
        {"lesson_id": "L", "generated_from": {"sessions": 6, "attempts": 27},
         "steps": [{"step": 1, "watch_out": "Most people put this cube on B2. It goes on A2.", "start_hint_level": 2},
                   {"step": 3, "watch_out": None, "start_hint_level": 2}]}, "ab" * 32)

    assert set(SCHEMAS) == set(rows) == {name for names in LAYERS.values() for name in names}
    for table, table_rows in rows.items():
        cols = columns(table)
        assert cols[0] == ("provenance", "STRING")
        assert all(typ in PY_TYPES for _, typ in cols)
        for row in table_rows:
            assert list(row) == [name for name, _ in cols], table
            for name, typ in cols:
                v = row[name]
                assert v is None or (type(v) is PY_TYPES[typ]), (table, name, v)  # exact type: no bool as int, no nesting
        assert all(len(tup) == len(cols) for tup in as_tuples(table, table_rows))

    # Tables that must have rows with this data (hints_file_effect stays empty: no hints file hash yet).
    for table in ("journal_raw", "sessions", "attempts", "hints", "first_try_by_mode", "step_difficulty", "verdict_mix",
                  "cell_error_map", "hint_effectiveness", "leaderboard", "robot_retry_stats", "hint_policy"):
        assert rows[table], table
    assert rows["hints_file_effect"] == []
    assert {r["provenance"] for r in rows["attempts"]} == {"simulation", "synthetic"}
    # The cut-off line stays in bronze as written.
    assert len(rows["journal_raw"]) == sum(len([l for l in p.read_text().splitlines() if l.strip()])
                                           for p in sessions.glob("*.jsonl"))
