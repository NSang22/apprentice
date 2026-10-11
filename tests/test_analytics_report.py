import json
import shutil
from pathlib import Path

from analytics.fake_journal import generate, write_journals
from analytics.report import main
from brain.settings import DEMO_LESSON
from shared.schemas import Lesson

FIXTURES = Path(__file__).parent / "fixtures"


def test_report_keeps_provenance_apart(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for p in FIXTURES.glob("brain_journal_*.jsonl"):
        shutil.copy(p, sessions / p.name)  # 2 brain sessions with FakeRobot: simulation
    write_journals(generate(Lesson.from_dict(json.loads(DEMO_LESSON.read_text())), seed=7), sessions)  # 16 synthetic
    out = tmp_path / "analytics" / "metrics.json"

    main(["--sessions", str(sessions), "--out", str(out)])
    report = json.loads(out.read_text())

    assert set(report) == {
        "schema_version", "generated_at", "sessions", "included_provenances", "never_started", "skipped_lines",
        "bad_lesson_files", "lessons_not_found", "lesson_mismatch", "contains_synthetic_data", "contains_simulation_data",
        "contains_unverified_data", "by_provenance", "robot_trials"}
    assert report["sessions"] == {"real": 0, "unverified": 0, "simulation": 2, "synthetic": 16}
    assert report["lesson_mismatch"] == []
    assert set(report["by_provenance"]) == {"real", "simulation", "unverified"}  # synthetic left out by default
    real = report["by_provenance"]["real"]
    assert (real["sessions"], real["attempts"], real["judge_facing"]) == (0, 0, True)
    assert real["first_try_pass_rate"] == real["leaderboard"] == real["robot_retry_stats"] == []
    assert real["cell_error_map"] == {"human": {}, "robot": {}}
    assert report["by_provenance"]["simulation"]["sessions"] == 2
    assert (report["contains_synthetic_data"], report["contains_simulation_data"], report["contains_unverified_data"]) == (
        False, True, False)
    assert report["robot_trials"]["cell_naming"] == "robot"
    assert report["robot_trials"]["tested_against_real_file"] is False
    assert list(out.parent.iterdir()) == [out]  # the temp file was renamed away

    main(["--sessions", str(sessions), "--out", str(out), "--include-synthetic"])
    report = json.loads(out.read_text())
    assert report["contains_synthetic_data"] is True
    assert report["by_provenance"]["synthetic"]["sessions"] == 16
    assert report["by_provenance"]["simulation"]["sessions"] == 2  # never pooled
