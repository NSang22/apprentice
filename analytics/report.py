"""Write every metric to one JSON file for the UI and the pitch. Offline, after sessions.

    python -m analytics.report --sessions data/sessions --out data/analytics/metrics.json
    python -m analytics.report --sessions data/sessions --out data/analytics/metrics.json --tasks robot/outputs/tasks
    python -m analytics.report ... --include-synthetic     # adds the synthetic bucket, flagged

Numbers are never pooled across provenance. Each provenance gets its own block under
"by_provenance", always present for real, simulation and unverified, empty when it has no
sessions. Only "real" is judge-facing. The shape is in analytics/METRICS_SCHEMA.md.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from analytics import metrics as m
from analytics.load import DEFAULT_LESSON_DIRS, PROVENANCES, Tables, load_all
from shared.config import REPO_ROOT, SESSIONS_DIR

SCHEMA_VERSION = 1
DEFAULT_OUT = REPO_ROOT / "data" / "analytics" / "metrics.json"
ALWAYS = ("real", "simulation", "unverified")  # buckets written even when empty
LABELS = {
    "real": "Real camera. Judge-facing.",
    "simulation": "Simulation: fake feed, simulated grid, or fake robot. Not judge-facing.",
    "unverified": "Older journals with no grid_source; source unknown. Not judge-facing.",
    "synthetic": "Synthetic journals from analytics/fake_journal.py. Never judge-facing.",
}


def bucket(t: Tables) -> dict[str, Any]:
    """Every metric over one provenance's rows."""
    a, s = t.attempts, t.sessions
    return {
        "sessions": len(s),
        "attempts": len(a),
        "first_try_pass_rate": m.first_try_pass_rate(a),
        "step_difficulty": m.step_difficulty(a),
        "verdict_mix": m.verdict_mix(a),
        "cell_error_map": {mode: m.cell_error_map(a, mode) for mode in ("human", "robot")},
        "hint_effectiveness": m.hint_effectiveness(a),
        "robot_retry_stats": m.robot_retry_stats(a, s),
        "leaderboard": m.leaderboard(s, a),
        "hints_file_effect": m.hints_file_effect(s, a),
        "gate_stats": m.gate_stats(s),
    }


def build_report(t: Tables, include_synthetic: bool = False, now: Optional[datetime] = None) -> dict[str, Any]:
    counts = Counter(s.provenance for s in t.sessions)
    names = list(ALWAYS) + (["synthetic"] if include_synthetic else [])
    by_provenance = {p: {"label": LABELS[p], "judge_facing": p == "real", **bucket(t.only([p]))} for p in names}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "sessions": {p: counts.get(p, 0) for p in PROVENANCES},  # everything on disk, included or not
        "included_provenances": names,
        "never_started": t.never_started,
        "skipped_lines": t.skipped_lines,
        "bad_lesson_files": t.bad_lessons,
        "lessons_not_found": t.unknown_lessons,
        "contains_synthetic_data": include_synthetic and counts.get("synthetic", 0) > 0,
        "contains_simulation_data": counts.get("simulation", 0) > 0,
        "contains_unverified_data": counts.get("unverified", 0) > 0,
        "by_provenance": by_provenance,
        "robot_trials": {
            **m.robot_trials(t.task_steps),
            "tested_against_real_file": False,  # no real robot task report fixture yet
            "task_steps": len(t.task_steps),
        },
    }


def write_json_atomic(path: Path, obj: Any) -> bytes:
    """Write to a temp file in the same folder, then rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return data


def _pct(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{r['rate']:.0%}"


def summary(report: dict[str, Any]) -> str:
    s = report["sessions"]
    lines = ["sessions: " + ", ".join(f"{p} {s[p]}" for p in PROVENANCES)
             + ("" if "synthetic" in report["included_provenances"] else " (synthetic left out)")]
    for name, b in report["by_provenance"].items():
        if not b["sessions"]:
            lines.append(f"{name}: no sessions")
            continue
        for r in b["first_try_pass_rate"]:
            lines.append(f"{name}: {r['lesson_id']} {r['mode']} first try {r['passes']}/{r['n']} ({_pct(r)})")
        for r in b["robot_retry_stats"]:
            lines.append(f"{name}: {r['lesson_id']} robot stopped {r['stopped']['passes']}/{r['n']} steps")
    t = report["robot_trials"]
    if t["task_steps"]:
        o = t["overall"]
        lines.append(f"robot trials (robot cell naming, not tested against a real file): ok {o['passes']}/{o['n']} ({_pct(o)})")
    lines.append(f"skipped lines {report['skipped_lines']}, never started {report['never_started']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tasks", type=Path, default=None, help="robot/outputs/tasks, optional")
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args(argv)

    tables = load_all(args.sessions, DEFAULT_LESSON_DIRS, args.tasks)
    report = build_report(tables, args.include_synthetic)
    write_json_atomic(args.out, report)
    print(summary(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
