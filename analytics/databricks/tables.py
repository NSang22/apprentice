"""Flat rows for the Delta tables, built from the loader's and metrics' output. Pure Python, no Spark.

One implementation: the attempt rules live in analytics/load.py and the numbers in
analytics/metrics.py (through analytics/report.bucket). This module only flattens them.

Every table has an explicit DDL schema in SCHEMAS. Nothing is ever inferred, because a column
that is all None breaks inference. conform() checks each row against its schema and casts
ints to floats for DOUBLE columns (Spark rejects an int in a DOUBLE column).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional

from analytics.load import Attempt, HintShown, SessionRow, Tables, parse_journal_lines, provenance
from analytics.report import ALWAYS, bucket

LAYERS = {
    "bronze": ["journal_raw"],
    "silver": ["sessions", "attempts", "hints"],
    "gold": ["first_try_by_mode", "step_difficulty", "verdict_mix", "cell_error_map", "hint_effectiveness",
             "hints_file_effect", "leaderboard", "robot_retry_stats", "hint_policy"],
}

_RATE = "n BIGINT, passes BIGINT, rate DOUBLE"
_STEP = f"{_RATE}, n_passed BIGINT, n_not_passed BIGINT, mean_attempts DOUBLE, median_seconds_to_pass DOUBLE"

SCHEMAS: dict[str, str] = {
    # bronze. provenance is the file's; null for a journal that never started.
    "journal_raw": "provenance STRING, source_file STRING, line_no BIGINT, line STRING",
    # silver
    "sessions": (
        "provenance STRING, session STRING, mode STRING, path STRING, lesson_id STRING, n_steps BIGINT, "
        "grid_source STRING, robot STRING, human_side STRING, git_commit STRING, git_dirty BOOLEAN, "
        "hints_file_hash STRING, gate_passed BOOLEAN, t_start DOUBLE, t_end DOUBLE, outcome STRING, "
        "end_reason STRING, steps_completed BIGINT, skipped_lines BIGINT"
    ),
    "attempts": (
        "provenance STRING, session STRING, mode STRING, lesson_id STRING, step BIGINT, step_color STRING, "
        "step_cell STRING, step_level BIGINT, verdict STRING, passed BOOLEAN, fault_cell STRING, "
        "fault_level BIGINT, expected_cell STRING, expected_level BIGINT, attempt_no BIGINT, "
        "hint_level_before BIGINT, retry BIGINT, t DOUBLE, seconds_since_step_start DOUBLE"
    ),
    "hints": "provenance STRING, session STRING, lesson_id STRING, step BIGINT, hint_level BIGINT, t DOUBLE",
    # gold
    "first_try_by_mode": f"provenance STRING, lesson_id STRING, mode STRING, {_RATE}",
    "step_difficulty": f"provenance STRING, lesson_id STRING, step BIGINT, mode STRING, {_STEP}",
    "verdict_mix": "provenance STRING, lesson_id STRING, mode STRING, verdict STRING, n BIGINT, verdict_count BIGINT, share DOUBLE",
    "cell_error_map": (
        f"provenance STRING, mode STRING, lesson_id STRING, cell STRING, grid_col STRING, grid_row STRING, "
        f"{_RATE}, fails BIGINT, fault_count BIGINT"
    ),
    "hint_effectiveness": f"provenance STRING, lesson_id STRING, failed_verdict STRING, hint_level BIGINT, {_RATE}",
    "hints_file_effect": f"provenance STRING, lesson_id STRING, step BIGINT, hints_file STRING, {_STEP}",
    "leaderboard": (
        "provenance STRING, lesson_id STRING, rank BIGINT, session STRING, total_seconds DOUBLE, "
        "mistakes BIGINT, max_hint_level BIGINT"
    ),
    "robot_retry_stats": "provenance STRING, lesson_id STRING, outcome STRING, n BIGINT, steps BIGINT, rate DOUBLE",
    "hint_policy": (
        "provenance STRING, lesson_id STRING, step BIGINT, watch_out STRING, start_hint_level BIGINT, "
        "sha256 STRING, generated_sessions BIGINT, generated_attempts BIGINT"
    ),
}

PY_TYPES = {"STRING": str, "BIGINT": int, "DOUBLE": float, "BOOLEAN": bool}


def columns(table: str) -> list[tuple[str, str]]:
    """[(name, type)] parsed from the table's DDL string."""
    out = []
    for part in SCHEMAS[table].split(","):
        name, typ = part.split()
        out.append((name, typ))
    return out


def conform(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """The row with exactly the schema's keys, in order, and values of the declared types.
    Raises on a missing or extra key, a nested value, or a wrong type: a bug, not data."""
    cols = columns(table)
    names = [n for n, _ in cols]
    if set(row) != set(names):
        raise ValueError(f"{table}: keys {sorted(set(row) ^ set(names))} do not match the schema")
    out = {}
    for name, typ in cols:
        v = row[name]
        if v is not None:
            if typ == "DOUBLE" and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            want = PY_TYPES[typ]
            if not isinstance(v, want) or (want is int and isinstance(v, bool)):
                raise TypeError(f"{table}.{name}: expected {typ}, got {type(v).__name__} {v!r}")
        out[name] = v
    return out


def as_tuples(table: str, rows: Iterable[dict[str, Any]]) -> list[tuple]:
    """Rows as tuples in schema column order, for spark.createDataFrame(..., schema=SCHEMAS[table])."""
    names = [n for n, _ in columns(table)]
    return [tuple(r[n] for n in names) for r in rows]


# ---------------------------------------------------------------- bronze and silver


def journal_raw_rows(source_file: str, text: str) -> list[dict]:
    """Every non-empty line of one journal file, as written, with the file's provenance."""
    lines = text.splitlines()
    events, _ = parse_journal_lines(lines)
    start = next((e for e in events if e["kind"] == "session_start"), None)
    prov = None if start is None else provenance(start, events)
    return [
        conform("journal_raw", {"provenance": prov, "source_file": source_file, "line_no": i, "line": line})
        for i, line in enumerate(lines, 1) if line.strip()
    ]


_SESSION_RENAME = {"commit": "git_commit", "dirty": "git_dirty"}


def session_rows(sessions: list[SessionRow]) -> list[dict]:
    return [conform("sessions", {_SESSION_RENAME.get(k, k): v for k, v in asdict(s).items()}) for s in sessions]


def attempt_rows(attempts: list[Attempt]) -> list[dict]:
    return [conform("attempts", {**asdict(a), "passed": a.passed}) for a in attempts]


def hint_rows(hints: list[HintShown]) -> list[dict]:
    return [conform("hints", asdict(h)) for h in hints]


def silver(t: Tables) -> dict[str, list[dict]]:
    return {"sessions": session_rows(t.sessions), "attempts": attempt_rows(t.attempts), "hints": hint_rows(t.hints)}


# ---------------------------------------------------------------- gold


def _gold_bucket(p: str, b: dict[str, Any]) -> dict[str, list[dict]]:
    """Flatten one provenance bucket from analytics.report.bucket()."""
    out: dict[str, list[dict]] = {name: [] for name in LAYERS["gold"] if name != "hint_policy"}
    out["first_try_by_mode"] = [{"provenance": p, **r} for r in b["first_try_pass_rate"]]
    out["step_difficulty"] = [{"provenance": p, **r} for r in b["step_difficulty"]]
    for r in b["verdict_mix"]:
        for v, count in r["counts"].items():
            out["verdict_mix"].append({"provenance": p, "lesson_id": r["lesson_id"], "mode": r["mode"], "verdict": v,
                                       "n": r["n"], "verdict_count": count, "share": r["shares"][v]})
    for mode, lessons in b["cell_error_map"].items():
        for lesson_id, cells in lessons.items():
            for cell, c in cells.items():
                out["cell_error_map"].append({"provenance": p, "mode": mode, "lesson_id": lesson_id, "cell": cell,
                                              "grid_col": cell[0], "grid_row": cell[1], **c})
    out["hint_effectiveness"] = [{"provenance": p, **r} for r in b["hint_effectiveness"]]
    for r in b["hints_file_effect"]:
        for side, label in (("with_file", "with"), ("without_file", "without")):
            out["hints_file_effect"].append({"provenance": p, "lesson_id": r["lesson_id"], "step": r["step"],
                                             "hints_file": label, **r[side]})
    out["leaderboard"] = [{**r, "provenance": p} for r in b["leaderboard"]]
    for r in b["robot_retry_stats"]:
        for outcome in ("pass_try_1", "pass_after_1_retry", "pass_after_2_retries", "stopped", "unfinished"):
            out["robot_retry_stats"].append({"provenance": p, "lesson_id": r["lesson_id"], "outcome": outcome,
                                             "n": r["n"], "steps": r[outcome]["passes"], "rate": r[outcome]["rate"]})
    return {name: [conform(name, row) for row in rows] for name, rows in out.items()}


def gold(t: Tables, include_synthetic: bool = False) -> dict[str, list[dict]]:
    """Every gold metric table except hint_policy, computed per provenance, never pooled."""
    names = list(ALWAYS) + (["synthetic"] if include_synthetic else [])
    out: dict[str, list[dict]] = {name: [] for name in LAYERS["gold"] if name != "hint_policy"}
    for p in names:
        for name, rows in _gold_bucket(p, bucket(t.only([p]))).items():
            out[name] += rows
    return out


def hint_policy_rows(hints: Optional[dict[str, Any]], sha256: Optional[str]) -> list[dict]:
    """One row per step of a hints file from analytics.hint_writeback (always real data)."""
    if not hints:
        return []
    g = hints.get("generated_from", {})
    return [
        conform("hint_policy", {
            "provenance": "real", "lesson_id": hints["lesson_id"], "step": s["step"], "watch_out": s["watch_out"],
            "start_hint_level": s["start_hint_level"], "sha256": sha256,
            "generated_sessions": g.get("sessions"), "generated_attempts": g.get("attempts"),
        })
        for s in hints["steps"]
    ]
