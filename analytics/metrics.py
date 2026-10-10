"""Metrics over the loader's rows. Pure functions: no I/O, no provenance filtering.

The report decides which sessions go in. Every rate comes back as {"n", "passes", "rate"},
and rate is None when n is 0, never 0.0. Results are always grouped by lesson_id first,
because step 2 of one lesson is not step 2 of another.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from analytics.load import VALUES, Attempt, SessionRow, TaskStep
from shared.config import CELLS
from shared.schemas import Verdict

VERDICTS = [v.value for v in Verdict if v is not Verdict.WAITING]  # WAITING is never an attempt
PASS = VALUES["pass"]
COMPLETE, ROBOT_STOPPED = "complete", "robot_stopped"


# ---------------------------------------------------------------- helpers


def rate(passes: int, n: int) -> dict[str, Any]:
    return {"n": n, "passes": passes, "rate": None if n == 0 else round(passes / n, 4)}


def _mean(xs: list[float]) -> Optional[float]:
    return round(statistics.fmean(xs), 3) if xs else None


def _median(xs: list[float]) -> Optional[float]:
    return round(statistics.median(xs), 3) if xs else None


def _sort_key(key: tuple) -> tuple:
    return tuple((v is None, str(v)) for v in key)


def _group(rows: Iterable[Any], fields: tuple[str, ...]) -> dict[tuple, list[Any]]:
    """Rows grouped by lesson_id plus fields, in a stable order."""
    fields = ("lesson_id",) + tuple(f for f in fields if f != "lesson_id")
    groups: dict[tuple, list[Any]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] if isinstance(r, dict) else getattr(r, f) for f in fields)].append(r)
    return {k: groups[k] for k in sorted(groups, key=_sort_key)}


def _labels(key: tuple, fields: tuple[str, ...]) -> dict[str, Any]:
    fields = ("lesson_id",) + tuple(f for f in fields if f != "lesson_id")
    return dict(zip(fields, key))


def _by_session_step(attempts: Iterable[Attempt]) -> dict[tuple, list[Attempt]]:
    """{(session, step): attempts in attempt_no order}."""
    out: dict[tuple, list[Attempt]] = defaultdict(list)
    for a in attempts:
        out[(a.session, a.step)].append(a)
    for tries in out.values():
        tries.sort(key=lambda a: a.attempt_no)
    return out


def _step_summary(tries_per_session: list[list[Attempt]]) -> dict[str, Any]:
    """First-try rate, mean attempts to pass, and median seconds to pass over session-steps."""
    passed = [tries for tries in tries_per_session if any(a.passed for a in tries)]
    to_pass = [next(i for i, a in enumerate(tries, 1) if a.passed) for tries in passed]
    seconds = [next(a for a in tries if a.passed).seconds_since_step_start for tries in passed]
    return {
        **rate(sum(tries[0].passed for tries in tries_per_session), len(tries_per_session)),
        "n_passed": len(passed),
        "mean_attempts": _mean(to_pass),          # attempts until the pass, over session-steps that passed
        "median_seconds_to_pass": _median(seconds),
    }


# ---------------------------------------------------------------- metrics


def first_try_pass_rate(attempts: list[Attempt], by: tuple[str, ...] = ("mode",)) -> list[dict]:
    """Share of session-steps whose first attempt passed, per lesson_id and `by`."""
    firsts = [a for a in attempts if a.attempt_no == 1]
    return [
        {**_labels(k, by), **rate(sum(a.passed for a in rows), len(rows))}
        for k, rows in _group(firsts, by).items()
    ]


def step_difficulty(attempts: list[Attempt]) -> list[dict]:
    """Per (lesson_id, step, mode): first-try n/passes/rate, mean attempts to pass, median seconds to pass."""
    out = []
    for k, rows in _group(attempts, ("step", "mode")).items():
        out.append({**_labels(k, ("step", "mode")), **_step_summary(list(_by_session_step(rows).values()))})
    return out


def verdict_mix(attempts: list[Attempt], by: tuple[str, ...] = ("mode",)) -> list[dict]:
    """Count and share of each verdict, per lesson_id and `by`. Every verdict is always listed."""
    out = []
    for k, rows in _group(attempts, by).items():
        counts = Counter(a.verdict for a in rows)
        n = len(rows)
        out.append({
            **_labels(k, by),
            "n": n,
            "passes": counts[PASS],
            "counts": {v: counts[v] for v in VERDICTS},
            "shares": {v: None if n == 0 else round(counts[v] / n, 4) for v in VERDICTS},
        })
    return out


def cell_error_map(attempts: list[Attempt], mode: str) -> dict[str, dict[str, dict]]:
    """{lesson_id: {cell: ...}} for one mode, with all 9 cells (brain naming) per lesson.
    n/passes/fails/rate count attempts whose verdict expected_cell is that cell (the slot that
    needed a cube); fault_count counts attempts whose fault_cell is that cell."""
    out: dict[str, dict[str, dict]] = {}
    for (lesson_id,), rows in _group([a for a in attempts if a.mode == mode], ()).items():
        cells = {}
        for cell in CELLS:
            expected = [a for a in rows if a.expected_cell == cell]
            passes = sum(a.passed for a in expected)
            cells[cell] = {
                **rate(passes, len(expected)),
                "fails": len(expected) - passes,
                "fault_count": sum(a.fault_cell == cell for a in rows),
            }
        out[lesson_id] = cells
    return out


def hint_effectiveness(attempts: list[Attempt], by: tuple[str, ...] = ("failed_verdict", "hint_level")) -> list[dict]:
    """After a failed attempt, the share of next attempts (same session and step) that pass.
    hint_level is the hint showing at that next attempt. Robot attempts carry no hint level,
    so they drop out. `by` may use failed_verdict, hint_level, step, mode."""
    pairs = []
    for tries in _by_session_step(attempts).values():
        for failed, nxt in zip(tries, tries[1:]):
            if failed.passed or nxt.hint_level_before is None:
                continue
            pairs.append({
                "lesson_id": failed.lesson_id, "step": failed.step, "mode": failed.mode,
                "failed_verdict": failed.verdict, "hint_level": nxt.hint_level_before, "passed": nxt.passed,
            })
    return [
        {**_labels(k, by), **rate(sum(p["passed"] for p in rows), len(rows))}
        for k, rows in _group(pairs, by).items()
    ]


def robot_retry_stats(attempts: list[Attempt], sessions: list[SessionRow]) -> list[dict]:
    """Per lesson_id, over robot session-steps: passed on try 1, after 1 retry, after 2 retries,
    stopped (no pass and the session ended robot_stopped), or unfinished (no pass otherwise).
    stops_by_reason counts robot_stopped sessions by session_end reason, step numbers folded."""
    outcome = {s.session: s.outcome for s in sessions}
    buckets = ("pass_try_1", "pass_after_1_retry", "pass_after_2_retries", "stopped", "unfinished")
    steps_by_lesson: dict[Any, Counter] = defaultdict(Counter)
    for tries in _by_session_step(a for a in attempts if a.mode == "robot").values():
        passed = next((a for a in tries if a.passed), None)
        if passed is None:
            bucket = "stopped" if outcome.get(tries[0].session) == ROBOT_STOPPED else "unfinished"
        else:
            retries = passed.retry if passed.retry is not None else passed.attempt_no - 1
            bucket = buckets[min(retries, 2)]
        steps_by_lesson[tries[0].lesson_id][bucket] += 1
    reasons: dict[Any, Counter] = defaultdict(Counter)
    for s in sessions:
        if s.mode == "robot" and s.outcome == ROBOT_STOPPED:
            reasons[s.lesson_id][re.sub(r"\bstep \d+", "step N", s.end_reason or "unknown")] += 1
    out = []
    for lesson_id in sorted(set(steps_by_lesson) | set(reasons), key=lambda v: _sort_key((v,))):
        counts = steps_by_lesson[lesson_id]
        n = sum(counts.values())
        out.append({
            "lesson_id": lesson_id,
            "n": n,
            **{b: rate(counts[b], n) for b in buckets},
            "stops_by_reason": dict(sorted(reasons[lesson_id].items())),
        })
    return out


def leaderboard(sessions: list[SessionRow], attempts: list[Attempt]) -> list[dict]:
    """Human sessions with outcome "complete", ranked per lesson by total seconds, then mistakes."""
    by_session: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        by_session[a.session].append(a)
    rows = []
    for s in sessions:
        if s.mode != "human" or s.outcome != COMPLETE or s.t_end is None:
            continue
        tries = by_session[s.session]
        rows.append({
            "lesson_id": s.lesson_id,
            "session": s.session,
            "provenance": s.provenance,
            "total_seconds": round(s.t_end - s.t_start, 3),
            "mistakes": sum(not a.passed for a in tries),
            "max_hint_level": max((a.hint_level_before or 0 for a in tries), default=0),
        })
    out = []
    for _, group in _group(rows, ()).items():
        group.sort(key=lambda r: (r["total_seconds"], r["mistakes"], r["session"]))
        out += [{"rank": i, **r} for i, r in enumerate(group, 1)]
    return out


def robot_trials(task_rows: list[TaskStep]) -> dict[str, Any]:
    """Robot task report steps, per cell in ROBOT naming (letter = row). Never merge these
    with brain cell names. A step whose ok was never recorded counts as unknown, not a fail."""

    def summary(rows: list[TaskStep]) -> dict[str, Any]:
        known = [r for r in rows if r.ok is not None]
        errors = [r.error_mm for r in rows if r.error_mm is not None]
        return {
            **rate(sum(r.ok is True for r in known), len(known)),
            "unknown": len(rows) - len(known),
            "mean_error_mm": _mean(errors),
            "max_error_mm": max(errors) if errors else None,
            "mean_seconds": _mean([r.seconds for r in rows if r.seconds is not None]),
        }

    by_cell: dict[Optional[str], list[TaskStep]] = defaultdict(list)
    for r in task_rows:
        by_cell[r.cell].append(r)
    return {
        "cell_naming": "robot",
        "cells": [{"cell": c, **summary(by_cell[c])} for c in sorted(by_cell, key=lambda c: _sort_key((c,)))],
        "overall": summary(list(task_rows)),
    }


def hints_file_effect(sessions: list[SessionRow], attempts: list[Attempt]) -> list[dict]:
    """Human sessions that loaded a hints file versus ones that did not, per (lesson_id, step).
    Empty until session_start carries the hints file hash (see load.FIELDS)."""
    if not any(s.hints_file_hash for s in sessions):
        return []
    with_file = {s.session for s in sessions if s.hints_file_hash}
    human = [a for a in attempts if a.mode == "human"]
    out = []
    for k, rows in _group(human, ("step",)).items():
        tries = _by_session_step(rows)
        out.append({
            **_labels(k, ("step",)),
            "with_file": _step_summary([t for (sid, _), t in tries.items() if sid in with_file]),
            "without_file": _step_summary([t for (sid, _), t in tries.items() if sid not in with_file]),
        })
    return out


def gate_stats(sessions: list[SessionRow]) -> list[dict]:
    """Per lesson_id: sessions, and how many ran a lesson that had a gate (n) and passed it."""
    return [
        {
            **_labels(k, ()),
            "sessions": len(rows),
            **rate(sum(s.gate_passed is True for s in rows), sum(s.gate_passed is not None for s in rows)),
        }
        for k, rows in _group(sessions, ()).items()
    ]
