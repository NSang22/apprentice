"""Every metric on a tiny hand-built data set, with the arithmetic in comments."""

from dataclasses import replace

from analytics.load import Attempt, SessionRow, TaskStep
from analytics.metrics import (
    cell_error_map,
    first_try_pass_rate,
    gate_stats,
    hint_effectiveness,
    hints_file_effect,
    leaderboard,
    robot_retry_stats,
    robot_trials,
    step_difficulty,
    verdict_mix,
)

P, W, C, M = "PASS", "WRONG_CELL", "WRONG_COLOR", "MISSING"


def att(session, step, no, verdict, cell, fault=None, hint=None, retry=None, secs=0.0, mode="human", lesson="L"):
    return Attempt(session=session, mode=mode, provenance="real", lesson_id=lesson, step=step,
                   step_color="red", step_cell=cell, step_level=0, verdict=verdict,
                   fault_cell=fault, fault_level=None if fault is None else 0,
                   expected_cell=cell, expected_level=0, attempt_no=no,
                   hint_level_before=hint, retry=retry, t=0.0, seconds_since_step_start=secs)


def sess(session, mode, outcome, t_start, t_end, gate=None, reason=None, lesson="L"):
    return SessionRow(session=session, mode=mode, provenance="real", path="", lesson_id=lesson, n_steps=3,
                      grid_source="perception:ws://jetson:8765", robot="none", commit=None, dirty=None,
                      hints_file_hash=None, gate_passed=gate, t_start=t_start, t_end=t_end,
                      outcome=outcome, end_reason=reason, steps_completed=0, skipped_lines=0)


# Lesson L targets A1, A2, C1 (steps 0, 1, 2). Lesson M has one step on B3.
ATTEMPTS = [
    att("H1", 0, 1, P, "A1", hint=0, secs=5),
    att("H1", 1, 1, W, "A2", fault="B2", hint=0, secs=4),
    att("H1", 1, 2, P, "A2", hint=1, secs=10),
    att("H1", 2, 1, C, "C1", fault="C1", hint=0, secs=3),
    att("H1", 2, 2, W, "C1", fault="B1", hint=1, secs=8),
    att("H1", 2, 3, P, "C1", hint=2, secs=20),
    att("H2", 0, 1, P, "A1", hint=0, secs=7),
    att("H2", 1, 1, P, "A2", hint=0, secs=6),
    att("H2", 2, 1, W, "C1", fault="B1", hint=0, secs=4),
    att("H2", 2, 2, P, "C1", hint=1, secs=9),
    att("H3", 0, 1, W, "A1", fault="A2", hint=0, secs=6),  # then reset
    att("R1", 0, 1, P, "A1", retry=0, secs=12, mode="robot"),
    att("R1", 1, 1, W, "A2", fault="B2", retry=0, secs=15, mode="robot"),
    att("R1", 1, 2, P, "A2", retry=1, secs=30, mode="robot"),
    att("R1", 2, 1, W, "C1", fault="B1", retry=0, secs=14, mode="robot"),
    att("R1", 2, 2, M, "C1", retry=1, secs=33, mode="robot"),  # a place that failed
    att("R1", 2, 3, P, "C1", retry=2, secs=50, mode="robot"),
    att("R2", 0, 1, P, "A1", retry=0, secs=14, mode="robot"),
    att("R2", 1, 1, W, "A2", fault="B2", retry=0, mode="robot"),
    att("R2", 1, 2, W, "A2", fault="B2", retry=1, mode="robot"),
    att("R2", 1, 3, W, "A2", fault="B2", retry=2, mode="robot"),
    att("H4", 0, 1, P, "B3", hint=0, secs=5, lesson="M"),
]
SESSIONS = [
    sess("H1", "human", "complete", 0, 100, gate=True),
    sess("H2", "human", "complete", 200, 260, gate=True),
    sess("H3", "human", "reset", 300, 330),
    sess("R1", "robot", "complete", 400, 520, gate=False),
    sess("R2", "robot", "robot_stopped", 600, 700, reason="step 2 still WRONG_CELL after 2 retries"),
    sess("H4", "human", "complete", 0, 30, lesson="M"),
]
TASKS = [
    TaskStep("t1.json", 1.0, "done", 1, "red -> cell C1", "C1", 20.0, 4.0, True),
    TaskStep("t1.json", 1.0, "stopped", 2, "blue -> cell C1", "C1", 25.0, 22.0, False),
    TaskStep("t2.json", 2.0, "done", 1, "red -> cell A1", "A1", 18.0, 6.0, True),
    TaskStep("t2.json", 2.0, "done", 2, "blue -> right of red", None, None, None, None),  # ok never recorded
]


def r(n, passes, rate):
    return {"n": n, "passes": passes, "rate": rate}


def test_every_metric_by_hand():
    # First attempts. L human: H1 s0 P, s1 W, s2 C; H2 s0 P, s1 P, s2 W; H3 s0 W -> 3/7 = 0.4286.
    # L robot: R1 s0 P, s1 W, s2 W; R2 s0 P, s1 W -> 2/5 = 0.4. M human: 1/1.
    assert first_try_pass_rate(ATTEMPTS) == [
        {"lesson_id": "L", "mode": "human", **r(7, 3, 0.4286)},
        {"lesson_id": "L", "mode": "robot", **r(5, 2, 0.4)},
        {"lesson_id": "M", "mode": "human", **r(1, 1, 1.0)},
    ]

    # Per session-step: mean attempts until the pass, median seconds of the passing attempt.
    # L s0 human: H1 [P@5], H2 [P@7], H3 [W] -> first try 2/3; passed 2; mean (1+1)/2 = 1; median(5,7) = 6.
    # L s1 human: H1 [W, P@10], H2 [P@6] -> 1/2; mean (2+1)/2 = 1.5; median(10,6) = 8.
    # L s2 human: H1 [C, W, P@20], H2 [W, P@9] -> 0/2 (rate 0.0, n is 2); mean (3+2)/2 = 2.5; median 14.5.
    # L s0 robot: R1 [P@12], R2 [P@14] -> 2/2; mean 1; median 13.
    # L s1 robot: R1 [W, P@30], R2 [W, W, W] -> 0/2; passed 1; mean 2; median 30.
    # L s2 robot: R1 [W, M, P@50] -> 0/1; mean 3; median 50.
    def sd(step, mode, n, passes, rate, n_passed, mean, median, lesson="L"):
        return {"lesson_id": lesson, "step": step, "mode": mode, **r(n, passes, rate),
                "n_passed": n_passed, "mean_attempts": mean, "median_seconds_to_pass": median}
    assert step_difficulty(ATTEMPTS) == [
        sd(0, "human", 3, 2, 0.6667, 2, 1.0, 6.0),
        sd(0, "robot", 2, 2, 1.0, 2, 1.0, 13.0),
        sd(1, "human", 2, 1, 0.5, 2, 1.5, 8.0),
        sd(1, "robot", 2, 0, 0.0, 1, 2.0, 30.0),
        sd(2, "human", 2, 0, 0.0, 2, 2.5, 14.5),
        sd(2, "robot", 1, 0, 0.0, 1, 3.0, 50.0),
        sd(0, "human", 1, 1, 1.0, 1, 1.0, 5.0, lesson="M"),
    ]

    # L human: 11 attempts = H1 (P W P C W P) + H2 (P P W P) + H3 (W) -> P 6, W 4, C 1.
    # Shares 6/11 = 0.5455, 4/11 = 0.3636, 1/11 = 0.0909.
    # L robot: 10 = R1 (P W P W M P) + R2 (P W W W) -> P 4, W 5, M 1.
    mix = {(m["lesson_id"], m["mode"]): m for m in verdict_mix(ATTEMPTS)}
    human = mix[("L", "human")]
    assert (human["n"], human["passes"]) == (11, 6)
    assert human["counts"] == {"PASS": 6, "WRONG_CELL": 4, "WRONG_COLOR": 1, "WRONG_LEVEL": 0, "EXTRA": 0, "MISSING": 0}
    assert human["shares"]["PASS"] == 0.5455 and human["shares"]["WRONG_CELL"] == 0.3636
    assert human["shares"]["WRONG_COLOR"] == 0.0909 and human["shares"]["EXTRA"] == 0.0
    robot = mix[("L", "robot")]
    assert (robot["n"], robot["counts"]["WRONG_CELL"], robot["counts"]["MISSING"]) == (10, 5, 1)
    assert set(mix) == {("L", "human"), ("L", "robot"), ("M", "human")}

    # Human, lesson L, by expected cell: A1 = H1 P, H2 P, H3 W -> 2/3, fails 1.
    # A2 = H1 W, H1 P, H2 P -> 2/3. C1 = H1 C W P, H2 W P -> 2/5 = 0.4, fails 3.
    # fault_cell: B2 once (H1 s1), C1 once (H1 color), B1 twice (H1 s2, H2 s2), A2 once (H3).
    cells = cell_error_map(ATTEMPTS, "human")
    assert set(cells) == {"L", "M"} and all(len(c) == 9 for c in cells.values())
    L = cells["L"]
    assert L["A1"] == {**r(3, 2, 0.6667), "fails": 1, "fault_count": 0}
    assert L["A2"] == {**r(3, 2, 0.6667), "fails": 1, "fault_count": 1}
    assert L["C1"] == {**r(5, 2, 0.4), "fails": 3, "fault_count": 1}
    assert L["B1"] == {**r(0, 0, None), "fails": 0, "fault_count": 2}
    assert L["B2"] == {**r(0, 0, None), "fails": 0, "fault_count": 1}
    assert L["C3"] == {**r(0, 0, None), "fails": 0, "fault_count": 0}
    assert cells["M"]["B3"] == {**r(1, 1, 1.0), "fails": 0, "fault_count": 0}

    # Failed attempt -> next attempt on the same step, keyed by the hint showing at that next try:
    # H1 s1 W -> P (hint 1); H1 s2 C -> W (hint 1); H1 s2 W -> P (hint 2); H2 s2 W -> P (hint 1).
    # H3 has no next attempt. Robot attempts carry no hint level, so they drop out.
    # (W, 1): 2/2. (W, 2): 1/1. (C, 1): 0/1.
    assert hint_effectiveness(ATTEMPTS) == [
        {"lesson_id": "L", "failed_verdict": W, "hint_level": 1, **r(2, 2, 1.0)},
        {"lesson_id": "L", "failed_verdict": W, "hint_level": 2, **r(1, 1, 1.0)},
        {"lesson_id": "L", "failed_verdict": C, "hint_level": 1, **r(1, 0, 0.0)},
    ]

    # Robot session-steps in L: R1 s0 try 1, R1 s1 after 1 retry, R1 s2 after 2 retries,
    # R2 s0 try 1, R2 s1 never passed and R2 ended robot_stopped -> n 5: 2/5, 1/5, 1/5, 1/5, 0/5.
    assert robot_retry_stats(ATTEMPTS, SESSIONS) == [{
        "lesson_id": "L", "n": 5,
        "pass_try_1": r(5, 2, 0.4), "pass_after_1_retry": r(5, 1, 0.2), "pass_after_2_retries": r(5, 1, 0.2),
        "stopped": r(5, 1, 0.2), "unfinished": r(5, 0, 0.0),
        "stops_by_reason": {"step N still WRONG_CELL after 2 retries": 1},
    }]

    # Complete human sessions only (H3 reset, robots out). L: H2 260-200 = 60 s, 1 mistake, max hint 1;
    # H1 100 s, 3 mistakes (W, C, W), max hint 2. M: H4 30 s.
    assert leaderboard(SESSIONS, ATTEMPTS) == [
        {"rank": 1, "lesson_id": "L", "session": "H2", "provenance": "real", "total_seconds": 60, "mistakes": 1, "max_hint_level": 1},
        {"rank": 2, "lesson_id": "L", "session": "H1", "provenance": "real", "total_seconds": 100, "mistakes": 3, "max_hint_level": 2},
        {"rank": 1, "lesson_id": "M", "session": "H4", "provenance": "real", "total_seconds": 30, "mistakes": 0, "max_hint_level": 0},
    ]

    # Robot naming cells. C1: 1 of 2 ok, error (4+22)/2 = 13, max 22, seconds (20+25)/2 = 22.5.
    # A1: 1/1, 6 mm, 18 s. The "beside" goal has no cell and no ok: unknown, not a fail.
    # Overall: 2 of 3 known, error (4+22+6)/3 = 10.667, max 22, seconds (20+25+18)/3 = 21.
    trials = robot_trials(TASKS)
    assert trials["cell_naming"] == "robot"
    assert trials["cells"] == [
        {"cell": "A1", **r(1, 1, 1.0), "unknown": 0, "mean_error_mm": 6.0, "max_error_mm": 6.0, "mean_seconds": 18.0},
        {"cell": "C1", **r(2, 1, 0.5), "unknown": 0, "mean_error_mm": 13.0, "max_error_mm": 22.0, "mean_seconds": 22.5},
        {"cell": None, **r(0, 0, None), "unknown": 1, "mean_error_mm": None, "max_error_mm": None, "mean_seconds": None},
    ]
    assert trials["overall"] == {**r(3, 2, 0.6667), "unknown": 1, "mean_error_mm": 10.667, "max_error_mm": 22.0, "mean_seconds": 21.0}

    # No session carries a hints file hash yet, so nothing to compare.
    assert hints_file_effect(SESSIONS, ATTEMPTS) == []
    # Once one does (H2 loaded a file): L s1 with = H2 [P@6] -> 1/1, mean 1, median 6;
    # without = H1 [W, P@10] -> 0/1, mean 2, median 10. M s0 with nobody -> n 0, rate None.
    with_hash = [replace(s, hints_file_hash="abc") if s.session == "H2" else s for s in SESSIONS]
    effect = {(e["lesson_id"], e["step"]): e for e in hints_file_effect(with_hash, ATTEMPTS)}
    assert effect[("L", 1)]["with_file"] == {**r(1, 1, 1.0), "n_passed": 1, "mean_attempts": 1.0, "median_seconds_to_pass": 6.0}
    assert effect[("L", 1)]["without_file"] == {**r(1, 0, 0.0), "n_passed": 1, "mean_attempts": 2.0, "median_seconds_to_pass": 10.0}
    assert effect[("M", 0)]["with_file"] == {**r(0, 0, None), "n_passed": 0, "mean_attempts": None, "median_seconds_to_pass": None}

    # L: 5 sessions; 3 had a gate (H1 pass, H2 pass, R1 fail) -> 2/3. M: H4 had none -> n 0, rate None.
    assert gate_stats(SESSIONS) == [
        {"lesson_id": "L", "sessions": 5, **r(3, 2, 0.6667)},
        {"lesson_id": "M", "sessions": 1, **r(0, 0, None)},
    ]

    # Empty input: empty results and None rates, never an exception or a divide by zero.
    assert first_try_pass_rate([]) == step_difficulty([]) == verdict_mix([]) == hint_effectiveness([]) == []
    assert cell_error_map([], "human") == {}
    assert robot_retry_stats([], []) == leaderboard([], []) == hints_file_effect([], []) == gate_stats([]) == []
    assert robot_trials([]) == {"cell_naming": "robot", "cells": [], "overall": {
        **r(0, 0, None), "unknown": 0, "mean_error_mm": None, "max_error_mm": None, "mean_seconds": None}}
