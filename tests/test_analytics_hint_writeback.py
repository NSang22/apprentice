import hashlib
import json
from dataclasses import replace

from analytics.hint_writeback import build_hints, write_hints
from analytics.load import Attempt, SessionRow

P, W, C = "PASS", "WRONG_CELL", "WRONG_COLOR"
TARGET = {0: "A1", 1: "A2", 2: "C1"}  # lesson L


def att(session, step, no, verdict, hint, fault=None):
    return Attempt(session=session, mode="human", provenance="real", lesson_id="L", step=step,
                   step_color="blue", step_cell=TARGET[step], step_level=0, verdict=verdict,
                   fault_cell=fault, fault_level=None if fault is None else 0,
                   expected_cell=TARGET[step], expected_level=0, attempt_no=no,
                   hint_level_before=hint, retry=None, t=0.0, seconds_since_step_start=1.0)


def sess(session, side="row3"):
    return SessionRow(session=session, mode="human", provenance="real", path="", lesson_id="L", n_steps=3,
                      grid_source="perception:ws://jetson:8765", robot="none", commit=None, dirty=None,
                      hints_file_hash=None, gate_passed=None, t_start=0.0, t_end=60.0, outcome="complete",
                      end_reason=None, steps_completed=3, skipped_lines=0, human_side=side)


SESSIONS = [sess(f"S{i}") for i in range(1, 7)]
ATTEMPTS = [att(f"S{i}", 0, 1, P, 0) for i in range(1, 7)]  # step 0: everyone right first time
ATTEMPTS += [  # step 1 (target A2): most people put it on B2
    att("S1", 1, 1, W, 0, "B2"), att("S1", 1, 2, W, 1, "B2"), att("S1", 1, 3, P, 2),
    att("S2", 1, 1, W, 0, "B2"), att("S2", 1, 2, W, 1, "B2"), att("S2", 1, 3, P, 2),
    att("S3", 1, 1, W, 0, "B2"), att("S3", 1, 2, P, 1),
    att("S4", 1, 1, W, 0, "B2"), att("S4", 1, 2, W, 1, "B2"), att("S4", 1, 3, P, 2),
    att("S5", 1, 1, P, 0),
    att("S6", 1, 1, C, 0, "A2"), att("S6", 1, 2, P, 1),
]
ATTEMPTS += [att("S1", 2, 1, W, 0, "C2"), att("S1", 2, 2, P, 1)] + [att(f"S{i}", 2, 1, P, 0) for i in range(2, 7)]


def test_hints_file_from_real_data_only(tmp_path):
    # Step 1, min_n 3. Failed attempts: S1 2, S2 2, S3 1, S4 2 (all WRONG_CELL on B2) + S6 1 color = 8.
    #   Top mistake (WRONG_CELL, B2) = 7 >= 3 and 7 >= 8/2 -> watch_out. From the row3 side, A2 -> B2 is
    #   one column toward C = "one cell right" (brain/checker.py direction()).
    #   First try 1/6 = 0.17 < 0.5. Level 1 next tries: S1 W, S2 W, S3 P, S4 W, S6 P -> 2/5 = 0.4, n 5 >= 3.
    #   Level 2 next tries: S1 P, S2 P, S4 P -> 3/3 = 1.0, n 3 >= 3. Gain 0.6 >= 0.25 -> start at level 2.
    # Step 0: no mistakes, first try 6/6 -> no entry. Step 2: 1 failed attempt < 3 -> no entry.
    hints = build_hints("L", SESSIONS, ATTEMPTS, min_n=3)
    assert hints == {
        "lesson_id": "L",
        "generated_from": {"sessions": 6, "attempts": 6 + 14 + 7},
        "steps": [{"step": 1, "watch_out": "Most people put this cube one cell right of its spot.", "start_hint_level": 2}],
    }

    # Sessions that stood on different sides: no safe direction, so name the cells.
    mixed = [replace(s, human_side="row1") if s.session == "S6" else s for s in SESSIONS]
    assert build_hints("L", mixed, ATTEMPTS, min_n=3)["steps"][0]["watch_out"] == "Most people put this in B2. It goes in A2."

    # Below min_n: 8 failed attempts < 9 and level 1 has 5 < 9 next tries -> no step qualifies, no file.
    assert build_hints("L", SESSIONS, ATTEMPTS, min_n=9) is None
    # Not real: the same rows as simulation or synthetic never produce a file.
    for p in ("simulation", "synthetic", "unverified"):
        assert build_hints("L", [replace(s, provenance=p) for s in SESSIONS],
                           [replace(a, provenance=p) for a in ATTEMPTS], min_n=3) is None

    path, digest = write_hints(hints, tmp_path)
    assert path == tmp_path / "L.hints.json"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = json.loads(path.read_text())
    assert set(loaded) == {"lesson_id", "generated_from", "steps"}
    assert set(loaded["generated_from"]) == {"sessions", "attempts"}
    assert all(set(s) == {"step", "watch_out", "start_hint_level"} and s["start_hint_level"] in (1, 2)
               for s in loaded["steps"])
