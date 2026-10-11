import json
import random
from pathlib import Path

import pytest

from brain.checker import CheckResult, check, direction, expected_board, slot_diff
from brain.fake_feed import make_frame, parse_script
from brain.settings import HUMAN_SIDES
from brain.stabilizer import Stabilizer
from shared.schemas import Lesson, Verdict, empty_board

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPTS_DIR = Path(__file__).parent / "scripts"

# staircase-v1 steps:
#   0 red A1 L0 | 1 teal B1 L0 | 2 skyblue C1 L0 | 3 skyblue A1 L1 | 4 red B1 L1 | 5 yellow A1 L2
LESSON = Lesson.from_dict(json.loads((FIXTURES_DIR / "staircase-v1.json").read_text()))


def board(**cells):
    return {**empty_board(), **cells}


AFTER_2 = board(A1=["red"], B1=["teal"], C1=["skyblue"])
AFTER_3 = board(A1=["red", "skyblue"], B1=["teal"], C1=["skyblue"])


def assert_result(result, verdict, fault_cell=None, hint_contains=None):
    assert result.verdict is verdict, result
    assert result.fault_cell == fault_cell, result
    if hint_contains:
        assert hint_contains in result.hint_text, result


# ---------------------------------------------------------------- basic verdicts


def test_not_stable_is_waiting():
    assert_result(check(None, LESSON, 2), Verdict.WAITING)


def test_correct_placement_passes():
    assert_result(check(AFTER_2, LESSON, 2), Verdict.PASS)
    assert_result(check(AFTER_3, LESSON, 3), Verdict.PASS)


def test_first_step_on_empty_board_is_missing():
    assert_result(check(empty_board(), LESSON, 0), Verdict.MISSING, hint_contains="Place a red cube on A1")


def test_still_placing_is_missing():
    r = check(board(A1=["red"], B1=["teal"]), LESSON, 2)
    assert_result(r, Verdict.MISSING, hint_contains="Place a skyblue cube on C1")


def test_missing_stacked_step_says_stack():
    assert_result(check(AFTER_2, LESSON, 3), Verdict.MISSING, hint_contains="Stack a skyblue cube on A1")


def test_wrong_cell():
    r = check(board(A1=["red"], B1=["teal"], C2=["skyblue"]), LESSON, 2, human_side="row1")
    assert_result(r, Verdict.WRONG_CELL, "C2", "Close! Move the skyblue cube one cell right.")


def test_wrong_cell_from_row3_side():
    r = check(board(A1=["red"], B1=["teal"], C2=["skyblue"]), LESSON, 2, human_side="row3")
    assert_result(r, Verdict.WRONG_CELL, "C2", "Close! Move the skyblue cube one cell left.")


def test_wrong_cell_picks_the_nearest_candidate():
    r = check(board(A1=["red"], B1=["teal"], A3=["skyblue"], C2=["skyblue"]), LESSON, 2)
    assert_result(r, Verdict.WRONG_CELL, "C2")


def test_wrong_color():
    r = check(board(A1=["red"], B1=["teal"], C1=["yellow"]), LESSON, 2)
    assert_result(r, Verdict.WRONG_COLOR, "C1", "Swap the yellow cube for a skyblue one")


def test_wrong_level():
    # Expected A1 = red, skyblue; the cubes went in upside down.
    r = check(board(A1=["skyblue", "red"], B1=["teal"], C1=["skyblue"]), LESSON, 3)
    assert_result(r, Verdict.WRONG_LEVEL, "A1", "on the second level")


def test_extra_cube():
    r = check(board(A1=["red"], B1=["teal"], C1=["skyblue"], B3=["yellow"]), LESSON, 2)
    assert_result(r, Verdict.EXTRA, "B3", "Please take the yellow block off B3.")


def test_extra_cube_on_top_of_a_correct_stack():
    r = check(board(A1=["red", "teal"], B1=["teal"], C1=["skyblue"]), LESSON, 2)
    assert_result(r, Verdict.EXTRA, "A1")


def test_wrong_color_in_wrong_cell_is_extra():
    r = check(board(A1=["red"], B1=["teal"], C3=["yellow"]), LESSON, 2)
    assert_result(r, Verdict.EXTRA, "C3")


# ---------------------------------------------------------------- older cubes


def test_knocked_cube_elsewhere_is_wrong_cell():
    # Step 3 placed correctly, but the teal cube from step 1 slid from B1 to B2.
    r = check(board(A1=["red", "skyblue"], B2=["teal"], C1=["skyblue"]), LESSON, 3, human_side="row1")
    assert_result(r, Verdict.WRONG_CELL, "B2", "Put it back one cell right, on B1")
    r = check(board(A1=["red", "skyblue"], B2=["teal"], C1=["skyblue"]), LESSON, 3, human_side="row3")
    assert_result(r, Verdict.WRONG_CELL, "B2", "Put it back one cell left, on B1")


def test_knocked_off_the_grid_is_missing_with_fault_cell():
    r = check(board(A1=["red", "skyblue"], C1=["skyblue"]), LESSON, 3)
    assert_result(r, Verdict.MISSING, "B1", "missing from B1")


def test_older_cube_swapped_color():
    r = check(board(A1=["red", "skyblue"], B1=["yellow"], C1=["skyblue"]), LESSON, 3)
    assert_result(r, Verdict.WRONG_COLOR, "B1", "should be teal")


def test_newest_step_is_reported_before_older_damage():
    # Newest skyblue went to C2 AND the teal cube got knocked off: fix the newest first.
    r = check(board(A1=["red"], C2=["skyblue"]), LESSON, 2)
    assert_result(r, Verdict.WRONG_CELL, "C2")


def test_check_only_reads_its_inputs():
    b = board(A1=["red"], B1=["teal"], C2=["skyblue"])
    before = json.dumps(b, sort_keys=True)
    check(b, LESSON, 2)
    assert json.dumps(b, sort_keys=True) == before


# ---------------------------------------------------------------- helpers and hints


@pytest.mark.parametrize(
    "src,dst,row1_text,row3_text",
    [
        ("A1", "A2", "one cell left", "one cell right"),          # numbers are columns
        ("A2", "A1", "one cell right", "one cell left"),
        ("A1", "B1", "one cell away from you", "one cell toward you"),  # letters are rows, A far
        ("B1", "A1", "one cell toward you", "one cell away from you"),
        ("C3", "A1", "two cells right and two cells toward you", "two cells left and two cells away from you"),
    ],
)
def test_direction_both_sides(src, dst, row1_text, row3_text):
    assert direction(src, dst, "row1") == row1_text
    assert direction(src, dst, "row3") == row3_text


# Where column 1 and the nearest row sit for a human on each side of the printed sheet.
SIDE_LAYOUT = {
    "row3": {"column_1": "left", "nearest_row": "C"},   # opposite the arm
    "row1": {"column_1": "right", "nearest_row": "A"},  # the arm's side
}


def test_side_layout_covers_every_human_side():
    assert set(SIDE_LAYOUT) == set(HUMAN_SIDES)


@pytest.mark.parametrize("side", HUMAN_SIDES)
def test_direction_matches_where_the_human_stands(side):
    one_side = SIDE_LAYOUT[side]["column_1"]
    other = "right" if one_side == "left" else "left"
    near = SIDE_LAYOUT[side]["nearest_row"]
    far = "A" if near == "C" else "C"
    assert direction("B2", "B1", side) == f"one cell {one_side}"  # toward column 1
    assert direction("B2", "B3", side) == f"one cell {other}"
    assert direction(f"{far}2", f"{near}2", side) == "two cells toward you"
    assert direction(f"{near}2", f"{far}2", side) == "two cells away from you"


def test_direction_defaults_to_settings_side():
    from brain.settings import HUMAN_SIDE
    assert HUMAN_SIDE == "row3"
    assert direction("A1", "B1") == direction("A1", "B1", "row3")


def test_direction_rejects_bad_side():
    with pytest.raises(ValueError):
        direction("A1", "B1", "row2")


# ---------------------------------------------------------------- fix fields (fault slot, expected slot)


def fix_fields(r):
    return (r.fault_cell, r.fault_level, r.expected_cell, r.expected_level)


@pytest.mark.parametrize(
    "b,step_index,verdict,fields",
    [
        (AFTER_2, 2, Verdict.PASS, (None, None, "C1", 0)),
        (board(A1=["red"], B1=["teal"]), 2, Verdict.MISSING, (None, None, "C1", 0)),
        (board(A1=["red"], B1=["teal"], C2=["skyblue"]), 2, Verdict.WRONG_CELL, ("C2", 0, "C1", 0)),
        (board(A1=["red"], B1=["teal"], C1=["yellow"]), 2, Verdict.WRONG_COLOR, ("C1", 0, "C1", 0)),
        (board(A1=["skyblue", "red"], B1=["teal"], C1=["skyblue"]), 3, Verdict.WRONG_LEVEL, ("A1", 0, "A1", 1)),
        (board(A1=["red"], B1=["teal"], C1=["skyblue"], B3=["yellow"]), 2, Verdict.EXTRA, ("B3", 0, None, None)),
        (board(A1=["red"], B1=["teal"], C3=["yellow"]), 2, Verdict.EXTRA, ("C3", 0, None, None)),  # take it off: no arrow
        (board(A1=["red", "skyblue"], B2=["teal"], C1=["skyblue"]), 3, Verdict.WRONG_CELL, ("B2", 0, "B1", 0)),
        (board(A1=["red", "skyblue"], B1=["yellow"], C1=["skyblue"]), 3, Verdict.WRONG_COLOR, ("B1", 0, "B1", 0)),
        (board(A1=["red", "skyblue"], C1=["skyblue"]), 3, Verdict.MISSING, ("B1", None, "B1", 0)),
    ],
)
def test_fix_fields(b, step_index, verdict, fields):
    r = check(b, LESSON, step_index)
    assert r.verdict is verdict, r
    assert fix_fields(r) == fields, r


def test_expected_board_bounds():
    assert expected_board(LESSON, -1) == empty_board()
    assert expected_board(LESSON, 5) == LESSON.final_board
    with pytest.raises(IndexError):
        expected_board(LESSON, 6)


def test_slot_diff():
    missing, extra = slot_diff(board(A1=["red", "skyblue"]), board(A1=["red", "teal"], C3=["yellow"]))
    assert [tuple(c) for c in missing] == [("A1", 1, "skyblue")]
    assert [tuple(c) for c in extra] == [("A1", 1, "teal"), ("C3", 0, "yellow")]


def test_hints_have_no_em_dashes():
    boards = [
        board(A1=["red"], B1=["teal"], C2=["skyblue"]),
        board(A1=["red"], B1=["teal"], C1=["yellow"]),
        board(A1=["skyblue", "red"], B1=["teal"], C1=["skyblue"]),
        board(A1=["red"], B1=["teal"], C1=["skyblue"], B3=["yellow"]),
        board(A1=["red", "skyblue"], C1=["skyblue"]),
    ]
    for b in boards:
        for i in (2, 3):
            r = check(b, LESSON, i)
            assert r.hint_text and "—" not in r.hint_text and "–" not in r.hint_text, r


# ---------------------------------------------------------------- scripted end to end


def verdicts_for_script(name):
    """Fake feed frames -> stabilizer -> checker, advancing on PASS like check_live does."""
    rng, stab, step_index, out = random.Random(0), Stabilizer(), 0, []
    for seg in parse_script(json.loads((SCRIPTS_DIR / name).read_text())):
        for _ in range(round(seg.duration_s * 10)):
            stable = stab.feed(make_frame(seg.board, seg.hands_clear, seg.arm_clear, seg.jitter, 0, rng))
            while stable is not None and step_index < len(LESSON.steps):
                r: CheckResult = check(stable, LESSON, step_index)
                out.append((step_index, r.verdict.value, r.fault_cell))
                if r.verdict is not Verdict.PASS:
                    break
                step_index += 1
    return out


def test_clean_staircase_passes_every_step():
    out = verdicts_for_script("staircase_clean.json")
    assert [v for _, v, _ in out if v == "PASS"] == ["PASS"] * 6
    assert all(v in ("PASS", "MISSING") for _, v, _ in out)


def test_staircase_mistake_is_caught_then_passes():
    out = verdicts_for_script("staircase_with_mistake.json")
    assert (2, "WRONG_CELL", "C2") in out
    assert out.index((2, "WRONG_CELL", "C2")) < out.index((2, "PASS", None))
    assert out[-1] == (5, "PASS", None)
