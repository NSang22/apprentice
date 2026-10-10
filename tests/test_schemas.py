import copy
import json
from pathlib import Path

import pytest

from shared.config import CELLS, MAX_LEVEL
from shared.schemas import (
    Gate,
    GateCheck,
    GridState,
    Lesson,
    Progress,
    SchemaError,
    SessionState,
    ShowMe,
    Step,
    Verdict,
    apply_step,
    boards_equal,
    empty_board,
    validate_board,
)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def board_with(**cells):
    b = empty_board()
    b.update(cells)
    return b


def json_round_trip(obj):
    """to_dict -> JSON text -> from_dict, proving the dict is JSON-safe."""
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))


# ---------------------------------------------------------------- round trips


def test_grid_state_round_trip():
    gs = GridState(
        t=1728580000.123,
        board=board_with(A1=["red"], B2=["blue", "yellow"], C3=["unknown"]),
        arm_clear=True,
        hands_clear=False,
        confidence={"A1": 0.97, "B2": 0.91},
        frame_id=18234,
    )
    assert json_round_trip(gs) == gs


def test_step_round_trip_with_and_without_optionals():
    full = Step("red", "A1", 0, t_video=4.2, narration="Start with red.")
    bare = Step("blue", "C3", 2)
    assert json_round_trip(full) == full
    assert json_round_trip(bare) == bare


def test_lesson_round_trip():
    lesson = Lesson(
        lesson_id="x",
        title="X",
        source_video="videos/x.mp4",
        steps=[Step("red", "A1", 0), Step("blue", "A1", 1)],
        final_board=board_with(A1=["red", "blue"]),
        gate=Gate(passed=True, checks=[GateCheck("replay_matches_final_frame", True)]),
    )
    assert json_round_trip(lesson) == lesson
    lesson.gate = None
    assert json_round_trip(lesson) == lesson


def test_session_state_round_trip():
    ss = SessionState(
        mode="human",
        lesson_id="staircase-v1",
        step_index=3,
        board=board_with(B3=["blue"]),
        expected_step=Step("blue", "B2", 1),
        verdict=Verdict.WRONG_CELL,
        fault_cell="B3",
        hint_level=2,
        hint_text="Close! Move the blue cube one cell to the left.",
        show_me=ShowMe("videos/expert_staircase.mp4", 11.0, 15.0),
        progress=Progress(human=["pass", "retry", "active"], robot=["pass"]),
    )
    d = ss.to_dict()
    assert d["verdict"] == "WRONG_CELL"
    assert json_round_trip(ss) == ss


def test_session_state_defaults_round_trip():
    ss = SessionState(mode="robot", lesson_id="x", step_index=0, board=empty_board())
    assert ss.verdict is Verdict.WAITING
    assert json_round_trip(ss) == ss


def test_verdict_order_is_frozen():
    assert [v.value for v in Verdict] == [
        "PASS", "WRONG_CELL", "WRONG_COLOR", "WRONG_LEVEL", "EXTRA", "MISSING", "WAITING"
    ]


def test_hand_written_lesson_loads_and_replays():
    lesson = Lesson.from_dict(json.loads((FIXTURES_DIR / "staircase-v1.json").read_text()))
    board = empty_board()
    for step in lesson.steps:
        board = apply_step(board, step)
    assert boards_equal(board, lesson.final_board)


@pytest.mark.parametrize("script", sorted(SCRIPTS_DIR.glob("*.json")), ids=lambda p: p.name)
def test_script_boards_are_valid(script):
    for seg in json.loads(script.read_text()):
        validate_board({**empty_board(), **seg["board"]})


def test_clean_script_ends_on_lesson_final_board():
    lesson = Lesson.from_dict(json.loads((FIXTURES_DIR / "staircase-v1.json").read_text()))
    for name in ("staircase_clean.json", "staircase_with_mistake.json"):
        last = json.loads((SCRIPTS_DIR / name).read_text())[-1]["board"]
        assert boards_equal({**empty_board(), **last}, lesson.final_board), name


# ---------------------------------------------------------------- validation


def test_validate_board_accepts_good_board():
    validate_board(board_with(A1=["red", "blue", "unknown"]))


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({c: [] for c in CELLS if c != "B2"}, id="missing-cell"),
        pytest.param({**empty_board(), "D1": []}, id="extra-cell"),
        pytest.param(board_with(A1=["purple"]), id="bad-color"),
        pytest.param(board_with(A1=["Red"]), id="wrong-case-color"),
        pytest.param(board_with(A1=["red"] * (MAX_LEVEL + 2)), id="too-tall"),
        pytest.param(board_with(A1="red"), id="stack-not-list"),
        pytest.param([], id="not-a-dict"),
    ],
)
def test_validate_board_rejects(bad):
    with pytest.raises(SchemaError):
        validate_board(bad)


def test_validate_board_can_forbid_unknown():
    with pytest.raises(SchemaError):
        validate_board(board_with(A1=["unknown"]), allow_unknown=False)


def test_from_dict_rejects_contract_violations():
    good = GridState(0.0, empty_board(), True, True, {}, 0).to_dict()
    with pytest.raises(SchemaError):
        GridState.from_dict({k: v for k, v in good.items() if k != "frame_id"})
    with pytest.raises(SchemaError):
        GridState.from_dict({**good, "confidence": {"Z9": 0.9}})
    with pytest.raises(SchemaError):
        Step.from_dict({"color": "unknown", "cell": "A1", "level": 0})
    with pytest.raises(SchemaError):
        Step.from_dict({"color": "red", "cell": "A1", "level": MAX_LEVEL + 1})
    with pytest.raises(SchemaError):
        SessionState.from_dict({"mode": "robot", "lesson_id": "x", "step_index": 0,
                                "board": empty_board(), "verdict": "MAYBE"})


# ---------------------------------------------------------------- helpers


def test_empty_board_has_all_cells_empty():
    b = empty_board()
    assert list(b) == CELLS
    assert all(stack == [] for stack in b.values())


def test_apply_step_stacks_bottom_to_top():
    b = empty_board()
    b = apply_step(b, Step("red", "B2", 0))
    b = apply_step(b, Step("blue", "B2", 1))
    b = apply_step(b, Step("green", "B2", 2))
    assert b["B2"] == ["red", "blue", "green"]
    assert all(b[c] == [] for c in CELLS if c != "B2")


def test_apply_step_rejects_wrong_level():
    with pytest.raises(SchemaError):
        apply_step(empty_board(), Step("red", "A1", 1))  # floating cube
    with pytest.raises(SchemaError):
        apply_step(board_with(A1=["red"]), Step("blue", "A1", 0))  # level taken


def test_apply_step_does_not_mutate_input():
    b = board_with(A1=["red"])
    before = copy.deepcopy(b)
    out = apply_step(b, Step("blue", "A1", 1))
    assert b == before
    assert out["A1"] == ["red", "blue"]
    out["C3"].append("green")  # output shares no lists with input
    assert b == before


def test_boards_equal():
    a = board_with(A1=["red", "blue"])
    assert boards_equal(a, board_with(A1=["red", "blue"]))
    assert not boards_equal(a, board_with(A1=["blue", "red"]))  # order matters
    assert not boards_equal(a, board_with(A2=["red", "blue"]))
