import random

from buddy import recall
from shared.config import CELLS

COLORS = ["red", "yellow", "teal", "purple"]


def board(**cells):
    b = {c: [] for c in CELLS}
    for cell, color in cells.items():
        b[cell] = [color]
    return b


def test_pattern_respects_kit():
    p = recall.make_pattern(7, COLORS, random.Random(3))
    assert len(p) == 7 and set(p) <= set(CELLS)
    assert all(list(p.values()).count(c) <= recall.MAX_PER_COLOR for c in COLORS)


def test_perfect_build():
    g = recall.grade({"A1": "red", "B2": "teal"}, board(A1="red", B2="teal"))
    assert g.perfect and g.right == 2


def test_mirror_mixup_is_counted():
    g = recall.grade({"A1": "red", "B2": "teal"}, board(A3="red", B2="teal"))
    assert not g.perfect
    assert g.errors["place"] == 1 and g.errors["mirror"] == 1 and g.errors["extra"] == 0


def test_arm_goes_to_the_square_the_camera_names():
    """Every cell name means the same point on the sheet to perception and to the robot."""
    from perception.cells import cell_center_mm, mm_to_cell
    from robot.perception.sheet import cell_center as robot_cell_center
    from buddy.table import local_cell_map
    mapping = local_cell_map()
    for cell in CELLS:
        rx, ry = robot_cell_center(mapping[cell])
        px, py = cell_center_mm(cell)
        assert abs(rx - px) < 1 and abs(ry - py) < 1
        assert mm_to_cell(rx, ry) == cell   # and the camera reads a block there as that cell


def test_robot_blocks_become_a_board():
    from buddy.table import blocks_to_board, board_diff
    blocks = [{"cell": "A1", "color": "red", "edge_mm": 20.0}, {"cell": "B2", "color": "teal", "edge_mm": 3.0},
              {"cell": None, "color": "yellow", "edge_mm": None}]   # off the grid: a supply spot
    b = blocks_to_board(blocks)
    assert b["A1"] == ["red"] and b["B2"] == ["unknown"] and sum(1 for s in b.values() if s) == 2
    assert board_diff(b, board(A1="red", B2="teal")) == {"B2": ["unknown", "teal"]}


def test_hint_points_at_missing_first_and_fix_takes_wrong_blocks():
    target = {"A1": "red", "B2": "teal"}
    g = recall.grade(target, board(A1="yellow", C3="purple"))
    assert recall.hint_cell(target, g) == "B2"
    assert recall.fix_cells(g) == ["A1", "C3"]


def test_extra_block_is_not_perfect():
    g = recall.grade({"A1": "red"}, board(A1="red", C3="purple"))
    assert g.right == 1 and not g.perfect and g.errors["extra"] == 1


def test_narration_lines_up_with_steps_without_square_names():
    """A real recording: the speaker never named a square, put purple down and covered it, and used red twice."""
    from buddy.teach import check_narration
    steps = [{"color": "yellow", "cell": "B2", "level": 0}, {"color": "red", "cell": "A1", "level": 0},
             {"color": "orange", "cell": "B1", "level": 0}, {"color": "teal", "cell": "C3", "level": 0}]
    said = [{"action": "place", "color": c, "cell": None, "text": f"{c} over here"}
            for c in ("yellow", "red", "purple", "orange", "teal", "red")]
    r = check_narration(steps, said)
    assert r["matched"] == 4 and r["unseen"] == ["purple", "red"]
