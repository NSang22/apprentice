from perception.cellmodel import CellVote, read_cells
from perception.cells import cell_center_mm
from perception.colors import DEFAULT_HEX
from tests.test_perception_grid_lock import BLOCK_MM, POSE, REFS, block_size, empty_sheet, with_stack  # noqa: F401


def test_two_high_stack_with_neighbours_reads_each_cell():
    ref = empty_sheet()
    live = with_stack(ref, cell_center_mm("A1"), [DEFAULT_HEX["red"], DEFAULT_HEX["teal"]])  # 2 high
    live = with_stack(live, cell_center_mm("A2"), [DEFAULT_HEX["yellow"]])                   # touching neighbour
    cells = read_cells(live, POSE, ref, REFS, BLOCK_MM).cells
    assert (cells["A1"].state, cells["A1"].level, cells["A1"].color) == ("block", 1, "teal")
    assert cells["A1"].sides in (("red",), (None,))      # red when enough of its side is in view
    assert (cells["A2"].state, cells["A2"].level, cells["A2"].color) == ("block", 0, "yellow")
    assert all(cells[c].state == "empty" for c in cells if c not in ("A1", "A2"))


def test_vote_needs_a_majority_of_the_last_five():
    vote = CellVote()
    for value in [("red",), ("red",), "unknown", ("red",), ()]:
        vote.add("B2", value)
    assert vote.winner("B2") == (None, False)            # last 3: unknown, red, empty
    vote.add("B2", ())                                    # last 3: red, empty, empty
    assert vote.winner("B2") == ((), True)
