import json

import pytest

from perception.cells import DEFAULT_SHEET_SPEC, mm_to_cell


@pytest.fixture(scope="module")
def geometry():
    grid = json.loads(DEFAULT_SHEET_SPEC.read_text())["grid"]
    return (*grid["origin_mm"], grid["cell_mm"])


def test_all_nine_cell_centers(geometry):
    x0, y0, size = geometry
    for column, letter in enumerate("ABC"):
        for row in range(3):
            assert mm_to_cell(x0 + (column + 0.5) * size, y0 + (row + 0.5) * size) == f"{letter}{row + 1}"


def test_outer_boundaries_are_half_open(geometry):
    x0, y0, size = geometry
    epsilon = 1e-7
    assert mm_to_cell(x0, y0) == "A1"
    assert mm_to_cell(x0 - epsilon, y0) is None
    assert mm_to_cell(x0, y0 - epsilon) is None
    assert mm_to_cell(x0 + 3 * size - epsilon, y0 + 3 * size - epsilon) == "C3"
    assert mm_to_cell(x0 + 3 * size, y0) is None
    assert mm_to_cell(x0, y0 + 3 * size) is None


def test_internal_boundaries_belong_to_next_cell(geometry):
    x0, y0, size = geometry
    assert mm_to_cell(x0 + size, y0 + size / 2) == "B1"
    assert mm_to_cell(x0 + size / 2, y0 + size) == "A2"
    assert mm_to_cell(x0 + 2 * size, y0 + 2 * size) == "C3"
