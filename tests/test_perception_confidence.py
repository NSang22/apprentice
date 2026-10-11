import pytest

from perception.cells import edge_margin_mm
from shared.config import MIN_CONFIDENCE
from perception.publish import STRADDLE_CONFIDENCE, area_score, fill_score


def test_fill_rescaled_so_angled_cubes_are_not_penalized():
    assert fill_score(0.58) == 1.0 and fill_score(0.73) == 1.0 and fill_score(0.5) == 1.0
    assert fill_score(0.25) == pytest.approx(0.5)


def test_area_full_inside_range_and_falls_off_outside():
    assert area_score(20000, (10000, 30000)) == 1.0
    assert area_score(5000, (10000, 30000)) == pytest.approx(0.5)
    assert area_score(60000, (10000, 30000)) == pytest.approx(0.5)  # two merged cubes


def test_cell_edge_margin():
    # Grid origin (22.225, 48.26) mm, 57.15 mm cells: A1 centre is 28.6 mm from every edge.
    assert edge_margin_mm(22.225 + 28.575, 48.26 + 28.575) == pytest.approx(28.575)
    assert edge_margin_mm(22.225 + 57.15 - 3.0, 48.26 + 28.0) == pytest.approx(3.0)  # near the A|B line
    assert edge_margin_mm(0.0, 0.0) < 0
    assert STRADDLE_CONFIDENCE < MIN_CONFIDENCE  # a straddling cube never passes
