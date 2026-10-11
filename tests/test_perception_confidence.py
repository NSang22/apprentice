import pytest

from perception.cells import edge_margin_mm
from shared.config import MIN_CONFIDENCE
from perception.blocks import STRADDLE_CONFIDENCE


def test_cell_edge_margin():
    # Grid origin (22.225, 48.26) mm, 57.15 mm cells: A1 centre is 28.6 mm from every edge.
    assert edge_margin_mm(22.225 + 28.575, 48.26 + 28.575) == pytest.approx(28.575)
    assert edge_margin_mm(22.225 + 57.15 - 3.0, 48.26 + 28.0) == pytest.approx(3.0)  # near the A|B line
    assert edge_margin_mm(0.0, 0.0) < 0
    assert STRADDLE_CONFIDENCE < MIN_CONFIDENCE  # a straddling cube never passes
