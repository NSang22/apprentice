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
    g = recall.grade({"A1": "red", "B2": "teal"}, board(C1="red", B2="teal"))
    assert not g.perfect
    assert g.errors["place"] == 1 and g.errors["mirror"] == 1 and g.errors["extra"] == 0


def test_extra_block_is_not_perfect():
    g = recall.grade({"A1": "red"}, board(A1="red", C3="purple"))
    assert g.right == 1 and not g.perfect and g.errors["extra"] == 1
