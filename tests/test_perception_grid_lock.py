import cv2
import numpy as np

from perception.cells import cell_center_mm
from perception.change import SHEET_PX_PER_MM, analyze, board_from, sheet_size_px
from perception.colors import DEFAULT_HEX, default_calibration
from perception.grid_lock import check_tags
from perception.perceive import hands_clear

CALIB = default_calibration()
LOCKED = {0: (600.0, 300.0), 1: (1300.0, 300.0), 2: (1300.0, 950.0), 3: (600.0, 950.0)}


def empty_sheet():
    """White paper with tag 0 as a black square at its spec position (22-54 mm, 11-43 mm)."""
    w, h = sheet_size_px()
    sheet = np.full((h, w, 3), 245, np.uint8)
    k = SHEET_PX_PER_MM
    cv2.rectangle(sheet, (int(22 * k), int(11 * k)), (int(54 * k), int(43 * k)), (15, 15, 15), -1)
    return sheet


def with_square(sheet, center_mm, size_mm, hex_color):
    out = sheet.copy()
    k, (cx, cy), half = SHEET_PX_PER_MM, center_mm, size_mm / 2
    bgr = tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))
    cv2.rectangle(out, (int((cx - half) * k), int((cy - half) * k)), (int((cx + half) * k), int((cy + half) * k)), bgr, -1)
    return out


def test_two_agreeing_tags_keep_the_grid():
    check = check_tags(LOCKED, {0: (601.5, 299.0), 2: (1298.0, 951.0)})
    assert check.status == "ok" and check.visible == 2


def test_a_moved_tag_loses_the_grid():
    check = check_tags(LOCKED, {0: (600.0, 300.0), 1: (1300.0, 300.0), 2: (1320.0, 950.0)})
    assert check.status == "moved" and check.moved == (2,)


def test_block_sized_block_color_region_is_a_block_even_over_a_tag():
    ref = empty_sheet()
    a1 = cell_center_mm("A1")
    live = with_square(ref, (a1[0], a1[1] - 14), 50, DEFAULT_HEX["teal"])  # top edge over tag 0
    result = analyze(live, ref, CALIB)
    board, confidence, crowded = board_from(result)
    assert board["A1"] == ["teal"] and confidence["A1"] >= 0.8 and not result.others
    assert hands_clear(result, crowded)


def test_large_non_block_region_is_a_hand():
    ref = empty_sheet()
    live = with_square(ref, cell_center_mm("B2"), 130, "#e0ac69")  # skin tone, bigger than any block
    result = analyze(live, ref, CALIB)
    assert result.others and not hands_clear(result, board_from(result)[2])


def test_hidden_tag_alone_is_never_a_hand():
    check = check_tags(LOCKED, {1: LOCKED[1], 2: LOCKED[2], 3: LOCKED[3]})  # tag 0 covered
    ref = empty_sheet()
    result = analyze(ref.copy(), ref, CALIB)
    assert check.status == "ok" and result.empty and hands_clear(result, board_from(result)[2])
