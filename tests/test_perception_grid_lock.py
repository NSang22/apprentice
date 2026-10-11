import cv2
import numpy as np
import pytest

from perception.cellmodel import read_cells
from perception.cells import cell_center_mm
from perception.change import SHEET_PX_PER_MM, sheet_size_px
from perception.colors import DEFAULT_HEX, active_refs, default_calibration
from perception.grid_lock import check_tags
from perception.perceive import hands_clear
from perception.robot_bridge import set_block_mm
from robot.perception import cubefit
from robot.perception.sheet import SheetPose

BLOCK_MM = 50.0
REFS = active_refs(default_calibration(), ["red", "purple", "yellow", "teal"])
LOCKED = {0: (600.0, 300.0), 1: (1300.0, 300.0), 2: (1300.0, 950.0), 3: (600.0, 950.0)}
# Image = the sheet at 2 px/mm, seen by a camera 432 mm above the sheet's middle (so a taller
# stack's top projects bigger and further out, as on the rig).
POSE = SheetPose(np.diag([1 / SHEET_PX_PER_MM, 1 / SHEET_PX_PER_MM, 1.0]), [0, 1, 2, 3], 0.0,
                 np.array([108.0, 140.0, 432.0]))


def empty_sheet():
    """White paper with tag 0 as a black square at its spec position (22-54 mm, 11-43 mm)."""
    w, h = sheet_size_px()
    sheet = np.full((h, w, 3), 245, np.uint8)
    k = SHEET_PX_PER_MM
    cv2.rectangle(sheet, (int(22 * k), int(11 * k)), (int(54 * k), int(43 * k)), (15, 15, 15), -1)
    return sheet


def bgr(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))


def with_stack(sheet, center_mm, colors, size_mm=BLOCK_MM):
    """Paint a stack (bottom first) the way the camera sees it: each block's column segment."""
    out = sheet.copy()
    sq = cubefit.cube_corners(center_mm, 0.0, size_mm=size_mm)
    for k, hex_color in enumerate(colors):
        pts = np.vstack([POSE.to_image(sq, k * size_mm), POSE.to_image(sq, (k + 1) * size_mm)]).astype(np.float32)
        cv2.fillConvexPoly(out, cv2.convexHull(pts).astype(np.int32), bgr(hex_color))
    return out


def with_square(sheet, center_mm, size_mm, hex_color):
    out = sheet.copy()
    k, (cx, cy), half = SHEET_PX_PER_MM, center_mm, size_mm / 2
    cv2.rectangle(out, (int((cx - half) * k), int((cy - half) * k)), (int((cx + half) * k), int((cy + half) * k)),
                  bgr(hex_color), -1)
    return out


@pytest.fixture(autouse=True)
def block_size():
    set_block_mm(BLOCK_MM)
    yield
    set_block_mm(None)  # back to the robot's CUBE_SIZE_MM for every other test


def read(frame, ref):
    return read_cells(frame, POSE, ref, REFS, BLOCK_MM)


def test_two_agreeing_tags_keep_the_grid():
    check = check_tags(LOCKED, {0: (601.5, 299.0), 2: (1298.0, 951.0)})
    assert check.status == "ok" and check.visible == 2


def test_a_moved_tag_loses_the_grid():
    check = check_tags(LOCKED, {0: (600.0, 300.0), 1: (1300.0, 300.0), 2: (1320.0, 950.0)})
    assert check.status == "moved" and check.moved == (2,)


def test_block_in_a_corner_cell_is_read_even_over_a_tag():
    ref = empty_sheet()
    a1 = cell_center_mm("A1")
    live = with_stack(ref, (a1[0], a1[1] - 8), [DEFAULT_HEX["teal"]])  # leans toward tag 0
    result = read(live, ref)
    r = result.cells["A1"]
    assert (r.state, r.level, r.color) == ("block", 0, "teal") and r.confidence >= 0.8
    assert not result.others and hands_clear(result)


def test_large_non_block_region_is_a_hand():
    ref = empty_sheet()
    live = with_square(ref, cell_center_mm("B2"), 130, "#e0ac69")  # skin tone, bigger than any block
    result = read(live, ref)
    assert result.others and not hands_clear(result)


def test_inactive_color_reads_unknown():
    ref = empty_sheet()
    live = with_stack(ref, cell_center_mm("B2"), ["#1a5fd6"])  # a blue block: not in play
    assert read(live, ref).cells["B2"].state == "unknown"


def test_hidden_tag_alone_is_never_a_hand():
    check = check_tags(LOCKED, {1: LOCKED[1], 2: LOCKED[2], 3: LOCKED[3]})  # tag 0 covered
    ref = empty_sheet()
    result = read(ref.copy(), ref)
    assert check.status == "ok" and result.empty and hands_clear(result)
