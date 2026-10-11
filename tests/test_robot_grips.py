"""robot.skills.rank_grips: choose pick and place grips that clear neighbours and keep the wrist roll small."""

import numpy as np
import pytest

pytest.importorskip("lerobot")  # robot.skills imports the arm driver

from robot.skills import SkillError, rank_grips  # noqa: E402

INF = float("inf")


def opt(roll, clear=INF, jaw=(-1, 0)):
    return {"tip": np.zeros(2), "roll": roll, "jaw": np.array(jaw, float), "clear_mm": clear, "turns": 0}


def test_smallest_wrist_roll_when_nothing_is_near():
    a, b = rank_grips([opt(90), opt(-5), opt(-95)], [opt(85), opt(3)])[0]
    assert a["roll"] == -5 and b["roll"] == 3


def test_place_roll_counts_more_than_pick_roll():
    # accuracy matters where the cube is released
    a, b = rank_grips([opt(0), opt(90)], [opt(10), opt(80)])[0]
    assert b["roll"] == 10


def test_neighbour_forces_a_bigger_roll():
    # B2 next to another block: the default grip's jaw would hit it
    a, b = rank_grips([opt(0), opt(90)], [opt(-2, clear=-17), opt(87, clear=21)])[0]
    assert b["roll"] == 87


def test_small_release_roll_beats_extra_room():
    # run 19:33: the normal grip had 5 mm of room, the 87 deg one more; the 87 deg one missed by 23 mm
    a, b = rank_grips([opt(0)], [opt(-2, clear=5), opt(87, clear=20)])[0]
    assert b["roll"] == -2


def test_prefers_room_to_spare_among_small_rolls():
    a, b = rank_grips([opt(0)], [opt(5, clear=2), opt(-20, clear=20)])[0]
    assert b["roll"] == -20


def test_every_pair_is_ranked_for_fallback():
    assert len(rank_grips([opt(0), opt(90)], [opt(0), opt(90), opt(-90, clear=-1)])) == 4


def test_stacked_cube_is_measured_at_its_height():
    from robot.executor import at_height
    cam = (142.8, 284.6, 414.7)
    # run 19:14: the scene put the stacked cube at (177.0, 103.3) as if it stood on the table
    assert np.allclose(at_height((177.0, 103.3), cam, 48.0), (172.8, 125.6), atol=0.3)
    assert np.allclose(at_height((50.0, 80.0), cam, 0.0), (50.0, 80.0))


def test_parking_beside_the_sheet_keeps_the_grid_in_view():
    from robot.executor import cell_center, grid_box
    from robot.skills import PARK_H, hides_area, max_tilt, side_park_spots
    from robot.workspace import Workspace
    ws, cam, box = Workspace(), (142.8, 284.6, 414.7), grid_box()
    over_b2 = ws.solve(cell_center("B2"), PARK_H, max_tilt_deg=max_tilt(PARK_H), max_err_mm=5.0)
    assert hides_area(ws, over_b2, cam, box)  # hovering over the grid does block the view
    spots = side_park_spots(ws, cell_center("C3"), cam, box)
    assert spots and spots[0][0][0] > box[2]  # after C3, park off the right edge
    assert not hides_area(ws, spots[0][1], cam, box)


def test_blocked_target_raises():
    with pytest.raises(SkillError, match="target"):
        rank_grips([opt(0)], [opt(0, clear=-1), opt(90, clear=-2)])


def test_jaws_stay_well_above_a_stack():
    # run 19:59: closing the jaws right above a 2-block tower knocked it over
    from robot.skills import MAX_TRAVEL_H, TRAVEL_H, Skills
    sk = Skills(arm=None, ws=None)
    assert sk.travel_h() == TRAVEL_H and sk._clear_h(48.0) == TRAVEL_H   # only single blocks: as before
    sk.tallest_mm = 96.0                                                  # a 2-block tower somewhere
    assert sk.travel_h() == 156.0                                         # run 20:41: 120 swung into it
    assert sk._clear_h(96.0) == 156.0
    sk.tallest_mm = 144.0
    assert sk.travel_h() == MAX_TRAVEL_H                                  # 3 high: capped at reach
