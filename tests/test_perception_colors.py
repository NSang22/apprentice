import numpy as np
import pytest

from perception.colors import (ColorRef, classify, default_calibration, load_calibration, margin_score,
                               sample_stats, save_calibration)
from shared.config import COLORS, UNKNOWN

REFS = default_calibration().refs


def test_nearest_color_wins_and_far_colors_are_unknown():
    for name, ref in REFS.items():
        assert classify(ref.lab, REFS)[0] == name
    assert classify((50.0, 0.0, 0.0), REFS)[0] == UNKNOWN  # mid gray is near no block color


def test_margin_separates_look_alikes_by_distance_not_hue_bands():
    teal, sky = np.array(REFS["teal"].lab), np.array(REFS["skyblue"].lab)
    assert classify(teal, REFS)[1] == 1.0
    midpoint = classify((teal + sky) / 2, REFS)
    assert midpoint[1] < 0.2  # equally close to both: never confident enough to pass
    assert margin_score(2.0, 30.0) == 1.0 and margin_score(5.0, 5.0) == 0.0


def test_highlights_do_not_shift_the_sample():
    block = np.tile(np.array([[55.0, -30.0, -5.0]], np.float32), (200, 1))   # teal-ish top face
    sheen = np.tile(np.array([[95.0, -3.0, 2.0]], np.float32), (150, 1))     # bright, colorless
    median, spread, used = sample_stats(np.vstack([block, sheen]))
    assert used == 200 and median == pytest.approx((55.0, -30.0, -5.0)) and spread == pytest.approx(0.0)


def test_calibration_round_trip_keeps_area_headroom(tmp_path):
    path = tmp_path / "colors.json"
    refs = {c: ColorRef(c, r.lab, 3.0, r.hex) for c, r in REFS.items()}
    save_calibration(path, refs, [0.40, 0.45, 0.50], cell_mm=57.15)
    calib = load_calibration(path)
    assert calib.calibrated and set(calib.refs) == set(COLORS)
    lo, med, hi = calib.area_frac
    assert lo < 0.40 and med == pytest.approx(0.45) and hi > 0.50
