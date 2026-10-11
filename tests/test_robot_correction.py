"""robot.correction: the placement correction must not extrapolate its stretch into the cube bank."""

import json

import numpy as np

from robot.correction import FITTED_BOX, Correction


def test_correction_is_unchanged_on_the_grid_and_flat_beyond_it(tmp_path):
    # The 2026-10-11 fit: x scaled by 0.92, which added 16 mm at the bank (x ~300) and the jaws missed.
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"A": [[0.918344, -0.006707], [0.000292, 0.968212]], "b": [9.976, 0.368]}))
    c = Correction(path)
    inside = np.array([108.0, 134.0])
    assert np.allclose(c(inside), c.Ainv @ (inside - c.b))               # the fit itself on the grid
    edge = np.array([FITTED_BOX[2], 134.0])
    assert np.allclose(c([300.0, 134.0]) - [300.0, 134.0], c(edge) - edge)  # bank: the edge's shift


def test_bank_placements_get_their_own_translation(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"A": [[0.918344, -0.006707], [0.000292, 0.968212]], "b": [9.976, 0.368],
                                "bank": {"b": [4.0, -6.0]}}))
    c = Correction(path)
    assert np.allclose(c([266.0, 125.0]), [262.0, 131.0])             # in the bank: aim minus the measured error
    inside = np.array([108.0, 134.0])
    assert np.allclose(c(inside), c.Ainv @ (inside - c.b))            # on the grid: the affine fit, as before
