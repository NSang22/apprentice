import cv2

from perception.colors import default_calibration
from perception.perceive import Perceiver
from shared.config import CELLS


def test_robot_perception_imports_without_lerobot():
    from robot.perception import detect_color, scene, sheet

    assert detect_color and scene and sheet


def test_without_a_grid_lock_frames_are_contract_grid_states_never_usable(tmp_path):
    perceiver = Perceiver(default_calibration(), lock_path=tmp_path / "lock.json", ref_path=tmp_path / "ref.png")
    result = perceiver.step(cv2.imread("robot/samples/overhead_02.png"), 0, timestamp=123.0)
    assert set(result.state.board) == set(CELLS) and set(result.state.confidence) == set(CELLS)
    assert result.state.hands_clear is False and result.status.startswith("GRID NOT LOCKED")
