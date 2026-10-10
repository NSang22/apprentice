import cv2
from pathlib import Path

from perception.publish import MIN_TAGS, build_grid_state
from robot.perception.sheet import SheetLocalizer
from shared.config import CELLS


def test_robot_perception_imports_without_lerobot():
    from robot.perception import detect_color, scene, sheet

    assert detect_color and scene and sheet


def test_sample_images_produce_contract_grid_states():
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    last_pose = None
    for frame_id, path in enumerate(sorted(Path("robot/samples").glob("overhead_*.png"))):
        frame = cv2.imread(str(path))
        state, last_pose, _ = build_grid_state(frame, localizer, last_pose, frame_id, timestamp=123.0)
        assert set(state.board) == set(CELLS)
        assert set(state.confidence) == set(CELLS)
        assert last_pose is not None
        assert state.arm_clear is True


def test_too_few_tags_reuses_last_pose_and_marks_hands_unclear():
    localizer = SheetLocalizer(min_tags=MIN_TAGS)
    visible = cv2.imread("robot/samples/overhead_02.png")
    first, pose, _ = build_grid_state(visible, localizer, None, 0)
    assert first.hands_clear is True

    occluded = visible.copy()
    occluded[:] = 255
    second, reused_pose, _ = build_grid_state(occluded, localizer, pose, 1)
    assert reused_pose is pose
    assert second.hands_clear is False
