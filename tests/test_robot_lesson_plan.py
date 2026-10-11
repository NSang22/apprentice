"""robot.lesson_plan: lesson -> run_task --plan steps, and the lessons the executor can't run."""

import json

import pytest

from robot.lesson_plan import LessonPlanError, lesson_to_plan
from shared.config import CELLS, LESSONS_DIR
from shared.schemas import Lesson


def lesson(*steps):
    final = {cell: [] for cell in CELLS}
    for color, cell, _ in steps:
        final[cell].append(color)
    return Lesson.from_dict({
        "lesson_id": "t", "title": "t", "source_video": None, "final_board": final, "gate": None,
        "steps": [{"color": c, "cell": cell, "level": lvl, "t_video": None, "narration": None}
                  for c, cell, lvl in steps],
    })


def test_expert_take1_is_five_cell_steps_with_light_blue():
    data = json.loads((LESSONS_DIR / "expert-take1-20261011-052329.json").read_text())
    plan = lesson_to_plan(Lesson.from_dict(data))
    assert [s["goal"]["type"] for s in plan["steps"]] == ["cell"] * 5
    assert plan["steps"][0] == {"object": "light blue", "goal": {"type": "cell", "cell": "B2"}}
    assert [s["goal"]["cell"] for s in plan["steps"]] == ["B2", "C1", "A1", "B3", "C3"]


def test_stacked_step_goes_on_top_of_the_block_below():
    plan = lesson_to_plan(lesson(("red", "C1", 0), ("purple", "C1", 1)))
    assert plan["steps"] == [
        {"object": "red", "goal": {"type": "cell", "cell": "C1"}},
        {"object": "purple", "goal": {"type": "on_top", "reference": "red"}},
    ]


def test_a_repeated_color_is_rejected():
    with pytest.raises(LessonPlanError, match="yellow is already used by step 1"):
        lesson_to_plan(lesson(("yellow", "A1", 0), ("yellow", "C2", 0)))
