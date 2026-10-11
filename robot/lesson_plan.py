"""Turn a lesson (shared.schemas.Lesson) into an executor task plan, as run_task --plan reads it.

    python -m robot.lesson_plan lessons/expert-take1-20261011-052329.json               # print
    python -m robot.lesson_plan lessons/expert-take1-20261011-052329.json -o plan.json  # write
    python -m robot.scripts.tests.run_task --plan plan.json

Each lesson step becomes one executor step, in order:
    level 0  {"object": color, "goal": {"type": "cell", "cell": cell}}
    level k  {"object": color, "goal": {"type": "on_top", "reference": <color at level k-1 in that cell>}}
The color under a stacked step comes from replaying the steps. "skyblue" becomes the executor's
"light blue"; other colors and every cell name pass through unchanged (both sides use the
printed-sheet names: letter = row, number = column).

The executor addresses blocks by color, so a lesson it can't run is refused with a clear reason
(LessonPlanError): a color used twice, a color stacked on itself, a color the executor can't name,
a level with nothing (or something else) under it, or a level above MAX_LEVEL.

Pure: imports shared and robot/perception/config.py (constants only), nothing from lerobot or cv2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from robot.perception.config import ACTIVE_COLORS, ROBOT_EXTRA_COLORS
from shared.config import CELLS, MAX_LEVEL
from shared.schemas import Lesson, SchemaError

TO_EXECUTOR = {"skyblue": "light blue"}  # shared color -> executor color
# The colors the executor's namer can give a block (robot/perception/naming.py NAMEABLE).
EXECUTOR_COLORS = ACTIVE_COLORS + [c for c in ROBOT_EXTRA_COLORS if c not in ACTIVE_COLORS]


class LessonPlanError(ValueError):
    pass


def executor_color(color: str) -> str:
    return TO_EXECUTOR.get(color, color)


def lesson_to_plan(lesson: Lesson) -> dict[str, Any]:
    """{"steps": [...]} for run_task --plan. Raises LessonPlanError if the executor can't run it."""
    stacks: dict[str, list[str]] = {cell: [] for cell in CELLS}
    seen: dict[str, int] = {}  # color -> step number that first used it
    steps = []
    for i, s in enumerate(lesson.steps, 1):
        where = f"step {i} ({s.color} on {s.cell} level {s.level})"
        if s.cell not in stacks:
            raise LessonPlanError(f"{where}: no cell {s.cell!r} (cells are A1-C3)")
        if s.level > MAX_LEVEL:
            raise LessonPlanError(f"{where}: level {s.level} is above the top level {MAX_LEVEL}")
        colour = executor_color(s.color)
        if colour not in EXECUTOR_COLORS:
            raise LessonPlanError(f"{where}: the executor can't recognise {s.color!r} "
                                  f"(it names {', '.join(EXECUTOR_COLORS)})")
        stack = stacks[s.cell]
        if s.level > len(stack):
            raise LessonPlanError(f"{where}: nothing under it ({s.cell} holds {len(stack)} block(s) then)")
        if s.level < len(stack):
            raise LessonPlanError(f"{where}: {s.cell} already has {stack[s.level]} at level {s.level}")
        if s.level > 0 and stack[-1] == s.color:
            raise LessonPlanError(f"{where}: stacks {s.color} on {s.color}; the executor names both the same")
        if s.color in seen:
            raise LessonPlanError(f"{where}: {s.color} is already used by step {seen[s.color]}; the executor "
                                  "addresses blocks by color, so each color may appear once")
        seen[s.color] = i
        if s.level == 0:
            goal = {"type": "cell", "cell": s.cell}
        else:
            goal = {"type": "on_top", "reference": executor_color(stack[-1])}
        steps.append({"object": colour, "goal": goal})
        stack.append(s.color)
    return {"steps": steps}


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lesson", type=Path, help="lesson JSON")
    ap.add_argument("-o", "--out", type=Path, help="write the plan here instead of printing it")
    args = ap.parse_args(argv)
    try:
        lesson = Lesson.from_dict(json.loads(args.lesson.read_text(encoding="utf-8")))
        plan = lesson_to_plan(lesson)
    except (OSError, ValueError, SchemaError) as e:  # LessonPlanError and bad JSON are ValueErrors
        sys.exit(f"{args.lesson}: {e}")
    text = json.dumps(plan, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {len(plan['steps'])} step(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
