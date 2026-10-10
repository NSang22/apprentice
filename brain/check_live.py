"""Demo: stabilizer + checker against a live GridState feed. Prints one line per check.

    python -m brain.fake_feed --script tests/scripts/staircase_with_mistake.json   # terminal 1
    python -m brain.check_live --lesson tests/fixtures/staircase-v1.json            # terminal 2

On each new stable board it checks the current step and prints step, verdict, fault cell
and hint. On PASS it advances and re-checks the same board against the next step, so the
next instruction shows up right away. Exits when the lesson is complete. If the feed is
not up yet, or drops, it retries once a second.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from brain.checker import check
from brain.settings import DEMO_LESSON
from brain.stabilizer import Stabilizer
from shared.config import GRID_WS_PORT
from shared.schemas import GridState, Lesson, SchemaError, Verdict

RETRY_S = 1.0


def report(step_index: int, total: int, verdict: Verdict, fault_cell, hint) -> str:
    fault = f" at {fault_cell}" if fault_cell else ""
    return f"step {step_index + 1}/{total}  {verdict.value:<11}{fault:<7}  {hint or ''}"


async def run(url: str, lesson: Lesson) -> None:
    stab = Stabilizer()
    step_index, total = 0, len(lesson.steps)
    print(f"lesson {lesson.lesson_id}: {total} steps")
    while True:
        try:
            async with connect(url) as ws:
                print(f"connected to {url}")
                async for msg in ws:
                    try:
                        gs = GridState.from_dict(json.loads(msg))
                    except (SchemaError, ValueError) as e:
                        print(f"skipping bad frame: {e}")
                        continue
                    stable = stab.feed(gs)
                    while stable is not None:
                        r = check(stable, lesson, step_index)
                        print(report(step_index, total, r.verdict, r.fault_cell, r.hint_text), flush=True)
                        if r.verdict is not Verdict.PASS:
                            break
                        step_index += 1
                        if step_index == total:
                            print("lesson complete")
                            return
        except (OSError, ConnectionClosed) as e:
            print(f"feed not available ({e.__class__.__name__}); retrying in {RETRY_S:g}s", flush=True)
            await asyncio.sleep(RETRY_S)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lesson", default=str(DEMO_LESSON))
    p.add_argument("--url", default=f"ws://localhost:{GRID_WS_PORT}")
    return p


def main() -> None:
    args = parser().parse_args()
    lesson = Lesson.from_dict(json.loads(Path(args.lesson).read_text()))
    try:
        asyncio.run(run(args.url, lesson))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
