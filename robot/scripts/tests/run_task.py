"""Run a multi-step block task, hands-free.

    python -m robot.scripts.tests.run_task "red->C1" "purple->C3"
    python -m robot.scripts.tests.run_task "red->B2" "purple->beside:red:right"
    python -m robot.scripts.tests.run_task "red->B2" "yellow->on:red"
    python -m robot.scripts.tests.run_task --plan my_task.json   # {"steps": [ ... ]} as in executor.py

Step syntax: <colour>-><cell>, <colour>->beside:<reference>:<left|right|top|bottom>[:<gap mm>],
or <colour>->on:<reference>. Colours are names like red or "light blue".
The camera checks the table before and after every step. A block that lands off target gets up
to 2 retries; any other problem stops the task. Blocks must all start on the table; stacks the
task builds are tracked (towers up to 3 blocks), so later steps can unstack or build on them.
Blocks crowding a target (under 8 mm edge to edge) or hemming in a block so no grip fits are moved
to the nearest empty cell first; the step's reference block and later steps' target cells are left alone.
"""

import argparse
import json
import time

from robot.arm import Arm
from robot.correction import Correction
from robot.executor import Executor, TaskError, describe
from robot.perception.camera import Camera
from robot.perception.sheet import SheetLocalizer
from robot.recorder import RunRecorder
from robot.session import ArmSession
from robot.skills import Skills
from robot.workspace import Workspace


def parse_step(text):
    obj, sep, goal = text.partition("->")
    if not sep or not obj.strip() or not goal.strip():
        raise SystemExit(f"bad step {text!r}; e.g. red->C1, purple->beside:red:right, yellow->on:red")
    parts = [p.strip() for p in goal.split(":")]
    if parts[0].lower() == "beside" and len(parts) >= 3:
        g = {"type": "beside", "reference": parts[1], "side": parts[2].lower()}
        if len(parts) > 3:
            g["gap_mm"] = float(parts[3])
    elif parts[0].lower() == "on" and len(parts) == 2:
        g = {"type": "on_top", "reference": parts[1]}
    elif len(parts) == 1:
        g = {"type": "cell", "cell": parts[0].upper()}
    else:
        raise SystemExit(f"bad goal in {text!r}")
    return {"object": obj.strip(), "goal": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--plan", help="JSON file with {\"steps\": [...]}")
    args = ap.parse_args()
    steps = json.loads(open(args.plan).read())["steps"] if args.plan else [parse_step(s) for s in args.steps]
    if not steps:
        ap.error("give steps, e.g. \"red->C1\" \"purple->C3\", or --plan file.json")
    print("Task:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {describe(s)}")

    ws, loc, cam, arm = Workspace(), SheetLocalizer(), Camera(), Arm()
    rec = RunRecorder(cam, loc, ws, arm, label="task")
    sk = Skills(arm, ws, log=lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}"), on_phase=rec,
                correct=Correction())
    t0 = time.time()
    try:
        with ArmSession(arm, sk):
            ex = Executor(sk, cam, loc, rec)
            try:
                report = ex.run(steps)
                print(f"\nTASK DONE: {len(steps)} step(s) in {time.time() - t0:.0f} s")
            finally:
                print(f"Task report: {ex.report_path}")
    except TaskError as e:
        print(f"\nTASK STOPPED: {e}")
    finally:
        cam.close()
        print(f"Saving evidence... {rec.save()}")


if __name__ == "__main__":
    main()
