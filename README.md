# Apprentice

**Record one demonstration. Apprentice teaches it to a robot and a person, and the same camera checks both.**

Knight Hacks IX (UCF), Oct 9 to 11, 2026. Team of four.

---

## 1. The idea in 30 seconds

Training someone to build something physical takes an expert standing next to them. Programming a robot to do the same task takes an engineer. Apprentice replaces both with one recording.

1. An expert builds a structure out of colored cubes on a 3x3 grid while narrating.
2. Apprentice turns that video into a **lesson**: an ordered list of steps (which cube, which cell, which level).
3. The lesson has two students:
   - **A person** follows it on a monitor. The camera checks every placement and gives escalating hints when they get it wrong.
   - **A robot arm** (SO-101) follows it through arm commands. The same camera checks every placement, and the robot retries when it gets it wrong.
4. Every attempt, human or robot, gets logged. That log becomes our eval numbers and our analytics.

**This is one project, not two.** Everything above the split below is shared. Only the last layer differs.

```
expert video --> grid reader --> extractor --> LESSON (step list) --> gate
                                                    |
                                  camera + checker (grades every step)
                                     /                          \
                          robot student                    human student
                        (arm commands, retry)        (monitor, hints, show-me clips)
```

Test question for anyone confused: "If we swapped the robot for a person, what code would change?" Only the output layer.

---

## 2. What we demo (and only what we demo)

Today it works on **cube structures on a 3x3 grid**. We never claim "teaches anything." The honest line: "Today it's blocks on a grid. A lesson is just steps with a part, a position, and an order, so the same approach extends to real assembly."

### Demo script (about 3 minutes)

1. **Lesson from video (20 s).** Play the expert video. Show the step list Apprentice extracted, plus the gate result ("replayed steps match the final frame").
2. **Human student (60 s).** A judge builds the structure following the monitor. We hand them a wrong cube on purpose. The checker catches it, the hints escalate, and the robot points to the correct cell.
3. **Robot student (60 s).** The robot builds the same structure. A judge moves one cube to the wrong cell (on-grid only). The camera catches it and the robot fixes it.
4. **Numbers (20 s).** Eval chart from our real trial log, plus the "what we tried first" story.
5. **Stretch, only if solid:** the judge builds something new and the robot rebuilds it ("I taught a robot in 30 seconds"). Then the robot tears it down to reset the table.

### Scope line

Nothing past the core loop starts until we have **10 clean end-to-end runs**.

| Tier | Features |
|---|---|
| Core | Video to lesson, gate, human mode with checker and hints, robot mode with checker and retry, journal, monitor UI |
| Next (after 10 clean runs) | Judge teaches mode (live build to lesson), robot points, robot teardown |
| Stretch | Human and robot alternate steps, voice "what's next", Databricks analytics loop, coarse circuit board lesson |
| Cut | Isaac Sim, RL training, Quest (only if borrowed and everything else works), "build anything" claims |

---

## 3. Hardware and physical setup

| Item | Spec |
|---|---|
| Grid | 3x3, 2.25 in cells, printed, **taped down** |
| Tags | AprilTag 36h11, IDs 0 to 3 at the corners (0 front-left, 1 front-right, 2 back-right, 3 back-left, "front" = operator side) |
| Cubes | Matte painted wooden cubes, red / blue / yellow / green. **Use 1.25 in cubes only.** Keep the 1 in cubes in the bag (mixing sizes breaks taught poses) |
| Camera | Webcam on a tripod, fixed above the grid. Never move it after calibration. Recalibrate color at the actual demo table |
| Perception | NVIDIA Jetson |
| Robot | SO-101 arm, **taught poses, no IK**. Work zone 7 to 14 in |
| Display | Monitor at the station, readable from arm's length |

**Cell naming.** Columns A to C left to right as seen from the operator side. Rows 1 to 3 from front (nearest the operator) to back. So `A1` is front-left, `C3` is back-right. Levels count up from 0 (on the board).

**Cube supply.** The robot picks from a fixed tray with one taught slot per color. `place(color, cell, level)` means "pick a cube of that color from the tray and place it there."

---

## 4. System architecture

Python everywhere except the monitor UI (web).

| Module | Runs on | Owner | Job |
|---|---|---|---|
| `perception/` | Jetson | Jetson owner | Camera frames to `GridState` at about 10 Hz: tag homography, per-cell color and stack height, `arm_clear`, `hands_clear` |
| `brain/` | Laptop | Nikhil | Stabilizer, checker, extractor, gate, session state machine, journal, fake feed |
| `robot/` | Laptop wired to arm | Robot owner | Taught poses, robot HTTP API, stub backend |
| `ui/` | Laptop browser on monitor | Face owner | Monitor app: current step, ghost cube, hints, show-me clips, stage strips |
| `analytics/` | Databricks | Whoever frees up | Stretch: journal to Delta table, insights, hint write-back |

### Data flow

```
perception --(ws :8765, GridState JSON ~10 Hz)--> brain --(ws :8766, SessionState JSON)--> ui
                                                   |
                                                   +--(HTTP :8770)--> robot
                                                   |
                                                   +--> data/sessions/<id>.jsonl (journal)
```

---

## 5. Contracts (the three interfaces)

These live in `shared/schemas.py` and `shared/config.py`. **Changing them requires the whole team to agree.** Everyone builds against fakes of each other's side.

### 5.1 GridState (perception to brain)

```json
{
  "t": 1728580000.123,
  "board": { "A1": ["red"], "B2": ["blue", "yellow"], "C3": [] },
  "arm_clear": true,
  "hands_clear": true,
  "confidence": { "A1": 0.97, "B2": 0.91 },
  "frame_id": 18234
}
```

- `board` always contains all 9 cells. Lists go bottom to top. Empty cell = `[]`.
- Colors are exactly `"red" | "blue" | "yellow" | "green"`. Anything unreadable becomes `"unknown"`.
- `arm_clear` = the arm is at home and nothing of the arm is over the grid.
- `hands_clear` = no hand over the grid.

### 5.2 Lesson (brain output, stored in `lessons/*.json`)

```json
{
  "lesson_id": "staircase-v1",
  "title": "Staircase",
  "source_video": "videos/expert_staircase.mp4",
  "steps": [
    { "color": "red", "cell": "A1", "level": 0, "t_video": 4.2, "narration": "Start with red in the front-left corner." }
  ],
  "final_board": { "A1": ["red"], "...": [] },
  "gate": { "passed": true, "checks": [ { "name": "replay_matches_final_frame", "passed": true } ] }
}
```

- `t_video` is when the cube appeared in the expert video. The UI uses it to cut the "show me" clip.
- `narration` comes from Whisper. When narration and grid disagree, **the grid wins** and the gate records the disagreement.

### 5.3 Robot API (brain to robot, HTTP JSON on :8770)

| Call | Body | Meaning |
|---|---|---|
| `POST /place` | `{color, cell, level}` | Pick that color from the tray, place it at cell and level |
| `POST /pick` | `{cell, level}` | Remove the cube at cell and level, drop it back in the tray |
| `POST /hover` | `{cell}` | Point: move above the cell and hold (used to show a human where to go) |
| `POST /home` | `{}` | Return to the home pose (clears the camera view) |
| `GET /status` | | `{busy, at_home, last_error}` |

Every call returns `{"ok": true}` or `{"ok": false, "error": "<reason>"}`. Calls block until the motion finishes. `robot/stub.py` implements the same API by editing a fake board, so brain can test retry logic with no arm.

### 5.4 SessionState (brain to UI, ws :8766)

```json
{
  "mode": "human",
  "lesson_id": "staircase-v1",
  "step_index": 3,
  "expected_step": { "color": "blue", "cell": "B2", "level": 1 },
  "verdict": "WRONG_CELL",
  "fault_cell": "B3",
  "hint_level": 2,
  "hint_text": "Close! Move the blue cube one cell to the left.",
  "show_me": { "video": "videos/expert_staircase.mp4", "start": 11.0, "end": 15.0 },
  "progress": { "human": ["pass", "pass", "retry", "active"], "robot": ["pass", "pass", "pass", "pass"] },
  "board": { "...": [] },
  "lesson_title": "Staircase",
  "final_board": { "...": [] },
  "human_side": "row3",
  "phase": "running",
  "expected_cell": "B2",
  "expected_level": 1,
  "robot_status": "idle"
}
```

The last seven fields are optional (older producers omit them; defaults in `shared/schemas.py`):

| Field | Meaning |
|---|---|
| `lesson_title`, `final_board` | From the lesson, for the overview before a session starts |
| `human_side` | `HUMAN_SIDE` from `brain/settings.py`: which edge the student stands at, so the UI draws the board their way |
| `phase` | `ready` (waiting for start, nothing is checked), `running`, `complete`, or `stopped` (robot needs a hand) |
| `expected_cell`, `expected_level` | The slot that needs a cube, from the latest check. Usually the current step's target; for a knocked older cube, where that cube belongs |
| `robot_status` | `idle`, `placing`, `checking` (robot mode, waiting for the camera), `fixing` (pick, re-place, or a hint hover), `homing`, or `stopped` |

**UI commands.** UI clients may send JSON on the same socket:

| Message | Effect |
|---|---|
| `{"cmd": "start"}` | Begin the lesson. Refused while the stable board has cubes on it; `hint_text` then reads "Please clear the board to start." |
| `{"cmd": "reset"}` | End the current session in the journal (`session_end`, outcome `reset`) and open a new session in `ready` at step 0 |

`python -m brain.server --autostart` skips `ready` and starts at once.

### 5.5 Verdicts

`PASS`, `WRONG_CELL`, `WRONG_COLOR`, `WRONG_LEVEL`, `EXTRA`, `MISSING`, `WAITING`.

**Unknown is never a pass.** Low confidence, hands in frame, the arm away from home, or any `"unknown"` cell means `WAITING`. A false pass in front of a judge costs more than a one-second delay.

---

## 6. Core logic (brain)

1. **Stabilizer.** Accept a board only when it stays identical for `STABLE_FRAMES` frames (default about 0.5 s) with `hands_clear` and `arm_clear` both true.
2. **Checker.** Compare the **whole** stable board to the expected board after step k (not only the newest cube), so it also catches knocked or moved cubes. Return the verdict plus `fault_cell`.
3. **Extractor.** Diff consecutive stable boards. Each added cube becomes a step with its timestamp. A removed cube deletes the step that placed it. This handles expert corrections, which the VLM could not. The same extractor powers expert video, judge teaches mode, and teardown (steps reversed).
4. **Gate.** Replay the steps on an empty board and confirm the result matches the final stable board from the video. Cross-check Whisper colors against grid colors. Store results in `lesson.gate`. Nothing reaches a student unchecked.
5. **Session state machine.**
   - **Human mode:** wait for a stable board, check, advance on `PASS`, otherwise escalate hints: (1) text, (2) pulsing ghost cube on the target cell, (3) "show me" clip from the expert video, (4) robot hovers over the correct cell.
   - **Robot mode:** `place`, then `home`, wait for a stable board, check. On failure: `pick` the faulty cube and re-place. **Max 2 retries, then stop and flag the operator.**
6. **Journal.** Append one JSON line per event to `data/sessions/<session_id>.jsonl`. Never rewrite a line. The journal is the source of truth for the eval chart, crash recovery, debugging replay, and analytics.

```json
{"t": 1728580001.2, "session": "s-0042", "mode": "robot", "kind": "verdict", "data": {"step": 3, "verdict": "WRONG_CELL", "fault_cell": "B3", "retry": 1}}
```

Journal `kind` values: `session_start`, `board`, `verdict`, `step_advance`, `hint`, `robot_cmd`, `robot_result`, `gate`, `note`, `session_end`.

---

## 7. Repo layout

```
apprentice/
  README.md
  AGENTS.md
  shared/
    schemas.py          # dataclasses for GridState, Step, Lesson, SessionState, Verdict
    config.py           # cells, colors, ports, thresholds
  perception/           # Jetson owner
    camera.py
    tags.py             # AprilTag detection + homography
    cells.py            # per-cell sampling, stack height, color classification
    occlusion.py        # arm_clear, hands_clear
    publish.py          # ws server :8765
    calibrate_color.py  # run at the demo table
  brain/                # Nikhil
    fake_feed.py        # scripted and keyboard GridState publisher (ship first)
    stabilizer.py
    checker.py
    extractor.py
    gate.py
    narration.py        # Whisper transcription + alignment
    session.py          # state machine, hint ladder, robot retry
    journal.py
    robot_client.py
    server.py           # ws server :8766 for the UI
  robot/                # Robot owner
    poses.json          # taught poses: home, tray slots, per cell per level, hover per cell
    teach.py            # record poses
    driver.py           # SO-101 control
    api.py              # HTTP :8770
    stub.py             # same API, fake board
  ui/                   # Face owner
  lessons/              # extracted lesson JSON files
  videos/               # expert recordings
  data/sessions/        # journals (gitignored except samples)
  analytics/            # Databricks notebook (stretch)
  docs/
    tried-first.md      # what we tried, what happened, what we switched to
    challenges-log.md   # every bug over 20 minutes, one line each
    pitch.md
    architecture.md
  tests/
```

---

## 8. How to run (fill in as modules land)

Run everything from the repo root. The default demo lesson is `lessons/sample-flat-v1.json` (5 cubes, all on the board, red, blue and yellow). Commands that take `--lesson` use it when you leave the flag out.

```bash
# Tests (Python 3.11+)
pip install -r requirements.txt pytest
pytest

# Fake perception feed (GridState on ws :8765 at ~10 Hz)
python -m brain.fake_feed --script tests/scripts/staircase_clean.json            # add --loop to repeat
python -m brain.fake_feed --script tests/scripts/staircase_with_mistake.json
python -m brain.fake_feed --script tests/scripts/jitter_and_hands.json --seed 1  # --seed makes jitter repeatable
python -m brain.fake_feed --interactive  # type: place red B2 | remove B2 | hands | arm | jitter | clear | show | quit
python -m tests.listen                   # second terminal: print each GridState on one line

# Live check: stabilizer + checker against the feed (second terminal; prints step, verdict, hint)
# The scripted feeds build the staircase, so pass its lesson. With --interactive, the default lesson works.
python -m brain.check_live --lesson tests/fixtures/staircase-v1.json
python -m brain.check_live

# Brain server, no hardware: stabilizer + session + journal, SessionState on ws :8766
# It waits in phase "ready" until a UI sends {"cmd": "start"} (README 5.4); --autostart starts at once.
# Journal: data/sessions/<session_id>.jsonl. Hint directions follow HUMAN_SIDE in brain/settings.py.
# Human mode: start a feed, then the server (second terminal)
python -m brain.fake_feed --script tests/scripts/staircase_with_mistake.json
python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human --autostart  # hints stop at level 3
python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human --robot fake --autostart  # level 4 hover (fake arm)
# Robot mode: fake arm plus an in-process feed of its board, one command, no feed needed
python -m brain.server --mode robot --robot fake --grid sim --autostart  # default demo lesson
python -m brain.server --mode robot --robot fake --grid sim --fake-wrong-on 2 --autostart  # 2nd place lands on the wrong cell, robot fixes it
python -m brain.server --mode robot --robot fake --grid sim --fake-wrong-on 2 --fake-wrong-on 3 --fake-wrong-on 4 --autostart  # hits the retry cap and stops

# Monitor UI (ui/): reads SessionState from brain.server on ws :8766, no network needed
python ui/serve.py      # then open http://localhost:8000 and press F11; keys 2 / 3 switch 2D / 3D board

# Robot module on its own (LeRobot venv, see robot/README.md and docs/SETUP_NOTES.md)
python -m robot.scripts.tests.scene_snapshot robot/samples/overhead_02.png   # offline, no hardware
python -m robot.scripts.tests.pick_place_test red_cube B2                     # moves the arm
```

Planned, not built yet:

```bash
python -m brain.extractor --video videos/expert_staircase.mp4 --out lessons/staircase-v1.json  # planned: lesson from an expert video
python -m perception.publish   # planned: real GridState on ws :8765
python -m robot.api            # planned: robot HTTP API on :8770
# planned full system: perception.publish + robot.api, then the brain (this flag already works;
# with --robot http, arm_clear also requires the robot's GET /status at_home)
#   python -m brain.server --mode robot --robot http://localhost:8770 --grid ws://<perception-host>:8765
```

---

## 9. Gates and checkpoints

| Gate | Pass condition | Decision |
|---|---|---|
| Arm go/no-go | 8 of 10 place tries across different cells | Pass: robot is the hero. Fail: human is the hero, robot does a narrow controlled role or runs in sim |
| Stacking gate | 4 of 5 two-high stacks (add three-high tries if the staircase needs them) | Fail: lesson stays one level for the robot |
| First integration | Real Jetson feed drives brain and UI for one human run | ASAP |
| Demo ready | 10 clean end-to-end runs in a row | Only then start "Next" features |

Log every try of every gate to the journal (or `data/trials.csv` before brain exists): timestamp, task, cell, pass/fail, failure reason. That file becomes the eval chart.

---

## 10. Judging rubric and how we hit it

Knight Hacks scores 1 to 5 on each:

| Category | What a 5 needs | Our plan |
|---|---|---|
| Originality | Never seen before | Lead with "one lesson, two students, one grader," never with "a robot stacks blocks" |
| Technical Understanding | Full understanding, with graphs or mind maps in the demo; admitting tools and struggles counts | Architecture diagram and eval chart in the demo; a 30 s "what broke and what we learned" segment; credit coding agents by name; everyone can explain the whole pipeline in 60 s |
| Functionality | 90% of what we **mention** works, fluid, end to end | Never mention a feature that isn't solid; call everything else "next" |
| Design | Seamless UX | The judge is the user: big text, one instruction at a time, clear ghost cube, clean station, one visual style |
| Wow | Acquire it? Hope for humanity? | Robot catches and fixes a mistake live; judge teaches the robot; one-line answer: factories train workers and program robots separately, Apprentice does both from one recording |

---

## 11. What we tried first (keep adding rows)

| Idea | What happened | What we did instead |
|---|---|---|
| VLM extracts steps from the expert video | Missed the correction in 1 of 3 test videos | Deterministic grid diffs plus Whisper; VLM only as a helper |
| Duplo blocks | Glare, studs confuse color, mixed boxes | Matte painted wooden cubes |
| Isaac Sim for the robot | Needs RT-core GPUs and 16 GB minimum; A100/H100 unsupported | Real arm with taught poses; MuJoCo only as backup |
| RL training in sim | Days of work plus sim-to-real gap | Taught poses now; imitation learning (LeRobot ACT) is the future path |

---

## 12. Credits

Built with help from coding agents. We will list which agent built what in the Devpost.
