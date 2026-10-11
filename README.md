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
| Tags | AprilTag 36h11, IDs 0 to 3 at the corners: 0 at A1, 1 at A3, 2 at C3, 3 at C1 (tags 0 and 1 on the arm's edge, 2 and 3 on the student's) |
| Cubes | Playskool silicone cubes, 2 each of red / orange / yellow / teal / skyblue / purple. Calibrate their colors with `perception.calibrate` (section 8) whenever the lighting changes |
| Camera | Webcam on a tripod, fixed above the grid. Never move it after calibration. Recalibrate color at the actual demo table |
| Perception | NVIDIA Jetson |
| Robot | SO-101 arm, **taught poses, no IK**. Work zone 7 to 14 in |
| Display | Monitor at the station, readable from arm's length |

**Cell naming** (as printed on the sheet, and as the robot uses it). The letter is the **row**: row A runs along the arm's edge, row C along the student's edge (the student stands opposite the arm). The number is the **column**: 1 to 3 left to right as the student sees it. So `A1` is far-left from the student (at tag 0), `A3` far-right (tag 1), `C3` near-right (tag 2), `C1` near-left (tag 3). Levels count up from 0 (on the board).

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
  "board": { "A1": ["red"], "B2": ["skyblue", "yellow"], "C3": [] },
  "arm_clear": true,
  "hands_clear": true,
  "confidence": { "A1": 0.97, "B2": 0.91 },
  "frame_id": 18234
}
```

- `board` always contains all 9 cells. Lists go bottom to top. Empty cell = `[]`.
- Colors are exactly `"red" | "orange" | "yellow" | "teal" | "skyblue" | "purple"`. Anything unreadable becomes `"unknown"`.
- `arm_clear` = the arm is at home and nothing of the arm is over the grid.
- `hands_clear` = no hand over the grid.

### 5.2 Lesson (brain output, stored in `lessons/*.json`)

```json
{
  "lesson_id": "staircase-v1",
  "title": "Staircase",
  "source_video": "videos/expert_staircase.mp4",
  "steps": [
    { "color": "red", "cell": "A1", "level": 0, "t_video": 4.2, "narration": "Start with red in the far-left corner." }
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
  "expected_step": { "color": "skyblue", "cell": "B2", "level": 1 },
  "verdict": "WRONG_CELL",
  "fault_cell": "B3",
  "hint_level": 2,
  "hint_text": "Close! Move the skyblue cube one cell to the left.",
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

An unreadable block does not freeze the lesson: the stabilizer still settles a board in which that cell reads `"unknown"` (a block below `MIN_CONFIDENCE` counts as unknown), and the verdict is `WAITING` with `fault_cell` set and the hint "I can't tell the color of the block on B2. Try nudging it to the center of the square." `EXTRA` means a block the rest of the lesson has no open slot for: "Please take the purple block off B2.", with no `expected_cell` (the UI shows a lift-off marker, never an arrow). A block whose color an open slot still needs is `WRONG_CELL`, with `expected_cell` pointing at that slot.

---

## 6. Core logic (brain)

1. **Stabilizer.** Accept a board only when it stays identical for `STABLE_FRAMES` frames (default about 0.5 s) with `hands_clear` and `arm_clear` both true.
2. **Checker.** Compare the **whole** stable board to the expected board after step k (not only the newest cube), so it also catches knocked or moved cubes. Return the verdict plus `fault_cell`.
3. **Extractor.** Diff consecutive stable boards. Each added cube becomes a step with its timestamp. A removed cube deletes the step that placed it. This handles expert corrections, which the VLM could not. The same extractor powers expert video, judge teaches mode, and teardown (steps reversed).
4. **Gate.** Replay the steps on an empty board and confirm the result matches the final stable board from the video. Cross-check Whisper colors against grid colors. Store results in `lesson.gate`. Nothing reaches a student unchecked.
5. **Session state machine.**
   - **Progress comes from the board.** On every stable board, `step_index` is the first step whose slot does not hold its color. Taking an earlier block off moves the lesson back to that step (journal `step_back`); putting it back moves it forward again. Mistakes, retries, and hint levels are kept per step.
   - **Human mode:** wait for a stable board, check, advance on `PASS`, otherwise escalate hints: (1) text, (2) pulsing ghost cube on the target cell, (3) "show me" clip from the expert video, (4) robot hovers over the correct cell.
   - **Robot mode:** `place`, then `home`, wait for a stable board, check. On failure: `pick` the faulty cube and re-place. **Max 2 retries, then stop and flag the operator.**
6. **Journal.** Append one JSON line per event to `data/sessions/<session_id>.jsonl`. Never rewrite a line. The journal is the source of truth for the eval chart, crash recovery, debugging replay, and analytics.

```json
{"t": 1728580001.2, "session": "s-0042", "mode": "robot", "kind": "verdict", "data": {"step": 3, "verdict": "WRONG_CELL", "fault_cell": "B3", "retry": 1}}
```

Journal `kind` values: `session_start`, `board`, `verdict`, `step_advance`, `step_back`, `hint`, `robot_cmd`, `robot_result`, `gate`, `note`, `session_end`.

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

### Demo startup

```bash
# 1. Jetson (ssh in; export DISPLAY=:0 first for the perception debug window). Kills stale publish /
#    robot.api / camera holders, finds the camera, starts perception.publish then robot.api in the
#    background (logs/publish.log, logs/robot.log), prints OK/FAIL per service.
#    Env: BLOCK_MM (default 47), CAMERA (skip the search), EXPOSURE. Stop: see the last line it prints.
scripts/jetson_up.sh                 # flags: --no-robot, --no-debug

# 2. Laptop (PowerShell). Once: copy scripts\jetson.env.example to scripts\jetson.env, set JETSON_HOST, JETSON_USER.
#    Frees ports 8765/8766/8770/8000, opens the SSH tunnel (8765, 8770), copies colors.json, waits for a
#    GridState, starts brain.server and ui/serve.py in their own windows, opens http://localhost:8000.
powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -Mode human   # or -Mode robot
#    One command for both machines: -StartJetson ssh-es in, git pulls and runs scripts/jetson_up.sh first.
#    -Lesson picks the lesson. -Sim needs no Jetson: simulated board, fake arm, brain + UI.
powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -StartJetson -Lesson lessons\stack-test-v2.json
powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -Sim -Lesson lessons\stack-test-v2.json
#    SSH key once, so nothing asks for a password (PowerShell):
#    ssh-keygen -t ed25519   (Enter through the prompts)
#    type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <user>@<jetson> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 3. Ready check (either machine): perception rate, hands/arm clear, board, brain phase and step, robot status.
#    Prints READY TO DEMO (exit 0) or the list of problems (exit 1). Never sends a command.
python scripts/check.py
```

Run everything from the repo root. The default demo lesson is `lessons/sample-flat-v1.json` (6 blocks, all on the board, red / purple / yellow / teal only, no two same-color blocks touching). Commands that take `--lesson` use it when you leave the flag out.

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
# It waits in phase "ready" until a UI sends {"cmd": "start"} (README 5.4); --autostart starts when the first grid frame arrives.
# Journal: data/sessions/<session_id>.jsonl. Hint directions follow HUMAN_SIDE in brain/settings.py.
# Human mode: start a feed, then the server (second terminal)
python -m brain.fake_feed --script tests/scripts/staircase_with_mistake.json
python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human --autostart  # hints stop at level 3
python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human --robot fake --autostart  # level 4 hover (fake arm)
# Robot mode: fake arm plus an in-process feed of its board, one command, no feed needed
python -m brain.server --mode robot --robot fake --grid sim --autostart  # default demo lesson
python -m brain.server --mode robot --robot fake --grid sim --fake-wrong-on 2 --autostart  # 2nd place lands on the wrong cell, robot fixes it
python -m brain.server --mode robot --robot fake --grid sim --fake-wrong-on 2 --fake-wrong-on 3 --fake-wrong-on 4 --autostart  # hits the retry cap and stops

# Real perception: GridState on ws :8765 from the overhead camera (or a video)
# One-time setup per rig and lighting (perception/calibrate.py): (1) clear the board, SPACE: grid lock
# + empty board (all 4 tags needed); (2) each color in play in B2, SPACE; writes perception/calib/{grid_lock.json,
# empty_board.png,colors.json}. On Linux it also locks exposure and white balance (--exposure, --wb).
# Blocks are found with robot.perception (segment.detect + cubefit, no vision model). --block-mm is the real
# block edge (without it: the robot's CUBE_SIZE_MM, 48 mm foam, with a warning). --colors defaults to the
# robot's ACTIVE_COLORS (red, purple, yellow, teal); other colors read unknown.
python -m perception.calibrate --camera 1 --block-mm <edge mm>
# --debug: status band (GRID LOCKED n/4 / GRID LOST: moved / GRID NOT LOCKED), each stack "B2 L2 [red, teal] 0.93"
# (height, colors bottom to top), a red "B2 ? <reason>" when a stack can't be read (the brain sees unknown),
# each cell's visible top face outlined, hands in magenta. R overlays the board the model believes; E
# re-takes the empty board. The board is read cell by cell (perception/cellmodel.py: every cell and height is
# rendered and compared with the camera view; a cell's reading is the majority of its last 5 frames).
python -m perception.publish --camera 1 --block-mm <edge mm> --debug
python -m perception.publish --list-cameras        # Windows: which index/backend sees the grid
python -m perception.publish --video videos/demo.mp4
# publish owns the webcam and shares it: GET http://127.0.0.1:8771/frame is the newest raw frame (headers
# X-Shape, X-Timestamp), GET /health its age. robot/api.py reads the camera through it. --frame-port 0 turns it off.

# Robot API on the Jetson (README 5.3), in the lerobot venv, started after perception.publish (it reads the
# camera through publish's frame server, and refuses frames older than 0.5 s). One command at a time; every
# command and result goes to robot/outputs/api_log.jsonl. --lesson keeps blocks already in their lesson cell
# out of the supply; supply blocks wait below the grid. Then the brain on the Jetson (or over a tunnel):
#   python -m robot.api
#   python -m brain.server --mode robot --robot http://127.0.0.1:8770

# Real setup: camera on the Jetson, brain + UI on the laptop (same network)
#   Jetson:  v4l2-ctl --list-devices                                   # camera index N (/dev/videoN)
#            python -m perception.calibrate --camera N --block-mm <edge mm>   # DISPLAY=:0 over ssh; prints the exposure it picked
#            python -m perception.publish --camera N --block-mm <edge mm> --exposure <picked> --host 0.0.0.0 --debug
#   Laptop:  scp <jetson>:apprentice/perception/calib/colors.json perception/calib/   # UI shows the measured colors
#            python -m brain.server --mode human --grid ws://<jetson-ip>:8765
#            python ui/serve.py      # http://localhost:8000, F11, Space starts, R resets

# Monitor UI (ui/): reads SessionState from brain.server on ws :8766, no network needed
python ui/serve.py      # then open http://localhost:8000 and press F11; keys 2 / 3 switch 2D / 3D board
#   Space on the ready screen plays the intro preview (Space again skips it), then the lesson starts.
#   L opens the lesson picker (newest on top, "new" if under 10 minutes old): arrows choose, Enter loads, L closes.
#   B on a finished human lesson hands it to the robot (needs --robot). R starts over. Each load or switch is its own session.

# Robot module on its own (LeRobot venv, see robot/README.md and docs/SETUP_NOTES.md)
python -m robot.scripts.tests.scene_snapshot robot/samples/overhead_02.png   # offline, no hardware
python -m robot.scripts.tests.pick_place_test red_cube B2                     # moves the arm
python -m robot.scripts.calibration.pin_camera                               # after moving the camera: clear the sheet first, then auto_calibrate
python -m robot.scripts.calibration.auto_calibrate --bank                    # also measures the arm's error in the cube bank (5 more spots, ~5 min)
python -m robot.scripts.tests.run_task "red->C2" "purple->on:red" "yellow->on:purple"   # builds from the cube bank right of the sheet
python -m robot.scripts.tests.run_task --plan robot/plans/row_tower.json         # the demo line with a tower, from the bank (the file says how to refill it)

# Analytics (offline, standard library only)
python -m analytics.fake_journal --seed 7   # synthetic_*.jsonl journals in data/sessions/, written by the real brain; --lesson picks the lesson
python -m analytics.report --sessions data/sessions --out data/analytics/metrics.json   # every metric, per provenance; add --tasks robot/outputs/tasks, --include-synthetic
python -m analytics.record_fixtures       # re-record tests/fixtures/brain_journal_{human,robot}.jsonl from the real brain (about a minute)
# The fixtures are never edited by hand: when the brain's journal changes, re-record them and commit the result.
```

Planned, not built yet:

```bash
python -m brain.extractor --video videos/expert_staircase.mp4 --out lessons/staircase-v1.json  # planned: lesson from an expert video
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
