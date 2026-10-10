# robot/

The robot module. Everything that touches the arm or the overhead camera.

Give it a target in **sheet millimetres** and it makes the arm put a cube there, then
photographs the result to check where the cube actually landed.

## Status

Working today, driven from the command line:

- Camera finds the cubes on the printed grid and reports positions in mm (4.4 mm mean error).
- The arm picks a cube and places it in any grid cell, or stacks it on another cube.
- Every run saves an annotated photo per phase plus an `events.jsonl`.

**Not built yet:** the HTTP API on `:8770` that `brain/` calls (`/place`, `/pick`, `/hover`,
`/home`, `/status`), and the `stub.py` fake backend. See [Open questions](#open-questions)
before building them, because three decisions in README section 3 conflict with what this
code does today.

## Running

Entry points run as modules **from the repo root**, so that `robot.*` imports resolve:

```bash
source ~/lerobot/.venv/Scripts/activate      # setup in ../docs/SETUP_NOTES.md

python -m robot.calibration_gui              # dashboard: camera/sheet check + one-click
                                              # launchers for every calibration step below

python -m robot.scripts.tests.scene_snapshot                            # what the camera sees
python -m robot.scripts.tests.scene_snapshot robot/samples/overhead_02.png   # offline, no hardware
python -m robot.scripts.tests.pick_place_test red_cube B2                # place red in cell B2
python -m robot.scripts.tests.pick_place_test red_cube --onto blue_cube  # stack red on blue
```

Run `scene_snapshot` on a sample image to check that a code change did not break perception.
It is the only entry point that needs no hardware.

### Scripts

`scripts/tests/`, in rough order of how much they trust the stack:

| Script | What it does |
|---|---|
| `first_move.py` | Connects, nudges one joint, returns. Proves the arm and COM port work. |
| `joint_tour.py` | Exercises each joint with small moves around a pose you set by hand. |
| `scene_snapshot.py` | Turns one overhead frame into cube positions in sheet mm. No arm needed. |
| `hover_test.py` | Checks IK: hovers above grid crossings and lowers to just above them. |
| `reach_check.py` | Stops above a cube and photographs where the gripper really ended up. |
| `pick_place_test.py` | Camera-guided pick and place, with a before and after check. |

`scripts/calibration/`:

| Script | When to run it |
|---|---|
| `touch_calibrate.py` | After the arm or sheet moves. `--full` once (16 crossings), then quick (4 corners). |
| `auto_calibrate.py` | After every touch calibration. The arm moves a cube through all 9 cells and fits a placement correction. |
| `recalibrate_gripper.py` | Only if the gripper's range drifts. |

### Modules

| Module | Role |
|---|---|
| `arm.py` | Safe joint-space motion. **All arm motion goes through here.** |
| `kinematics.py` | Forward and inverse kinematics from `urdf/so101_new_calib.urdf`, in plain numpy. |
| `workspace.py` | Turns a sheet-frame target into follower joint commands, using the calibration. |
| `skills.py` | `pick`, `place` and `park` in sheet mm. The layer a robot API would call. |
| `correction.py` | Closed-loop affine correction: where the arm actually puts things. |
| `arm_calibration.py` | Fits the sheet to arm-base transform and the joint zero offsets. |
| `recorder.py` | Per-run evidence: an annotated photo and a JSON event at each skill phase. |
| `perception/` | Camera capture, AprilTag sheet localisation, cube detection, scene building. |
| `assets/` | The printable grid PDF and its exact geometry in mm. |
| `calibration/` | Fitted rig numbers. Read [`calibration/README.md`](calibration/README.md) before moving anything. |

`perception/config.py` holds the measured rig constants: camera index, lens height, cube edge.
Measure them, do not guess, because they feed straight into position accuracy.

## Accuracy, last measured 2026-10-10

| Measurement | Result |
|---|---|
| Cube position from the camera | 4.4 mm mean error corrected, 7.2 mm raw |
| Sheet homography fit | 0.9 mm mean, 1.7 mm max, no lens undistortion |
| Touch calibration fit | 1.8 mm RMS over 16 points |
| Placement correction | 36 mm RMS from only 7 placements |

The placement correction is the weakest link and the first thing to improve. Re-run
`auto_calibrate` to fit it on all 9 cells.

## Open questions

This module predates README sections 3 and 5, so three things need a team decision. Each one
is a silent failure if nobody looks at it.

1. **Cell naming is transposed.** README section 3 says columns are A to C left to right and
   rows are 1 to 3 front to back, so `A1` is front-left. This code reads the letter as the
   **row** and the number as the **column**, with row A nearest the arm. `cell_center("B3")`
   therefore means middle-right here and middle-back in the README. The labels match, the
   cells do not, and only the diagonal agrees. One of the two has to change.
2. **Cube size.** README section 3 mandates 1.25 in (31.75 mm) matte wooden cubes in four
   colors. This code is calibrated for 48 mm foam cubes in three colors, which sets the grasp
   width, the stack heights and the parallax correction, and it has no green detector. Moving
   to 1.25 in cubes needs a re-measure of `CUBE_SIZE_MM` plus a colour recalibration.
3. **Taught poses versus IK.** README section 3 says taught poses, no IK. This module does
   full IK with a fitted calibration and hits 4.4 mm, which should clear the 8 of 10 arm
   gate. Keeping IK means the arm reaches arbitrary mm targets instead of only taught cells.
   Dropping it means rebuilding `place` on recorded poses.

## Dependencies

The arm driver comes from LeRobot (Seeed fork), installed as an editable checkout **outside**
this repo, so there is no simple `pip install -r` for it. Follow
[`../docs/SETUP_NOTES.md`](../docs/SETUP_NOTES.md). Beyond LeRobot this code needs only
`opencv-python` (with `cv2.aruco`) and `numpy`, both of which arrive with it.
