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
`/home`, `/status`), and the `stub.py` fake backend. See [Decisions](#decisions-robot-owner-2026-10-10)
before building them: the robot's cell naming is transposed from README section 3 and the
shared contract, so the API needs an agreed convention or a translation at its boundary.

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

Scripts that move the arm are hands-free (`session.py`): a 3 s countdown, then the arm turns on
where it is, rises straight up and parks. At the end it sets itself down on the table just left of
the sheet, below its parking spot (keep that spot clear), and releases torque. Between task
steps it parks instead just off the nearest side edge of the sheet, 120 mm up (keep ~5 cm beside
the sheet clear at that height), at a spot checked to leave the whole grid in the camera's view. Motion streams at 50 Hz with smooth
speed ramps (`Arm.follow`). Ctrl+C stops the arm and holds it, then asks what to do; the 12V
plug is the hard stop.

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
| `touch_calibrate.py` | After the arm or sheet moves. `--full` once (16 crossings), then quick (4 corners). If the middle of the sheet is off after a quick run, `--middle` (B2's 4 corners) refits everything on those plus the quick run's corners. |
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

## Decisions (robot owner, 2026-10-10)

These were open questions because this module predates README sections 3 and 5. The robot
owner has decided all three for the robot side.

1. **Cell naming: the robot keeps its own convention.** The letter is the **row** (A nearest
   the arm, C farthest) and the number is the **column** (1 to 3 left to right, seen from
   above with the arm at the top):

   ```
   arm side -> A1  A2  A3
               B1  B2  B3
               C1  C2  C3
   ```

   **Integration warning.** README section 3, `shared/config.py` and `perception/cells.py`
   (decision D1) use the transposed convention: letter = column, number = row. Only A1, B2
   and C3 name the same square in both. A cell name passed from perception or brain to this
   module unchanged lands on the wrong square for the other six cells, with no error. Before
   the robot API (`:8770`) carries cell names between modules, the team has to agree on one
   of: perception and shared adopt this convention, or the robot API translates at its
   boundary. The `shared/` contract is frozen, so changing it needs team sign-off.
2. **Cubes: 48 mm foam cubes**, not the 1.25 in wooden cubes in README section 3.
   `CUBE_SIZE_MM`, the grasp, stack heights and colour calibration are set for them.
3. **Motion: IK**, not taught poses. The arm reaches any sheet-mm target from the fitted
   calibration (`calibration/`), so placement is not limited to recorded cells.

Still open: where the robot's working set of blocks comes from (a tray with one slot per
colour, a supply area beside the grid the camera searches, or blocks that start on the grid).

## Dependencies

The arm driver comes from LeRobot (Seeed fork), installed as an editable checkout **outside**
this repo, so there is no simple `pip install -r` for it. Follow
[`../docs/SETUP_NOTES.md`](../docs/SETUP_NOTES.md). Beyond LeRobot this code needs only
`opencv-python` (with `cv2.aruco`) and `numpy`, both of which arrive with it.
