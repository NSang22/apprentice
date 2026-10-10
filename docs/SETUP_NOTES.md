# SO-ARM101 Setup Notes

## Environment
- LeRobot (Seeed fork) at `~/lerobot` (`C:/Users/taher/lerobot`), version 0.6.2
- Venv: `~/lerobot/.venv`, Python 3.12.3, created with `uv venv --python 3.12 .venv`
- Installed with: `uv pip install --python .venv/Scripts/python.exe -e ".[feetech]"`
- torch 2.11.0+cpu (CPU only). System Python 3.14 untouched.
- Activate (Git Bash): `source ~/lerobot/.venv/Scripts/activate`
- Robot import: `from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig`

## Hardware map
| Arm | Color | Driver board | Power | COM port | Board serial | LeRobot id |
|---|---|---|---|---|---|---|
| Follower (gripper jaw) | white | white base | **12V only** | `COM7` | `5B79080156` | `my_follower` |
| Leader (trigger handle) | black | black base | **5V only** | `COM5` | `5B79080010` | `my_leader` |

### Cameras (`lerobot-find-cameras opencv`, DSHOW backend)
| Index | What it sees | Use |
|---|---|---|
| `1` | **Kit USB camera, mounted top-down over the workspace** (black mat, arm, blocks) | Workspace camera |
| `0` | Another webcam facing the user | Ignore |

- Kit camera: up to 1920x1080 (MJPG). Default capture is 640x480. Frame rate drops in dim light (auto-exposure).
- Indices can shift if cameras are re-plugged. Re-run `lerobot-find-cameras opencv` and check the saved
  images in `robot/outputs/captured_images` to confirm which index is top-down.
- Test objects: three foam cubes (red, blue, yellow) on a black mat.

- Close the Windows Camera app (and anything else using the camera) before running scripts.
  While another app holds the camera, OpenCV fails to open index 1.

### Workspace sheet (AprilTag grid)
- `robot/assets/Apprentice_AprilTag_Grid.pdf` on US Letter, taped to the mat under the camera. **Do not move it.**
- Tags: `DICT_APRILTAG_36h11`, IDs 0-3, black square 31.75 mm. 3x3 grid of 57.15 mm cells.
- Exact geometry (tag corners in mm, sheet frame) is in `robot/assets/workspace_sheet.json`, extracted from the PDF vectors.
- Check on 2026-10-10 at 1920x1080: tags 1-3 detected (tag 0 hidden by the arm). The image-to-sheet
  homography fit to 0.9 mm mean / 1.7 mm max, with no lens undistortion yet.
- Still to do: confirm the 2.00 in scale bar measures 50.8 mm on the printout.

### Cube detection (`python -m robot.scripts.tests.scene_snapshot [image]`, code in `robot/perception/`)
- Color detector (HSV), restricted to the sheet area plus 40 mm. One cube per color on the sheet.
- Measured: cube edge 48 mm; camera lens about 468 mm above the table (in `robot/perception/config.py`).
- Height correction: the cube blob centroid is mapped at mid-cube height (24 mm), using the camera
  centre estimated from the tags plus the measured lens height.
- Accuracy test 2026-10-10 (cubes hand-centered in B1/B3/C2, all 4 tags visible): mean 7.2 mm
  uncorrected, **4.4 mm corrected**. Blue and yellow were within 3 mm. Red was off 7.5 mm in x, but
  the image shows it was placed left of center.
- Weak point: the camera centre relies on a hand-measured height and an assumed focal length and
  principal point. If the camera moves, re-measure the height, or do a one-time intrinsic calibration.

- Driver boards: WCH CH343 USB-serial (VID:PID 1A86:55D3). Windows driver was already present.
- Boards enumerate on USB power alone; motors need the barrel-jack supply.
- COM numbers can change if boards move to different USB ports. Re-check with:
  `python -c "from serial.tools import list_ports; [print(p.device, p.serial_number) for p in list_ports.comports()]"`
  and match by board serial.

## Calibration
Files live in `~/.cache/huggingface/lerobot/calibration/`.

**Follower** (done): `robots/so_follower/my_follower.json`
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=my_follower
```
- Torque is off during calibration, so the arm falls under gravity. Hold it in the middle pose and press Enter.
- The first run missed the gripper range (2047-2048). Fixed with `python -m robot.scripts.calibration.recalibrate_gripper`,
  which re-records only the gripper range. Result: 2034-3479.
- Units: body joints in degrees (`use_degrees=True`), gripper 0-100 (0 = closed).

**Leader**: not yet calibrated. Command when ready (5V brick on the black board):
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_leader
```

## Scripts in this folder (run with the venv active)
| Script | What it does |
|---|---|
| `robot/scripts/tests/first_move.py` | Connect, print joints, wrist_roll +10 deg and back, disconnect. **Working.** `--hold`: pose by hand first. `--dry-run`: checks only, no torque. |
| `robot/scripts/tests/joint_tour.py` | Pose by hand, then small moves on each joint in turn (about 20 deg/s). |
| `robot/scripts/calibration/recalibrate_gripper.py` | Re-record only the gripper's range of motion. |

## Gotcha: stale Goal_Position at torque-on
With torque off, the motors keep an old `Goal_Position` (it read about -180 deg on every body
joint after calibration). LeRobot's `connect()`/`configure()` turns torque on without
resetting it, so the arm can jump toward that old value. The scripts here copy
`Present_Position` into `Goal_Position` while torque is still off, verify it, and only then
call `robot.configure()`. Do the same in any new control code. `lerobot-teleoperate` is
mostly safe because the leader's position is sent right away, but hold the follower near
the leader's pose when starting it.

## Joint check (robot/scripts/tests/joint_tour.py, 2026-10-09)
All six joints respond and track within about 1 deg when unloaded or moving with gravity.
Moving against gravity, they sag: elbow_flex about 2 deg, shoulder_lift about 4 deg at steady state.
The cause is LeRobot's soft position gain (P=16, I=0). Gravity pulls shoulder_lift toward
higher values. At full reach, 4 deg at the shoulder is about 2 cm at the gripper.
Options for the project's control code:
- Software outer loop: after a move, read Present_Position and nudge the goal by the error.
- Raise P for shoulder_lift and elbow_flex (`position_p_coefficient`, try 24-32) and watch for jitter.
- Keep `max_relative_target` above the expected sag. 5 deg clipped shoulder commands; the scripts use 10.

## Arm kinematics + sheet calibration (2026-10-10)
- placo (LeRobot's IK) has no Windows build, so it failed to install. Instead, `robot/kinematics.py` does
  numpy FK/IK from the SO-101 URDF (`robot/urdf/so101_new_calib.urdf`, from TheRobotStudio/SO-ARM100).
  LeRobot feeds degrees to this URDF as plain deg->rad.
- `python -m robot.scripts.calibration.touch_calibrate`: torque off, touch the closed jaw tips to 8 grid crossings, and it saves
  `robot/outputs/touch_points.json`. `robot/arm_calibration.py` then fits joint zero offsets, the
  sheet->base transform, the table height and the tip length. Saved to `robot/calibration/arm_sheet_calibration.json`.
- Result: elbow zero off by -9.4 deg vs the URDF (pan +3.3, lift -1.8, wrist -1.6). Fit RMS 2.2 mm.
  Leave-one-out: 0.5-4 mm inside the grid, 8-10 mm at the corners (extrapolation).
  Touching all 16 crossings would tighten the corners.
- IK limits use this arm's calibrated ranges minus 3 deg, not the URDF's nominal ones.
  The gripper tilts up to 21 deg from vertical at 120 mm over the far row; 0-3 deg at grasp height.
- `robot/arm.py`: all motion goes through it. It syncs goal to present before torque on, ramps moves,
  and corrects sag with an outer loop (2 rounds, max 6 deg).
- `python -m robot.scripts.tests.hover_test`: hovered at 120 mm and lowered to 10 mm over crossings r2c2 and r1c1.
  By eye, the tips sat right on the crossings. The model readback was within 0.7 mm at 9.2-9.5 mm height.
- **Re-run touch calibration if the arm base or the sheet moves.**

## Pick and place (2026-10-10)
- `python -m robot.scripts.tests.pick_place_test red_cube B3` (or `--onto blue_cube`): park, detect, pick, place, park,
  then verify with the camera. Evidence goes to `robot/outputs/runs/<time>_<label>/` (a photo per phase plus events.jsonl).
- The SO-101 has ONE fixed jaw and one moving jaw. The jaws close along gripper_frame **y**; the moving jaw
  sits on its **-y** side (confirmed from run photos). The fixed jaw goes 4 mm outside one cube face, with
  the cube centre half a cube toward the moving jaw. Gripper opens to 50% (70% swung the jaw out nearly flat).
- A descent that stops >10 mm above its target is treated as blocked: the arm backs off and stops.
- Travel moves swing around the base (pan axis at sheet (80, -65) mm). All 72 cell-to-cell moves plan OK.
- 6 placements (B1 x3, B3, C3, A1) all grasped (19-24% contact). Final errors 1.8-12 mm, location
  dependent (A1 worst), so this is calibration, not randomness.
- `python -m robot.scripts.calibration.auto_calibrate`: the arm moves one cube through all 9 cells, the camera measures each landing,
  and it fits an affine correction to `robot/calibration/placement_correction.json`. pick_place_test applies it automatically.
  Re-run whenever the arm, sheet or camera moves.

## Later / tabled
- IK planning is slow (~8 s per pick+place: numeric Jacobian in Python). An analytic Jacobian would speed it up.
- Lens calibration (intrinsics + distortion) to drop the hand-measured camera height. Needs a second
  printout of the tag sheet held in about 15 poses.

## Safety
- Power off (unplug barrel jack) before plugging or unplugging servo, USB or power cables.
- 12V → follower only, 5V → leader only.
