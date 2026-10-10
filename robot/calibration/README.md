# Calibration state for this rig

These files are committed on purpose. There is exactly one rig, and the fitted numbers are
slow and fiddly to reproduce. But they are not all equally portable: some describe the
hardware, others describe only where things happen to sit on the table right now.

## `arm_sheet_calibration.json`

| Field | Survives a move? | Why |
|---|---|---|
| `joint_offsets_deg` | Yes | Servo zero errors. They travel with the arm. |
| `tip_extension_mm` | Yes | Where the jaw tips really are. A property of the gripper. |
| `table_z_mm` | Mostly | Sheet surface height in the base frame. It changes if you remount the arm or change the table surface. |
| `sheet_to_base` | **No** | Where the sheet sits relative to the arm base. It goes invalid the moment either one moves. |

So if you nudge the arm or re-tape the sheet, you do not need the full calibration again. Run
the quick one, which refits only `sheet_to_base` and keeps the rest:

```bash
python -m robot.scripts.calibration.touch_calibrate          # quick, 4 corners, about 2 min
python -m robot.scripts.calibration.touch_calibrate --full   # full, 16 crossings, also refits offsets
```

Torque stays off for both. You move the arm by hand and the script records joint angles with
the closed jaw tips on known grid crossings. It backs the previous file up to `robot/outputs/`
first.

## `placement_correction.json`

A fitted affine map, `observed = A * commanded + b`: where the arm actually puts a cube, as the
camera sees it. This one is setup-dependent, because it absorbs whatever `sheet_to_base` and
the camera geometry get slightly wrong. It is only valid for the calibration it was fitted
against.

Re-run it after any touch calibration:

```bash
python -m robot.scripts.calibration.auto_calibrate     # about 10 min, the arm moves a cube through 9 cells
```

There is currently **no** correction file: the earlier fit (7 placements, 36 mm RMS) included two
placements where the cube had been moved by hand, and it scaled x by 0.35. It was moved to
`robot/outputs/placement_correction_BROKEN_36mm_rms.json`. Without the file, no correction is
applied. `auto_calibrate` now drops failed placements (over 30 mm off, or outliers after fitting)
and refuses to save a fit that is implausible or still over 6 mm RMS. Don't move the cube by hand
while it runs.

## Order of operations after moving anything

1. Did the camera move? Re-measure the lens height into `perception/config.py`.
2. Did the arm or sheet move? Run `touch_calibrate`, with `--full` if you remounted the arm.
3. Then always run `auto_calibrate`.
4. Verify with `python -m robot.scripts.tests.pick_place_test red_cube B2`.

LeRobot's own servo calibration lives outside the repo, in
`~/.cache/huggingface/lerobot/calibration/`. See `../../docs/SETUP_NOTES.md`.
