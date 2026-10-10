# jetson-kushi/

The Jetson deployment environment. This is where you `cd` after cloning the repo
onto the Jetson - it is not a second copy of the project.

## The one rule

**Nothing in here duplicates logic from `robot/`, `perception/`, `brain/`, or `ui/`.**
Those stay the single source of truth, imported and run unchanged on the Jetson exactly
as on a dev laptop. We already paid the cost of getting this wrong once today -
`perception/setup_gui.py` quietly duplicated camera/grid work that `robot/perception/`
already did, and better, before anyone noticed. Don't repeat it here.

This folder only holds:
- Jetson-specific dependencies (`requirements.txt`, `setup.sh`)
- Environment configuration (`env.example` - serial port, etc.)
- Launch / systemd service files (once `robot/api.py` and `brain/server.py` exist -
  see the fault-isolation discussion: each should be its own supervised process)
- This README

If the Jetson genuinely needs to *behave* differently from Windows (different camera
backend, different serial port), that's a small, backward-compatible branch in the
original file - `sys.platform` check, or an environment variable defaulting to the
current value - never a copied file in here. Already done, as the pattern to follow:
- `robot/perception/camera.py` picks DSHOW vs V4L2 by `sys.platform` automatically.
- `robot/arm.py` and the calibration/test scripts read `ROBOT_PORT` (env var,
  defaults to `"COM7"` so Windows behavior is unchanged).

## Setup

```bash
git clone <repo-url> apprentice
cd apprentice/jetson-kushi
./setup.sh
cp env.example .env     # then edit ROBOT_PORT for this specific board
source .env
cd ..
python -m robot.scripts.tests.first_move --dry-run   # sanity check, no motion
```

## Open, unverified questions

Don't assume these - confirm on the real board before trusting anything built on top:

- Does `lerobot` + its `torch`/`feetech` deps actually install cleanly on this
  JetPack version? (Good evidence this works on Jetson Orin NX in general - an
  active SO-ARM101 + LeRobot + Jetson tutorial exists - but not verified on this
  specific board yet.)
- Real `ROBOT_PORT` for this board. `COM7` was the Windows rig; the Linux path will
  differ and (per `docs/SETUP_NOTES.md`) isn't guaranteed stable across reboots or
  replugs - find it with `python -c "from serial.tools import list_ports; ..."`
  (full command in `env.example`) and match by board serial, not by path.
- Real `CAMERA_INDEX` for this board - V4L2 enumerates differently than DirectShow,
  don't assume index 1 carries over. Check with `v4l2-ctl --list-devices`.
- IK solve time on Jetson's CPU vs. the ~8s measured on a laptop
  (`docs/SETUP_NOTES.md`) - profile this early, it directly affects demo pacing.
- Use JetPack's preinstalled OpenCV, not `pip install opencv-python` - the pip wheel
  lacks GStreamer/CUDA support that the system one has. See `requirements.txt`.
