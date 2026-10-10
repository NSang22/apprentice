"""Logic shared by every front end for robot/ calibration and control - the Tkinter
desktop GUI (calibration_gui.py) and the phone-facing web server (web_control.py).

Nothing UI-framework-specific lives here: file I/O, the step/command registry, and
the interpreter-resolution logic. Both front ends import from this module instead of
each defining their own copy - see jetson-kushi/README.md's "one rule" for why that
matters (we already duplicated camera/grid logic once today and paid for it).
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROBOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROBOT_DIR.parent

ARM_SHEET_CALIB = ROBOT_DIR / "calibration" / "arm_sheet_calibration.json"
PLACEMENT_CORR = ROBOT_DIR / "calibration" / "placement_correction.json"
CONFIG_PY = ROBOT_DIR / "perception" / "config.py"

PLACEMENT_RMS_OK_MM = 20.0  # robot/README.md: current 36mm is "provisional", flag anything that high

DEV_SAMPLE_IMAGES = [
    ROBOT_DIR / "samples" / "overhead_02.png",         # tags + 3 cubes detected, known good
    ROBOT_DIR / "samples" / "overhead_01_untaped.png",
]

OPEN_QUESTIONS = (
    "Cell naming is transposed vs README section 3 (letter=row here, letter=column in README).",
    "Calibrated for 48mm foam cubes, 3 colors (no green) vs README's 1.25in wood cubes, 4 colors.",
    "This module does full IK; README section 3 says taught poses, no IK.",
)


@dataclass
class Button:
    key: str     # unique id, used as the web terminal's URL/WebSocket path
    label: str
    module: str
    args: list[str] = field(default_factory=list)


@dataclass
class Step:
    id: str
    group: str  # "order" (robot/calibration/README.md's required sequence) or "extra"
    title: str
    desc: str
    buttons: list[Button] = field(default_factory=list)


# Mirrors robot/calibration/README.md's "Order of operations after moving anything",
# plus the diagnostic scripts from robot/README.md's scripts table. One definition,
# both front ends render it.
STEPS: list[Step] = [
    Step("1", "order", "Camera moved?",
         "Re-measure the lens height with a ruler and save it in rig constants. No script to run."),
    Step("2a", "order", "Arm nudged or sheet re-taped?",
         "Quick touch calibration: 4 corners, ~2 min. Keeps joint offsets, refits only where the sheet sits.",
         [Button("2a-quick", "Run quick", "robot.scripts.calibration.touch_calibrate")]),
    Step("2b", "order", "Arm remounted?",
         "Full touch calibration: 16 crossings, ~5 min. Also refits joint offsets and jaw-tip length.",
         [Button("2b-full", "Run full", "robot.scripts.calibration.touch_calibrate", ["--full"])]),
    Step("3", "order", "After any touch calibration",
         "Auto-calibrate: the arm moves one cube through all 9 cells, fits the placement correction. "
         "~10 min. Put one cube on the sheet first.",
         [
             Button("3-red", "Run (red)", "robot.scripts.calibration.auto_calibrate", ["red_cube"]),
             Button("3-blue", "Run (blue)", "robot.scripts.calibration.auto_calibrate", ["blue_cube"]),
             Button("3-yellow", "Run (yellow)", "robot.scripts.calibration.auto_calibrate", ["yellow_cube"]),
         ]),
    Step("4", "order", "Verify",
         "Camera-guided pick and place, with a before/after check against tolerance.",
         [Button("4-verify", "Run red->B2", "robot.scripts.tests.pick_place_test", ["red_cube", "B2"])]),
    Step("x1", "extra", "First move",
         "Connects and checks everything, without moving (safe to run any time).",
         [
             Button("x1-dry", "Dry run", "robot.scripts.tests.first_move", ["--dry-run"]),
             Button("x1-full", "Full (nudges wrist)", "robot.scripts.tests.first_move"),
         ]),
    Step("x2", "extra", "Joint tour",
         "Exercises each joint with small moves around a pose you set by hand.",
         [Button("x2-run", "Run", "robot.scripts.tests.joint_tour")]),
    Step("x3", "extra", "Recalibrate gripper",
         "Only if the gripper's range of motion has drifted. Torque stays off.",
         [Button("x3-run", "Run", "robot.scripts.calibration.recalibrate_gripper")]),
]

BUTTONS_BY_KEY: dict[str, Button] = {b.key: b for s in STEPS for b in s.buttons}


def find_lerobot_python() -> tuple[str | None, str]:
    """A Python interpreter that can import lerobot, for launching robot/ scripts.

    Prefers the interpreter already running this process (no venv juggling needed if
    launched correctly already). Falls back to the conventional venv location from
    docs/SETUP_NOTES.md (~/lerobot/.venv), since a front end may reasonably be started
    from the system Python instead. Returns (path_or_None, human-readable source).
    """
    if importlib.util.find_spec("lerobot") is not None:
        return sys.executable, "current interpreter"
    venv_python = Path.home() / "lerobot" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv_python.exists():
        return str(venv_python), f"found at {venv_python}"
    return None, f"not found (checked current interpreter and {venv_python})"


def run_module_detached(python_exe: str, module: str, *args: str) -> None:
    """Launch `python -m <module> <args>` in its own console window, detached (Tkinter
    front end only - the web front end uses run_module_interactive instead, since a
    detached console isn't reachable from a phone)."""
    cmd = [python_exe, "-m", module, *args]
    kwargs = {"cwd": str(REPO_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(cmd, **kwargs)


def start_module_process(python_exe: str, module: str, *args: str) -> subprocess.Popen:
    """Start `python -m <module> <args>` with stdin/stdout/stderr piped, text mode,
    unbuffered - for the web front end's interactive terminal-over-websocket bridge."""
    cmd = [python_exe, "-m", module, *args]
    return subprocess.Popen(
        cmd, cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def file_age_str(path: Path) -> str:
    if not path.exists():
        return "missing"
    age_s = time.time() - path.stat().st_mtime
    if age_s < 60:
        return "just now"
    if age_s < 3600:
        return f"{int(age_s // 60)} min ago"
    if age_s < 86400:
        return f"{age_s / 3600:.1f} hr ago"
    return f"{age_s / 86400:.1f} days ago"


def read_config_value(name: str) -> float | None:
    text = CONFIG_PY.read_text()
    m = re.search(rf"^{name}\s*:.*?=\s*([\d.]+|None)", text, re.MULTILINE)
    if not m or m.group(1) == "None":
        return None
    return float(m.group(1))


def write_config_value(name: str, value: float, note: str) -> None:
    text = CONFIG_PY.read_text()
    pattern = rf"(^{name}\s*:.*?=\s*)(?:[\d.]+|None)(.*)$"

    def repl(m: re.Match) -> str:
        return f"{m.group(1)}{value}  # {note}"

    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Could not find {name} in {CONFIG_PY}")
    CONFIG_PY.write_text(new_text)
