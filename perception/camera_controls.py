"""Lock the overhead camera's exposure and white balance (Linux / Jetson, through v4l2-ctl).

Auto exposure keeps re-deciding brightness as hands and blocks come and go, which shifts every
color, and in a bright room it blows the white sheet out (the tags wash to gray). So at start-up
the camera goes to manual exposure and fixed white balance:

  --exposure auto   (default) start from the camera's own value and step it down until the
                    brightest part of the picture (the sheet) is no longer clipped: the 99th
                    percentile of gray between TARGET_LOW and TARGET_HIGH. Prints the value; pass
                    it as --exposure N next time so every session looks the same.
  --exposure N      exposure_time_absolute (or exposure_absolute) = N, in the camera's units
                    (usually 100 us steps)
  --exposure keep   leave exposure alone
  --wb N            white_balance_temperature = N kelvin, auto white balance off
  (no --wb)         auto white balance off, frozen at whatever it has settled on

Control names differ between cameras and kernels, so each setting tries the known names and uses
whichever this camera lists. If setting fails, the camera's whole control list is printed.
Elsewhere (Windows dev laptops) this does nothing and says so.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Callable, Optional

import cv2
import numpy as np

AUTO_EXPOSURE = ("auto_exposure", "exposure_auto")
EXPOSURE = ("exposure_time_absolute", "exposure_absolute")
AUTO_WB = ("white_balance_automatic", "white_balance_temperature_auto")
WB = ("white_balance_temperature",)
MANUAL_EXPOSURE = 1         # V4L2_EXPOSURE_MANUAL
TARGET_LOW, TARGET_HIGH = 200.0, 240.0  # 99th percentile gray: the sheet bright but not clipped
TUNE_STEPS = 12
SETTLE_FRAMES = 6           # frames to drop after a change before measuring

_CTRL = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")


def available() -> bool:
    return sys.platform.startswith("linux") and shutil.which("v4l2-ctl") is not None


def device(index: int) -> str:
    return f"/dev/video{index}"


def parse_controls(text: str) -> dict[str, dict[str, str]]:
    """`v4l2-ctl --list-ctrls` output -> {name: {"type": "int", "min": "1", "value": "250", ...}}."""
    out = {}
    for line in text.splitlines():
        m = _CTRL.match(line)
        if m:
            fields = dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
            out[m.group(1)] = {"type": m.group(2), **fields}
    return out


def list_controls(dev: str) -> tuple[dict[str, dict[str, str]], str]:
    r = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls"], capture_output=True, text=True, timeout=5)
    return parse_controls(r.stdout), r.stdout + r.stderr


def set_control(dev: str, name: str, value: int) -> Optional[str]:
    """None on success, else the error text."""
    r = subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={name}={value}"], capture_output=True, text=True, timeout=5)
    return None if r.returncode == 0 and not r.stderr.strip() else (r.stderr or r.stdout).strip() or "failed"


def _first(controls: dict, names: tuple[str, ...]) -> Optional[str]:
    return next((n for n in names if n in controls), None)


def lock_exposure(dev: str, exposure: str = "auto", wb: Optional[int] = None,
                  grab: Optional[Callable[[], np.ndarray]] = None, log: Callable[[str], None] = print) -> Optional[int]:
    """Manual exposure and fixed white balance. Returns the exposure value used, if any."""
    if not available():
        log("[camera] exposure lock skipped: needs Linux with v4l2-ctl (sudo apt install v4l-utils)")
        return None
    controls, listing = list_controls(dev)
    errors = []

    awb = _first(controls, AUTO_WB)
    if awb:
        errors += [f"{awb}: {e}" for e in [set_control(dev, awb, 0)] if e]
    wb_name = _first(controls, WB)
    if wb is not None and wb_name:
        errors += [f"{wb_name}: {e}" for e in [set_control(dev, wb_name, wb)] if e]
    wb_now = list_controls(dev)[0].get(wb_name, {}).get("value") if wb_name else None
    log(f"[camera] white balance: auto off ({awb or 'no auto control'}), {wb_name or 'temperature'} = {wb_now or 'n/a'}")

    used = None
    if exposure != "keep":
        ae, exp = _first(controls, AUTO_EXPOSURE), _first(controls, EXPOSURE)
        if ae:
            errors += [f"{ae}: {e}" for e in [set_control(dev, ae, MANUAL_EXPOSURE)] if e]
        if exp is None:
            errors.append(f"no exposure control among {EXPOSURE}")
        elif exposure == "auto":
            if grab is None:
                errors.append("--exposure auto needs frames")
            else:
                used = tune_exposure(dev, exp, controls[exp], grab, log)
        else:
            e = set_control(dev, exp, int(exposure))
            errors += [f"{exp}: {e}"] if e else []
            used = None if e else int(exposure)
        log(f"[camera] exposure: manual ({ae or 'no auto control'}), {exp or 'exposure'} = {used if used is not None else 'unchanged'}")

    if errors:
        log("[camera] could not set: " + "; ".join(errors))
        log("[camera] this camera's controls:\n" + listing)
    return used


def p99_gray(frame_bgr: np.ndarray) -> float:
    return float(np.percentile(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 99))


def tune_exposure(dev: str, name: str, ctrl: dict[str, str], grab: Callable[[], np.ndarray],
                  log: Callable[[str], None] = print) -> Optional[int]:
    """Step exposure until the 99th percentile gray sits in [TARGET_LOW, TARGET_HIGH]."""
    lo, hi = int(ctrl.get("min", 1)), int(ctrl.get("max", 10000))
    value = int(ctrl.get("value", ctrl.get("default", hi // 4)))
    for _ in range(TUNE_STEPS):
        if set_control(dev, name, value):
            return None
        for _ in range(SETTLE_FRAMES):
            grab()
        p = p99_gray(grab())
        log(f"[camera]   {name}={value}: brightest 1% at gray {p:.0f}")
        if p > TARGET_HIGH and value > lo:
            value = max(lo, int(value * 0.7))
        elif p < TARGET_LOW and value < hi:
            value = min(hi, int(value * 1.25) + 1)
        else:
            break
    log(f"[camera] picked {name}={value}; pass --exposure {value} to use it every time")
    return value
