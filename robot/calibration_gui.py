"""One dashboard for every calibration step, so you don't have to remember module
paths, flags, or what order to run things in.

Run from the repo root:

    python -m robot.calibration_gui

You do NOT need to launch this from inside the lerobot venv specifically: at startup
it checks whether the running interpreter can import `lerobot`, and if not, looks for
the conventional venv at ~/lerobot/.venv (docs/SETUP_NOTES.md) and uses that Python
for the "Run" buttons instead. The resolved interpreter is shown at the top of the
window, so it's never a silent guess.

What it does inline (safe, read-only, no arm motion):
  - Grabs one camera frame, detects the AprilTag sheet and cubes, shows the result.
  - Shows the age and fit quality of the two calibration files.
  - Lets you edit the measured rig constants in perception/config.py.

What it launches instead of reimplementing: every step that moves the arm or needs
you to physically touch it (touch_calibrate, auto_calibrate, pick_place_test,
first_move, joint_tour, recalibrate_gripper) opens the real, already-tested script
in its own console window, exactly as if you'd typed the command yourself. This
GUI does not contain any arm-motion or calibration-fitting code of its own - that
logic (gravity-sag correction, push detection, blocked-path backoff, the IK and
affine fits) is tuned and safety-critical; duplicating it here would only add risk.

Order of operations (robot/calibration/README.md), reflected top to bottom below:
  1. Camera moved? Re-measure lens height.
  2. Arm or sheet moved? Run touch calibration (quick, or full if the arm was remounted).
  3. Always run auto-calibrate after a touch calibration.
  4. Verify with a real pick and place.
"""

from __future__ import annotations

import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2

ROBOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from robot.control_shared import (  # noqa: E402
    ARM_SHEET_CALIB,
    DEV_SAMPLE_IMAGES,
    OPEN_QUESTIONS,
    PLACEMENT_CORR,
    PLACEMENT_RMS_OK_MM,
    STEPS,
    file_age_str,
    find_lerobot_python,
    run_module_detached,
    write_config_value,
)
from robot.perception import config as cam_config  # noqa: E402
from robot.perception.camera import Camera  # noqa: E402
from robot.perception.scene import annotate, build_scene  # noqa: E402
from robot.perception.sheet import SheetLocalizer  # noqa: E402

PREVIEW_W, PREVIEW_H = 560, 315
AUTO_REFRESH_MS = 800


class CalibrationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Apprentice calibration")
        self.geometry("980x760")
        self.camera: Camera | None = None
        self.localizer = SheetLocalizer()
        self.auto_refresh = tk.BooleanVar(value=False)
        self.dev_mode = tk.BooleanVar(value=False)
        self._dev_img_idx = 0
        self.python_exe, self.python_source = find_lerobot_python()
        self._build_ui()
        self._refresh_calib_status()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        warn = tk.Label(
            self, fg="#a05a00",
            text="Open questions vs README (robot/README.md): " + " | ".join(OPEN_QUESTIONS),
            wraplength=940, justify="left", font=("Segoe UI", 8),
        )
        warn.pack(fill="x", **pad)

        if self.python_exe is not None:
            interp_text = f"Run buttons use: {self.python_exe}  ({self.python_source})"
            interp_fg = "#2a7a2a" if self.python_source == "current interpreter" else "#a05a00"
        else:
            interp_text = (
                f"Could not find a Python with lerobot installed ({self.python_source}). "
                "Run buttons will fail until this is fixed - see docs/SETUP_NOTES.md."
            )
            interp_fg = "red"
        tk.Label(self, fg=interp_fg, text=interp_text, wraplength=940, justify="left",
                 font=("Segoe UI", 8)).pack(fill="x", padx=8)

        dev_row = tk.Frame(self, bg="#fff3cd")
        dev_row.pack(fill="x", padx=8, pady=(4, 0))
        tk.Checkbutton(
            dev_row, text="DEV MODE: use sample images instead of the camera, "
                          "and show (don't launch) what Run buttons would do",
            variable=self.dev_mode, command=self._on_dev_mode_toggle, bg="#fff3cd",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=4, pady=2)

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        cam_frame = ttk.LabelFrame(top, text="Camera / sheet check (safe, no arm motion)")
        cam_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        btn_row = ttk.Frame(cam_frame)
        btn_row.pack(fill="x", padx=6, pady=4)
        ttk.Button(btn_row, text="Check now", command=self._check_once).pack(side="left")
        ttk.Checkbutton(
            btn_row, text="Auto-refresh", variable=self.auto_refresh, command=self._toggle_auto_refresh
        ).pack(side="left", padx=10)

        self.preview_label = tk.Label(cam_frame, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        self.preview_label.pack(padx=6, pady=4)

        self.sheet_status = tk.Label(cam_frame, text="Not checked yet.", justify="left", anchor="w")
        self.sheet_status.pack(fill="x", padx=6, pady=(0, 6))

        rig_frame = ttk.LabelFrame(top, text="Rig constants (perception/config.py)")
        rig_frame.pack(side="left", fill="both", padx=(6, 0))

        ttk.Label(rig_frame, text=f"Camera index: {cam_config.CAMERA_INDEX}").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Label(rig_frame, text=f"Frame size: {cam_config.FRAME_W}x{cam_config.FRAME_H}").pack(anchor="w", padx=6)
        ttk.Label(rig_frame, text=f"Objects: {', '.join(cam_config.OBJECTS)}").pack(anchor="w", padx=6)

        self.height_var = tk.StringVar(value=str(cam_config.CAMERA_HEIGHT_MM))
        self._labeled_entry(rig_frame, "Camera height (mm):", self.height_var,
                             lambda: self._save_constant("CAMERA_HEIGHT_MM", self.height_var))

        self.cube_var = tk.StringVar(value=str(cam_config.CUBE_SIZE_MM))
        self._labeled_entry(rig_frame, "Cube size (mm):", self.cube_var,
                             lambda: self._save_constant("CUBE_SIZE_MM", self.cube_var))

        self.config_msg = tk.Label(rig_frame, text="", fg="#2a7a2a")
        self.config_msg.pack(anchor="w", padx=6, pady=(0, 6))

        calib_frame = ttk.LabelFrame(self, text="Calibration file status")
        calib_frame.pack(fill="x", **pad)
        self.arm_calib_label = tk.Label(calib_frame, justify="left", anchor="w")
        self.arm_calib_label.pack(fill="x", padx=6, pady=2)
        self.corr_label = tk.Label(calib_frame, justify="left", anchor="w")
        self.corr_label.pack(fill="x", padx=6, pady=2)
        ttk.Button(calib_frame, text="Refresh status", command=self._refresh_calib_status).pack(
            anchor="e", padx=6, pady=4
        )

        steps = ttk.LabelFrame(self, text="Order of operations after moving anything")
        steps.pack(fill="both", expand=True, **pad)
        extras = ttk.LabelFrame(self, text="Extra diagnostics (not part of the required order)")
        extras.pack(fill="x", **pad)

        for step in STEPS:
            self._step(steps if step.group == "order" else extras, step)

        self.status_bar = tk.Label(self, text="Ready.", anchor="w", bd=1, relief="sunken")
        self.status_bar.pack(fill="x", side="bottom")

    def _labeled_entry(self, parent, label, var, on_save):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        ttk.Button(row, text="Save", command=on_save).pack(side="left", padx=4)

    def _step(self, parent, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=4)
        head = ttk.Frame(row)
        head.pack(fill="x")
        ttk.Label(head, text=f"{step.id}.", width=3, font=("Segoe UI", 9, "bold")).pack(side="left")
        text_col = ttk.Frame(head)
        text_col.pack(side="left", fill="x", expand=True)
        ttk.Label(text_col, text=step.title, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Label(text_col, text=step.desc, wraplength=700, justify="left").pack(anchor="w")
        btn_col = ttk.Frame(row)
        btn_col.pack(side="right")
        for button in step.buttons:
            ttk.Button(btn_col, text=button.label, command=self._launch(button)).pack(side="left", padx=2)
        ttk.Separator(parent).pack(fill="x", padx=6, pady=2)

    def _launch(self, button):
        def wrapped():
            try:
                self._run(button.module, *button.args)
                self.status_bar.config(text=f"Launched: {button.label} (new console window)")
            except Exception as e:
                self.status_bar.config(text=f"Could not launch {button.label}: {e}")
        return wrapped

    def _run(self, module: str, *args: str) -> None:
        if self.dev_mode.get():
            cmd_str = " ".join([self.python_exe or "python", "-m", module, *args])
            self.status_bar.config(text=f"[DEV MODE] would run: {cmd_str}")
            return
        if self.python_exe is None:
            raise RuntimeError("no Python with lerobot found - see docs/SETUP_NOTES.md")
        run_module_detached(self.python_exe, module, *args)

    # -- dev mode --------------------------------------------------------
    def _on_dev_mode_toggle(self):
        self.auto_refresh.set(False)
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        if self.dev_mode.get():
            self.sheet_status.config(text="Dev mode on: click 'Check now' to load a sample image.", fg="black")
            self.status_bar.config(text="Dev mode on. No camera or robot needed.")
        else:
            self.sheet_status.config(text="Dev mode off. Not checked yet.", fg="black")
            self.status_bar.config(text="Dev mode off.")

    # -- camera check --------------------------------------------------------
    def _ensure_camera(self):
        if self.camera is None:
            self.status_bar.config(text="Opening camera...")
            self.update_idletasks()
            self.camera = Camera()

    def _check_once(self):
        try:
            if not self.dev_mode.get():
                self._ensure_camera()
            self._refresh_preview()
            self.status_bar.config(text="Checked.")
        except Exception as e:
            self.sheet_status.config(text=f"Camera error: {e}", fg="red")
            self.status_bar.config(text="Camera check failed.")

    def _toggle_auto_refresh(self):
        if self.auto_refresh.get():
            if not self.dev_mode.get():
                try:
                    self._ensure_camera()
                except Exception as e:
                    self.sheet_status.config(text=f"Camera error: {e}", fg="red")
                    self.auto_refresh.set(False)
                    return
            self._auto_refresh_tick()

    def _auto_refresh_tick(self):
        if not self.auto_refresh.get():
            return
        try:
            self._refresh_preview()
        except Exception as e:
            self.sheet_status.config(text=f"Camera error: {e}", fg="red")
        self.after(AUTO_REFRESH_MS, self._auto_refresh_tick)

    def _read_frame(self):
        if self.dev_mode.get():
            path = DEV_SAMPLE_IMAGES[self._dev_img_idx % len(DEV_SAMPLE_IMAGES)]
            self._dev_img_idx += 1
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"could not read sample image {path}")
            return frame
        return self.camera.read()

    def _refresh_preview(self):
        frame = self._read_frame()
        scene, pose, detections = build_scene(frame, self.localizer)
        annotated = annotate(frame, scene, pose, detections)
        small = cv2.resize(annotated, (PREVIEW_W, PREVIEW_H))
        ok, buf = cv2.imencode(".png", small)
        photo = tk.PhotoImage(data=buf.tobytes())
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # keep a reference, tkinter drops unreferenced images

        prefix = "[DEV MODE - sample image] " if self.dev_mode.get() else ""
        if scene is None:
            self.sheet_status.config(text=f"{prefix}SHEET NOT FOUND (need 2+ tags visible)", fg="red")
            return
        lines = [f"{prefix}Sheet OK: tags {scene['sheet']['tags']}, fit error {scene['sheet']['fit_error_mm']:.2f} mm"]
        if scene["objects"]:
            for o in scene["objects"]:
                lines.append(f"  {o['name']}: ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm, "
                             f"yaw {o['yaw_deg']:.0f}, score {o['score']:.2f}")
        else:
            lines.append("  no cubes detected")
        self.sheet_status.config(text="\n".join(lines), fg="#2a7a2a")

    # -- config editing --------------------------------------------------------
    def _save_constant(self, name, var):
        try:
            value = float(var.get())
            write_config_value(name, value, f"measured via calibration GUI, {time.strftime('%Y-%m-%d')}")
            self.config_msg.config(text=f"Saved {name} = {value}. Restart scripts to pick it up.")
        except Exception as e:
            self.config_msg.config(text=f"Could not save: {e}", fg="red")

    # -- calibration file status --------------------------------------------------------
    def _refresh_calib_status(self):
        if ARM_SHEET_CALIB.exists():
            c = json.loads(ARM_SHEET_CALIB.read_text())
            self.arm_calib_label.config(
                text=f"arm_sheet_calibration.json: fit RMS {c['fit_rms_mm']} mm, {c['n_points']} points, "
                     f"mode {c['mode']}  ({file_age_str(ARM_SHEET_CALIB)})",
                fg="#2a7a2a",
            )
        else:
            self.arm_calib_label.config(text="arm_sheet_calibration.json: MISSING - run full touch calibration first.", fg="red")

        if PLACEMENT_CORR.exists():
            c = json.loads(PLACEMENT_CORR.read_text())
            rms = c["rms_after_fit_mm"]
            color = "#2a7a2a" if rms <= PLACEMENT_RMS_OK_MM else "#a05a00"
            flag = "" if rms <= PLACEMENT_RMS_OK_MM else "  (provisional, consider re-running with more placements)"
            self.corr_label.config(
                text=f"placement_correction.json: {rms} mm RMS from {c['n_placements']} placements"
                     f"  ({file_age_str(PLACEMENT_CORR)}){flag}",
                fg=color,
            )
        else:
            self.corr_label.config(text="placement_correction.json: MISSING - run auto-calibrate.", fg="red")

    def _on_close(self):
        self.auto_refresh.set(False)
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.destroy()


def main():
    CalibrationGUI().mainloop()


if __name__ == "__main__":
    main()
