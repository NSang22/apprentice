"""Phone-facing web version of calibration_gui.py, for a Jetson reachable only by its
own WiFi AP - no laptop, no SSH terminal needed for calibration.

    python -m robot.web_control            # serves on :8080, open http://<jetson-ip>:8080

Same split as the Tkinter GUI: camera/sheet check, calibration-file status, and rig
constant editing happen inline here (safe, read-only, no arm motion). Steps that touch
the arm (touch_calibrate, auto_calibrate, pick_place_test, first_move, joint_tour,
recalibrate_gripper) run the real, unmodified script as a subprocess - same scripts, same
safety-critical confirmation prompts ("hold the arm, press ENTER") - just streamed over a
WebSocket as a terminal instead of a detached console window, since a phone can't open
one of those. No arm-motion or calibration-fitting logic lives in this file.

Safety properties worth knowing before changing this file:
  - Only ONE arm-touching script can run at a time (RUNNING lock below). Two scripts
    fighting over the same servo bus is a real hardware risk, not just a UX nicety.
  - On disconnect we send SIGINT, not a hard kill. Every script under robot/scripts/
    catches KeyboardInterrupt specifically to disable torque and disconnect cleanly in
    its `finally` block - a hard kill (SIGTERM/kill -9) skips that and can leave torque
    enabled on a disconnected bus.
"""

from __future__ import annotations

import asyncio
import json
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

ROBOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from robot.control_shared import (  # noqa: E402
    ARM_SHEET_CALIB,
    BUTTONS_BY_KEY,
    DEV_SAMPLE_IMAGES,
    OPEN_QUESTIONS,
    PLACEMENT_CORR,
    PLACEMENT_RMS_OK_MM,
    STEPS,
    file_age_str,
    find_lerobot_python,
    start_module_process,
    write_config_value,
)
from robot.perception import config as cam_config  # noqa: E402
from robot.perception.camera import Camera  # noqa: E402
from robot.perception.scene import annotate, build_scene  # noqa: E402
from robot.perception.sheet import SheetLocalizer  # noqa: E402

PORT = 8080
PREVIEW_W, PREVIEW_H = 640, 360


class State:
    def __init__(self):
        self.camera: Camera | None = None
        self.localizer = SheetLocalizer()
        self.dev_mode = False
        self._dev_img_idx = 0
        self.python_exe, self.python_source = find_lerobot_python()
        self.running_key: str | None = None  # the one arm-touching step allowed to run at a time


state = State()
app = FastAPI()


def read_frame() -> "cv2.typing.MatLike":
    if state.dev_mode:
        path = DEV_SAMPLE_IMAGES[state._dev_img_idx % len(DEV_SAMPLE_IMAGES)]
        state._dev_img_idx += 1
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"could not read sample image {path}")
        return frame
    if state.camera is None:
        state.camera = Camera()
    return state.camera.read()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(ROBOT_DIR / "web_control_static" / "index.html")


@app.get("/preview.png")
def preview():
    try:
        frame = read_frame()
        scene, pose, detections = build_scene(frame, state.localizer)
        annotated = annotate(frame, scene, pose, detections)
        small = cv2.resize(annotated, (PREVIEW_W, PREVIEW_H))
        ok, buf = cv2.imencode(".png", small)
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        return Response(content=str(e), media_type="text/plain", status_code=500)


@app.get("/status")
def status():
    try:
        frame = read_frame()
        scene, pose, detections = build_scene(frame, state.localizer)
    except Exception as e:
        scene = None
        sheet_error = str(e)
    else:
        sheet_error = None

    def calib_file_status(path, ok_fn=None):
        if not path.exists():
            return {"exists": False, "age": "missing"}
        data = json.loads(path.read_text())
        return {"exists": True, "age": file_age_str(path), **data}

    arm_calib = calib_file_status(ARM_SHEET_CALIB)
    placement = calib_file_status(PLACEMENT_CORR)
    placement_ok = (
        placement.get("exists") and placement.get("rms_after_fit_mm", 999) <= PLACEMENT_RMS_OK_MM
    )

    return {
        "dev_mode": state.dev_mode,
        "python_exe": state.python_exe,
        "python_source": state.python_source,
        "running_key": state.running_key,
        "open_questions": OPEN_QUESTIONS,
        "rig": {
            "camera_index": cam_config.CAMERA_INDEX,
            "frame_size": f"{cam_config.FRAME_W}x{cam_config.FRAME_H}",
            "objects": cam_config.OBJECTS,
            "camera_height_mm": cam_config.CAMERA_HEIGHT_MM,
            "cube_size_mm": cam_config.CUBE_SIZE_MM,
        },
        "sheet_error": sheet_error,
        "scene": scene,
        "arm_calib": arm_calib,
        "placement": placement,
        "placement_ok": placement_ok,
    }


class ConfigUpdate(BaseModel):
    name: str
    value: float


@app.post("/config")
def update_config(body: ConfigUpdate):
    if body.name not in ("CAMERA_HEIGHT_MM", "CUBE_SIZE_MM"):
        return Response(content="unknown constant", status_code=400)
    try:
        write_config_value(body.name, body.value, f"measured via web control, {time.strftime('%Y-%m-%d')}")
        return {"ok": True}
    except Exception as e:
        return Response(content=str(e), status_code=500)


class DevModeUpdate(BaseModel):
    enabled: bool


@app.post("/dev-mode")
def set_dev_mode(body: DevModeUpdate):
    state.dev_mode = body.enabled
    if state.camera is not None:
        try:
            state.camera.close()
        except Exception:
            pass
        state.camera = None
    return {"dev_mode": state.dev_mode}


@app.get("/steps")
def steps():
    return [
        {
            "id": s.id, "group": s.group, "title": s.title, "desc": s.desc,
            "buttons": [{"key": b.key, "label": b.label} for b in s.buttons],
        }
        for s in STEPS
    ]


@app.websocket("/terminal/{key}")
async def terminal(websocket: WebSocket, key: str):
    await websocket.accept()
    button = BUTTONS_BY_KEY.get(key)
    if button is None:
        await websocket.send_text(f"[error] unknown step: {key}")
        await websocket.close()
        return

    cmd_str = " ".join([state.python_exe or "python", "-m", button.module, *button.args])

    if state.dev_mode:
        await websocket.send_text(f"[DEV MODE] would run: {cmd_str}")
        await websocket.send_text("[closed]")
        await websocket.close()
        return

    if state.python_exe is None:
        await websocket.send_text("[error] no Python with lerobot found - see docs/SETUP_NOTES.md")
        await websocket.close()
        return

    if state.running_key is not None:
        await websocket.send_text(
            f"[error] {state.running_key} is already running. Only one arm-touching step "
            "runs at a time (two scripts on the same servo bus is a real hardware risk)."
        )
        await websocket.close()
        return

    state.running_key = key
    proc = start_module_process(state.python_exe, button.module, *button.args)
    out_q: queue.Queue[str | None] = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                out_q.put(line.rstrip("\n"))
        finally:
            out_q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    await websocket.send_text(f"[started] {cmd_str}")

    recv_task = asyncio.ensure_future(websocket.receive_text())
    disconnected = False
    try:
        while True:
            try:
                while True:
                    line = out_q.get_nowait()
                    if line is None:
                        code = proc.wait()
                        await websocket.send_text(f"[closed] exit code {code}")
                        return
                    await websocket.send_text(line)
            except queue.Empty:
                pass

            if recv_task.done():
                try:
                    msg = recv_task.result()
                except WebSocketDisconnect:
                    disconnected = True
                    break
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.write(msg + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                recv_task = asyncio.ensure_future(websocket.receive_text())

            await asyncio.sleep(0.05)
    finally:
        recv_task.cancel()
        if proc.poll() is None:
            # SIGINT (not terminate/kill): every script under robot/scripts/ catches
            # KeyboardInterrupt specifically to disable torque and disconnect cleanly.
            # A hard kill would skip that and could leave torque enabled.
            try:
                proc.send_signal(signal.SIGINT if hasattr(signal, "SIGINT") else signal.SIGTERM)
            except Exception:
                pass
        state.running_key = None
        if disconnected:
            try:
                await websocket.close()
            except Exception:
                pass


def main():
    import uvicorn

    static_dir = ROBOT_DIR / "web_control_static"
    if not (static_dir / "index.html").exists():
        raise SystemExit(f"Missing {static_dir / 'index.html'}")
    print(f"Serving on http://0.0.0.0:{PORT}  (python: {state.python_exe or 'NOT FOUND - see /status'})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
