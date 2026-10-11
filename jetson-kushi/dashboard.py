"""Jetson dashboard: one page to troubleshoot the rig, watch the camera, and run the sim.

    ./jetson-kushi/dashboard.sh          # from the repo root; open http://<jetson-ip>:8080

This is deployment glue, not a second copy of anything (see jetson-kushi/README.md, "the one
rule"). It imports robot/web_control.py's FastAPI app unchanged, moves its calibration page to
/calibrate, and adds:

  /                 dashboard page (jetson-kushi/dashboard_static/index.html)
  /stream.mjpg      live MJPEG from the overhead camera (?overlay=1 draws the robot's scene)
  /snapshot.jpg     one full-resolution frame
  /ui/              the monitor UI (ui/), so a phone or laptop needs only this one address
  /api/...          services (start/stop/logs), live feed taps, system health

Services are the existing entry points, run unmodified as subprocesses:
  sim         brain.server --mode robot --robot fake --grid sim   (fake arm, no hardware)
  perception  perception.publish --camera N                       (GridState on ws :8765)
  brain_live  brain.server --mode human --grid ws://localhost:8765
None of them moves the real arm. Arm scripts still run from /calibrate, one at a time.

The camera has one owner at a time. The dashboard opens it only while someone watches the
stream, closes it after STREAM_IDLE_S without a viewer, and gives it up at once while an arm
script (/calibrate) or the perception service runs. Perception needs the camera itself: its
grid lock was taken at the camera's native 640x480, which a resized 1080p stream cannot match.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

from buddy.routes import router as buddy_router, table_task as buddy_table_task  # noqa: E402
from robot import web_control  # noqa: E402
from robot.control_shared import ARM_SHEET_CALIB, PLACEMENT_CORR, file_age_str  # noqa: E402
from robot.perception import config as cam_config  # noqa: E402
from robot.perception.camera import Camera  # noqa: E402
from robot.perception.scene import annotate, build_scene  # noqa: E402
from shared.config import BRAIN_WS_PORT, GRID_WS_PORT, ROBOT_HTTP_PORT  # noqa: E402

app, state = web_control.app, web_control.state
PORT = web_control.PORT

STREAM_IDLE_S = 20.0      # close the camera this long after the last stream/snapshot frame
STREAM_MAX_W = 1920
LOG_LINES = 2000          # per service
STOP_GRACE_S = 6.0        # SIGINT, then this long before SIGTERM (scripts clean up on SIGINT)
PERCEPTION_CALIB = REPO_ROOT / "perception" / "calib"


# ---------------------------------------------------------------- camera

class CameraBusy(Exception):
    pass


def device_holders(dev: str) -> list[str]:
    """'name (pid N)' for every other process with dev open, from /proc (what fuser does)."""
    out = []
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = fd_dir.split("/")[2]
        if int(pid) == os.getpid():
            continue
        try:
            if any(os.readlink(os.path.join(fd_dir, fd)) == dev for fd in os.listdir(fd_dir)):
                out.append(f"{read(f'/proc/{pid}/comm')} (pid {pid})")
        except OSError:
            continue
    return out


class CameraHub:
    """Shares robot/web_control's camera (state.camera) between the stream and /preview.png,
    and hands it over to whatever else needs the device."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_use = 0.0
        self.last_error: str | None = None

    def busy_reason(self) -> str | None:
        if state.running_key:
            return f"arm script {state.running_key} (from /calibrate)"
        for svc in SERVICES.values():
            if svc.needs_camera and svc.running():
                return f"{svc.title} service"
        return None

    def release(self) -> None:
        with self.lock:
            if state.camera is not None:
                try:
                    state.camera.close()
                except Exception:
                    pass
                state.camera = None

    def frame(self, fresh: bool = False) -> np.ndarray:
        reason = self.busy_reason()
        if reason:
            self.release()
            raise CameraBusy(f"Camera in use by {reason}")
        self.last_use = time.time()
        if state.dev_mode:
            return ORIGINAL_READ_FRAME()
        with self.lock:
            if state.camera is None:
                try:
                    state.camera = Camera()
                    self.last_error = None
                except Exception as e:
                    holders = device_holders(f"/dev/video{cam_config.CAMERA_INDEX}")
                    self.last_error = str(e) if not holders else (
                        f"/dev/video{cam_config.CAMERA_INDEX} is held by " + ", ".join(holders))
                    raise RuntimeError(self.last_error) from e
            cam = state.camera
        frame = cam.read() if fresh else cam.latest()
        if frame is None:
            frame = cam.read()
        return frame

    def status(self) -> dict:
        reason = self.busy_reason()
        return {
            "index": cam_config.CAMERA_INDEX,
            "open": state.camera is not None,
            "busy": reason,
            "dev_mode": state.dev_mode,
            "error": self.last_error,
            "idle_s": round(time.time() - self.last_use, 1) if self.last_use else None,
        }

    def watch(self) -> None:
        """Background thread: give the camera up when idle or when something else needs it."""
        while True:
            time.sleep(0.25)
            if state.camera is None:
                continue
            if self.busy_reason() or time.time() - self.last_use > STREAM_IDLE_S:
                self.release()


hub = CameraHub()

# /preview.png and /status in web_control call read_frame() by name, so routing it through the
# hub keeps them from reopening the camera under a running arm script.
ORIGINAL_READ_FRAME = web_control.read_frame
web_control.read_frame = lambda: hub.frame(fresh=True)


def placeholder(text: str, w: int) -> np.ndarray:
    h = int(w * 9 / 16)
    img = np.full((h, w, 3), 24, np.uint8)
    scale = max(0.5, w / 1100)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (int(20 * scale), int((50 + 40 * i) * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def render_jpeg(overlay: bool, w: int, quality: int = 75) -> tuple[bytes, bool]:
    """One JPEG for the stream. Returns (jpeg, live) where live is False for a placeholder."""
    live = True
    try:
        frame = hub.frame()
        if overlay:
            try:
                scene, pose, detections = build_scene(frame, state.localizer)
                frame = annotate(frame, scene, pose, detections)
            except Exception as e:
                cv2.putText(frame, f"overlay: {e}"[:90], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 2, cv2.LINE_AA)
        if frame.shape[1] != w:
            frame = cv2.resize(frame, (w, int(frame.shape[0] * w / frame.shape[1])), interpolation=cv2.INTER_AREA)
    except CameraBusy as e:
        live, frame = False, placeholder(f"{e}.\nThe stream resumes when it finishes.", w)
    except Exception as e:
        live, frame = False, placeholder(f"Camera error:\n{str(e)[:80]}\nRetrying...", w)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes(), live


@app.get("/stream.mjpg")
async def stream(request: Request, overlay: int = 0, w: int = 960, fps: float = 12.0):
    w = max(320, min(STREAM_MAX_W, w))
    period = 1.0 / max(1.0, min(30.0, fps))

    async def frames():
        while not await request.is_disconnected():
            started = time.monotonic()
            jpg, live = await asyncio.to_thread(render_jpeg, bool(overlay), w)
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(max(0.0, (period if live else 1.0) - (time.monotonic() - started)))

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/snapshot.jpg")
def snapshot(overlay: int = 0):
    try:
        frame = hub.frame(fresh=True)
    except Exception as e:
        return Response(content=str(e), media_type="text/plain", status_code=503)
    jpg, _ = render_jpeg(bool(overlay), frame.shape[1], quality=92)
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Content-Disposition": f'inline; filename="snapshot_{time.strftime("%H%M%S")}.jpg"'})


# ---------------------------------------------------------------- services

def lesson_choices() -> list[str]:
    paths = sorted((REPO_ROOT / "lessons").glob("*.json")) + sorted((REPO_ROOT / "tests" / "fixtures").glob("*.json"))
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in paths]


def int_list(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text or "")][:6]


class Service:
    def __init__(self, name: str, title: str, desc: str, build_args, group: str | None = None,
                 needs_camera: bool = False, ports: tuple[int, ...] = (), options: list[dict] | None = None):
        self.name, self.title, self.desc = name, title, desc
        self.build_args, self.group, self.needs_camera = build_args, group, needs_camera
        self.ports, self.options = ports, options or []
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        self.exit_code: int | None = None
        self.cmd_str = ""
        self.log: collections.deque[tuple[int, str]] = collections.deque(maxlen=LOG_LINES)
        self.seq = 0
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def add_log(self, line: str) -> None:
        with self.lock:
            self.seq += 1
            self.log.append((self.seq, line))

    def start(self, opts: dict) -> None:
        if self.running():
            raise ValueError(f"{self.title} is already running")
        for other in SERVICES.values():
            if other is not self and self.group and other.group == self.group and other.running():
                raise ValueError(f"Stop {other.title} first: both use ws :{BRAIN_WS_PORT}")
        if not state.python_exe:
            raise ValueError("No Python with lerobot found, see docs/SETUP_NOTES.md")
        args = self.build_args(opts)
        if self.needs_camera:
            if state.running_key:
                raise ValueError(f"Arm script {state.running_key} is using the camera")
            hub.release()
            time.sleep(0.3)
        cmd = [state.python_exe, "-u", "-m", *args]
        self.cmd_str = "python -m " + " ".join(args)
        self.add_log(f"[dashboard] starting: {self.cmd_str}")
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.started_at, self.exit_code = time.time(), None
        proc = self.proc

        def reader():
            for line in proc.stdout:
                self.add_log(line.rstrip("\n"))
            code = proc.wait()
            self.exit_code = code
            self.add_log(f"[dashboard] exited with code {code}")

        threading.Thread(target=reader, daemon=True).start()

    def stop(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.add_log("[dashboard] stopping (SIGINT)")
        # SIGINT first, like web_control: every entry point cleans up on KeyboardInterrupt.
        for sig, wait in ((signal.SIGINT, STOP_GRACE_S), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                self.add_log(f"[dashboard] still running after {sig.name}")

    def info(self) -> dict:
        return {
            "name": self.name, "title": self.title, "desc": self.desc, "group": self.group,
            "needs_camera": self.needs_camera, "ports": list(self.ports), "options": self.options,
            "running": self.running(), "pid": self.proc.pid if self.running() else None,
            "uptime_s": round(time.time() - self.started_at) if self.running() else None,
            "exit_code": self.exit_code, "cmd": self.cmd_str, "log_seq": self.seq,
        }


def sim_args(opts: dict) -> list[str]:
    args = ["brain.server", "--mode", "robot", "--robot", "fake", "--grid", "sim", "--ui-host", "0.0.0.0"]
    lesson = opts.get("lesson") or ""
    if lesson:
        if lesson not in lesson_choices():
            raise ValueError(f"unknown lesson {lesson}")
        args += ["--lesson", lesson]
    for n in int_list(str(opts.get("wrong_on", ""))):
        args += ["--fake-wrong-on", str(n)]
    if opts.get("autostart", True):
        args.append("--autostart")
    return args


def perception_args(opts: dict) -> list[str]:
    exposure = str(opts.get("exposure") or "auto").strip()
    if not re.fullmatch(r"auto|\d{1,5}", exposure):
        raise ValueError("exposure must be 'auto' or a number")
    return ["perception.publish", "--camera", str(cam_config.CAMERA_INDEX), "--host", "0.0.0.0", "--exposure", exposure]


def brain_live_args(opts: dict) -> list[str]:
    args = ["brain.server", "--mode", "human", "--grid", f"ws://localhost:{GRID_WS_PORT}", "--ui-host", "0.0.0.0"]
    lesson = opts.get("lesson") or ""
    if lesson:
        if lesson not in lesson_choices():
            raise ValueError(f"unknown lesson {lesson}")
        args += ["--lesson", lesson]
    if opts.get("autostart"):
        args.append("--autostart")
    return args


SERVICES: dict[str, Service] = {s.name: s for s in [
    Service("sim", "Simulation",
            "Robot mode with a fake arm and an in-process fake board. No hardware, no camera. "
            "Watch it in the monitor UI.",
            sim_args, group="brain", ports=(BRAIN_WS_PORT,),
            options=[
                {"key": "lesson", "label": "Lesson", "type": "lesson"},
                {"key": "wrong_on", "label": "Fake wrong place on (e.g. 2 or 2,3,4)", "type": "text", "default": ""},
                {"key": "autostart", "label": "Autostart", "type": "bool", "default": True},
            ]),
    Service("perception", "Perception",
            "Overhead camera to GridState on ws :8765. Takes the camera, so the stream pauses.",
            perception_args, needs_camera=True, ports=(GRID_WS_PORT,),
            options=[{"key": "exposure", "label": "Exposure (auto or the value calibrate picked)", "type": "text", "default": "auto"}]),
    Service("brain_live", "Brain (live, human mode)",
            "Real perception feed into the brain, SessionState on ws :8766. Start Perception first.",
            brain_live_args, group="brain", ports=(BRAIN_WS_PORT,),
            options=[
                {"key": "lesson", "label": "Lesson", "type": "lesson"},
                {"key": "autostart", "label": "Autostart", "type": "bool", "default": False},
            ]),
]}


class StartBody(BaseModel):
    options: dict = {}


@app.get("/api/services")
def list_services():
    return {"services": [s.info() for s in SERVICES.values()], "lessons": lesson_choices()}


@app.post("/api/services/{name}/start")
async def start_service(name: str, body: StartBody):
    svc = SERVICES.get(name)
    if svc is None:
        return JSONResponse({"ok": False, "error": "unknown service"}, status_code=404)
    try:
        await asyncio.to_thread(svc.start, body.options)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "service": svc.info()}


@app.post("/api/services/{name}/stop")
async def stop_service(name: str):
    svc = SERVICES.get(name)
    if svc is None:
        return JSONResponse({"ok": False, "error": "unknown service"}, status_code=404)
    await asyncio.to_thread(svc.stop)
    return {"ok": True, "service": svc.info()}


@app.get("/api/services/{name}/log")
def service_log(name: str, after: int = 0):
    svc = SERVICES.get(name)
    if svc is None:
        return JSONResponse({"error": "unknown service"}, status_code=404)
    with svc.lock:
        lines = [{"seq": s, "text": t} for s, t in svc.log if s > after]
    return {"lines": lines, "seq": svc.seq}


# ---------------------------------------------------------------- feed taps

class FeedTap:
    """Listens on a ws feed like any other client and keeps the latest message and its rate."""

    def __init__(self, name: str, url: str):
        self.name, self.url = name, url
        self.connected = False
        self.last: dict | None = None
        self.last_t = 0.0
        self.count = 0
        self.times: collections.deque[float] = collections.deque(maxlen=50)
        self.ws = None
        self.error: str | None = None

    async def run(self) -> None:
        while True:
            try:
                async with connect(self.url, open_timeout=2, max_size=4 * 1024 * 1024) as ws:
                    self.ws, self.connected, self.error = ws, True, None
                    async for msg in ws:
                        now = time.time()
                        self.count += 1
                        self.times.append(now)
                        self.last_t = now
                        try:
                            self.last = json.loads(msg)
                        except ValueError:
                            self.last = {"raw": str(msg)[:500]}
            except Exception as e:
                self.error = type(e).__name__ if not str(e) else str(e)[:120]
            self.ws, self.connected = None, False
            await asyncio.sleep(1.0)

    def info(self) -> dict:
        now = time.time()
        recent = [t for t in self.times if now - t < 3.0]
        return {
            "url": self.url, "connected": self.connected, "count": self.count,
            "hz": round(len(recent) / 3.0, 1),
            "age_s": round(now - self.last_t, 1) if self.last_t else None,
            "last": self.last, "error": None if self.connected else self.error,
        }


TAPS = {
    "grid": FeedTap("grid", f"ws://127.0.0.1:{GRID_WS_PORT}"),
    "session": FeedTap("session", f"ws://127.0.0.1:{BRAIN_WS_PORT}"),
}


@app.get("/api/feeds")
def feeds():
    return {k: t.info() for k, t in TAPS.items()}


class BrainCmd(BaseModel):
    cmd: str


@app.post("/api/brain/cmd")
async def brain_cmd(body: BrainCmd):
    if body.cmd not in ("start", "reset"):
        return JSONResponse({"ok": False, "error": "cmd must be start or reset"}, status_code=400)
    ws = TAPS["session"].ws
    if ws is None:
        return JSONResponse({"ok": False, "error": f"brain not connected on ws :{BRAIN_WS_PORT}"}, status_code=409)
    await ws.send(json.dumps({"cmd": body.cmd}))
    return {"ok": True}


# ---------------------------------------------------------------- health

def run_cmd(*cmd: str, timeout: float = 3.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)).stdout.strip()
    except Exception:
        return ""


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def meminfo() -> dict:
    out = {}
    for line in read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            out[k] = int(v.split()[0]) // 1024  # MiB
    return out


def temps() -> dict:
    out = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = read(f"{zone}/temp")
        if t.lstrip("-").isdigit():
            out[read(f"{zone}/type")] = round(int(t) / 1000, 1)
    return out


def health() -> dict:
    mem, tz = meminfo(), temps()
    du = shutil.disk_usage(str(REPO_ROOT))
    robot_port = os.environ.get("ROBOT_PORT")
    serial = [{"path": p, "rw": os.access(p, os.R_OK | os.W_OK)} for p in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))]
    cameras = [{"path": p, "name": read(f"/sys/class/video4linux/{Path(p).name}/name")} for p in sorted(glob.glob("/dev/video*"))]
    colors_json = PERCEPTION_CALIB / "colors.json"
    lock_json = PERCEPTION_CALIB / "grid_lock.json"
    try:
        lock_created = json.loads(lock_json.read_text()).get("created")
    except Exception:
        lock_created = None
    ports = {str(p): port_open(p) for p in (GRID_WS_PORT, BRAIN_WS_PORT, ROBOT_HTTP_PORT)}

    warnings = []
    hottest = max(tz.values(), default=0)
    if hottest >= 85:
        warnings.append(f"Hot: {hottest} C. Check the fan and airflow.")
    if mem.get("MemAvailable", 1e9) < 600:
        warnings.append(f"Low memory: {mem['MemAvailable']} MiB available.")
    if du.free < 2 * 1024**3:
        warnings.append("Disk almost full.")
    if not cameras:
        warnings.append("No camera found (/dev/video*). Check the USB cable.")
    elif not os.path.exists(f"/dev/video{cam_config.CAMERA_INDEX}"):
        warnings.append(f"CAMERA_INDEX={cam_config.CAMERA_INDEX} but /dev/video{cam_config.CAMERA_INDEX} does not exist. Set it in jetson-kushi/.env.")
    if hub.last_error and state.camera is None:
        warnings.append(f"Camera: {hub.last_error}")
    if "CAMERA_INDEX" not in os.environ:
        warnings.append("CAMERA_INDEX is not set, so the Windows default (1) applies. Start with jetson-kushi/dashboard.sh.")
    if not robot_port:
        warnings.append("ROBOT_PORT is not set. Start with jetson-kushi/dashboard.sh.")
    elif not os.path.exists(robot_port):
        warnings.append(f"ROBOT_PORT={robot_port} does not exist. Is the arm's USB plugged in?")
    elif not os.access(robot_port, os.R_OK | os.W_OK):
        warnings.append(f"No permission on {robot_port}. Add yourself to dialout and log in again.")
    if not colors_json.exists():
        warnings.append("Perception colors not calibrated (perception/calib/colors.json missing). Run python -m perception.calibrate.")
    if not lock_json.exists():
        warnings.append("No perception grid lock. Run python -m perception.calibrate.")

    return {
        "host": {
            "hostname": socket.gethostname(),
            "ips": run_cmd("hostname", "-I").split(),
            "uptime_s": int(float(read("/proc/uptime").split()[0] or 0)) if read("/proc/uptime") else None,
            "load": os.getloadavg(),
            "cpus": os.cpu_count(),
            "l4t": read("/etc/nv_tegra_release").splitlines()[0][2:60] if read("/etc/nv_tegra_release") else None,
        },
        "temps_c": tz,
        "mem_mib": mem,
        "disk_gb": {"free": round(du.free / 1024**3, 1), "total": round(du.total / 1024**3, 1)},
        "cameras": cameras,
        "camera": hub.status(),
        "serial": serial,
        "robot_port": robot_port,
        "robot_port_real": os.path.realpath(robot_port) if robot_port else None,
        "ports": ports,
        "git": {
            "branch": run_cmd("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run_cmd("git", "log", "-1", "--format=%h %s (%cr)"),
            "dirty": len([l for l in run_cmd("git", "status", "--porcelain").splitlines() if l.strip()]),
        },
        "python": {"exe": state.python_exe, "source": state.python_source},
        "calibration": {
            "perception_colors": file_age_str(colors_json),
            "perception_grid_lock": lock_created or "missing",
            "arm_sheet": file_age_str(ARM_SHEET_CALIB),
            "placement_correction": file_age_str(PLACEMENT_CORR),
        },
        "arm_script": state.running_key,
        "warnings": warnings,
    }


@app.get("/api/health")
async def api_health():
    return await asyncio.to_thread(health)


# ---------------------------------------------------------------- pages and lifecycle

# web_control serves its calibration page at "/"; the dashboard takes "/" and the
# calibration page moves to /calibrate, unchanged.
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


@app.get("/")
def dashboard_page():
    return FileResponse(HERE / "dashboard_static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/calibrate")
def calibrate_page():
    return FileResponse(REPO_ROOT / "robot" / "web_control_static" / "index.html")


@app.get("/ui")
def ui_redirect():
    return RedirectResponse("/ui/")


# Apprentice Buddy, the children's game app (buddy/): /buddy and /api/buddy/*.
app.include_router(buddy_router)


app.mount("/ui", StaticFiles(directory=str(REPO_ROOT / "ui"), html=True), name="ui")
app.mount("/perception/calib", StaticFiles(directory=str(PERCEPTION_CALIB), check_dir=False), name="calib")
app.mount("/videos", StaticFiles(directory=str(REPO_ROOT / "videos"), check_dir=False), name="videos")


@contextlib.asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=hub.watch, daemon=True).start()
    tasks = [asyncio.create_task(tap.run()) for tap in TAPS.values()]
    tasks.append(asyncio.create_task(buddy_table_task()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for svc in SERVICES.values():
            await asyncio.to_thread(svc.stop)
        hub.release()


app.router.lifespan_context = lifespan


def main() -> None:
    import uvicorn

    print(f"[dashboard] http://0.0.0.0:{PORT}  camera index {cam_config.CAMERA_INDEX}, "
          f"python {state.python_exe or 'NOT FOUND'}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
