#!/usr/bin/env bash
# Demo startup on the Jetson: perception.publish (ws :8765), then robot.api (HTTP :8770).
#
#   scripts/jetson_up.sh                 # BLOCK_MM=47 by default; --debug window when DISPLAY is set
#   BLOCK_MM=48 scripts/jetson_up.sh --no-robot
#   CAMERA=2 scripts/jetson_up.sh --no-debug     # skip the camera search, use /dev/video2
#
# Env: BLOCK_MM (block edge, default 47), CAMERA (index, default: first /dev/videoN that gives a
# frame), EXPOSURE (passed to publish --exposure, default publish's own). jetson-kushi/.env
# (ROBOT_PORT) is sourced when present. Logs go to logs/publish.log and logs/robot.log.
# Services run detached (setsid nohup), so they survive closing this ssh session.

set -u

ROBOT=1
DEBUG=1
for arg in "$@"; do
    case "$arg" in
        --no-robot) ROBOT=0 ;;
        --no-debug) DEBUG=0 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg (use --no-robot, --no-debug)"; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
mkdir -p logs

GREEN=$'\e[32m'; RED=$'\e[31m'; YELLOW=$'\e[33m'; RESET=$'\e[0m'
say() { echo "[jetson_up] $*"; }

VENV="$HOME/lerobot/.venv/bin/activate"
if [ -f "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV"
    say "venv: $VENV"
else
    say "${YELLOW}no lerobot venv at $VENV, using $(command -v python3)${RESET}"
fi
if [ -f jetson-kushi/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source jetson-kushi/.env
    set +a
    say "env: jetson-kushi/.env (ROBOT_PORT=${ROBOT_PORT:-unset})"
fi
PY="$(command -v python || command -v python3)"

# ---------------------------------------------------------------- stale processes
say "stopping stale perception.publish / robot.api"
pkill -f -- "-m perception.publish" 2>/dev/null
pkill -f -- "-m robot.api" 2>/dev/null
holders=""
if command -v fuser >/dev/null 2>&1; then
    holders="$(fuser /dev/video* 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u)"
elif command -v lsof >/dev/null 2>&1; then
    holders="$(lsof -t /dev/video* 2>/dev/null | sort -u)"
fi
for pid in $holders; do
    say "killing pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null)) holding a /dev/video device"
    kill "$pid" 2>/dev/null
done
for _ in $(seq 1 10); do
    pgrep -f -- "-m perception.publish|-m robot.api" >/dev/null || break
    sleep 0.5
done
pkill -9 -f -- "-m perception.publish" 2>/dev/null
pkill -9 -f -- "-m robot.api" 2>/dev/null
for pid in $holders; do kill -9 "$pid" 2>/dev/null; done
sleep 0.5

# ---------------------------------------------------------------- camera
probe_camera() {  # exit 0 when index $1 opens on V4L2 and returns a frame
    timeout 10 "$PY" - "$1" <<'EOF' >/dev/null 2>&1
import sys, cv2
cap = cv2.VideoCapture(int(sys.argv[1]), cv2.CAP_V4L2)
ok = cap.isOpened() and any(cap.read()[0] for _ in range(5))
cap.release()
sys.exit(0 if ok else 1)
EOF
}

if [ -n "${CAMERA:-}" ]; then
    CAM="$CAMERA"
    say "camera: /dev/video$CAM (from CAMERA)"
else
    CAM=""
    for dev in $(ls /dev/video* 2>/dev/null | sort -V); do
        n="${dev#/dev/video}"
        if probe_camera "$n"; then CAM="$n"; break; fi
        say "  $dev: no frame"
    done
    if [ -z "$CAM" ]; then
        echo "${RED}FAIL  camera: no /dev/videoN returned a frame (is the webcam plugged in? v4l2-ctl --list-devices)${RESET}"
        exit 1
    fi
    say "camera: /dev/video$CAM"
fi

# ---------------------------------------------------------------- perception.publish
BLOCK_MM="${BLOCK_MM:-47}"
PUB_ARGS=(--camera "$CAM" --block-mm "$BLOCK_MM")
[ -n "${EXPOSURE:-}" ] && PUB_ARGS+=(--exposure "$EXPOSURE")
if [ "$DEBUG" = 1 ] && [ -n "${DISPLAY:-}" ]; then PUB_ARGS+=(--debug); fi
say "starting: python -m perception.publish ${PUB_ARGS[*]}  (logs/publish.log)"
setsid nohup "$PY" -m perception.publish "${PUB_ARGS[@]}" >logs/publish.log 2>&1 </dev/null &
PUB_PID=$!

wait_gridstate() {  # exit 0 once ws :8765 sends one valid GridState
    timeout 5 "$PY" - <<'EOF' >/dev/null 2>&1
import asyncio, json
from websockets.asyncio.client import connect
from shared.schemas import GridState
async def main():
    async with connect("ws://localhost:8765", open_timeout=2) as ws:
        GridState.from_dict(json.loads(await asyncio.wait_for(ws.recv(), 3)))
asyncio.run(main())
EOF
}

PUB_OK=0
PUB_MSG=""
for _ in $(seq 1 60); do  # up to ~90 s: camera lock and model load are slow on first start
    if ! kill -0 "$PUB_PID" 2>/dev/null; then PUB_MSG="exited, see logs/publish.log"; break; fi
    if wait_gridstate; then PUB_OK=1; break; fi
    sleep 1
done
[ "$PUB_OK" = 0 ] && [ -z "$PUB_MSG" ] && PUB_MSG="no GridState on ws://localhost:8765 after 90 s, see logs/publish.log"

# ---------------------------------------------------------------- robot.api
ROBOT_STATE="skipped"
ROBOT_MSG=""
if [ "$ROBOT" = 1 ] && [ "$PUB_OK" = 1 ]; then
    if [ -f robot/api.py ]; then
        say "starting: python -m robot.api  (logs/robot.log)"
        setsid nohup "$PY" -m robot.api >logs/robot.log 2>&1 </dev/null &
        ROBOT_PID=$!
        ROBOT_STATE="fail"
        for _ in $(seq 1 120); do  # start-up looks at the sheet and parks the arm: give it 2 min
            if ! kill -0 "$ROBOT_PID" 2>/dev/null; then ROBOT_MSG="exited, see logs/robot.log"; break; fi
            if STATUS="$(curl -sf --max-time 2 http://127.0.0.1:8770/status)"; then
                ROBOT_STATE="ok"; ROBOT_MSG="$STATUS"; break
            fi
            sleep 1
        done
        [ "$ROBOT_STATE" = fail ] && [ -z "$ROBOT_MSG" ] && ROBOT_MSG="no answer on GET :8770/status after 120 s, see logs/robot.log"
    else
        ROBOT_MSG="robot/api.py not in this checkout"
    fi
elif [ "$ROBOT" = 1 ]; then
    ROBOT_MSG="not started: perception is down"
    ROBOT_STATE="fail"
else
    ROBOT_MSG="--no-robot"
fi

# ---------------------------------------------------------------- summary
echo
if [ "$PUB_OK" = 1 ]; then
    echo "${GREEN}OK    perception.publish  ws://localhost:8765  camera /dev/video$CAM  block ${BLOCK_MM} mm  pid $PUB_PID${RESET}"
else
    echo "${RED}FAIL  perception.publish  $PUB_MSG${RESET}"
    tail -n 15 logs/publish.log 2>/dev/null | sed 's/^/      /'
fi
case "$ROBOT_STATE" in
    ok)   echo "${GREEN}OK    robot.api           http://localhost:8770  $ROBOT_MSG${RESET}" ;;
    fail) echo "${RED}FAIL  robot.api           $ROBOT_MSG${RESET}"
          [ -f logs/robot.log ] && tail -n 15 logs/robot.log | sed 's/^/      /' ;;
    *)    echo "${YELLOW}SKIP  robot.api           $ROBOT_MSG${RESET}" ;;
esac
echo "Stop: pkill -f -- '-m perception.publish'; pkill -f -- '-m robot.api'"

[ "$PUB_OK" = 1 ] && [ "$ROBOT_STATE" != fail ]
