#!/usr/bin/env bash
# Start the Jetson dashboard (jetson-kushi/dashboard.py) on :8080 with the lerobot venv and
# this board's .env (ROBOT_PORT, CAMERA_INDEX). Run from anywhere:
#
#   ~/Desktop/apprentice/jetson-kushi/dashboard.sh
#
# then open http://<jetson-ip>:8080 on a phone or laptop on the same network.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

# shellcheck disable=SC1091
source "$HOME/lerobot/.venv/bin/activate"
if [ -f "$HERE/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$HERE/.env"
    set +a
fi

exec python "$HERE/dashboard.py" "$@"
