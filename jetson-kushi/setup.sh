#!/usr/bin/env bash
# One-time Jetson environment setup. Run from inside this folder, after cloning
# the main repo onto the Jetson:
#
#   git clone <repo-url> apprentice
#   cd apprentice/jetson-kushi
#   ./setup.sh
#
# This installs dependencies only - it does not copy or duplicate any logic from
# robot/, perception/, brain/, or ui/. See README.md in this folder for why that
# matters.

set -euo pipefail

ENV_DIR="$HOME/lerobot/.venv"   # same convention docs/SETUP_NOTES.md uses on Windows

if [ ! -d "$HOME/lerobot/.git" ]; then
    echo "~/lerobot (the Seeed lerobot fork) isn't cloned yet - do that first, see docs/SETUP_NOTES.md"
    exit 1
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv at $ENV_DIR"
    python3 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "Installing lerobot (editable, feetech extra - same as the Windows setup)..."
pip install -e "$HOME/lerobot[feetech]"

echo "Installing jetson-kushi/requirements.txt..."
pip install -r "$(dirname "$0")/requirements.txt"

echo
echo "Done. Verify with:"
echo "  python -c 'import lerobot, cv2; print(\"lerobot ok, cv2\", cv2.__version__)'"
echo "Then copy env.example to .env, fill in ROBOT_PORT for this board, and source it"
echo "before running anything under robot/."
