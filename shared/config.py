"""Shared constants for every Apprentice module.

Changing anything here requires the whole team to agree (see AGENTS.md).
"""

from pathlib import Path

# Grid. Columns A to C left to right from the operator side; rows 1 to 3
# front (nearest the operator) to back. A1 = front-left, C3 = back-right.
COLUMNS = ["A", "B", "C"]
ROWS = ["1", "2", "3"]
CELLS = [c + r for c in COLUMNS for r in ROWS]  # A1, A2, A3, B1, ..., C3

COLORS = ["red", "blue", "yellow", "green"]
UNKNOWN = "unknown"
ALL_COLORS = COLORS + [UNKNOWN]  # every color string allowed in a GridState board

# Ports
GRID_WS_PORT = 8765  # perception -> brain (GridState)
BRAIN_WS_PORT = 8766  # brain -> ui (SessionState)
ROBOT_HTTP_PORT = 8770  # brain -> robot

# Thresholds
STABLE_FRAMES = 5  # feed runs ~10 Hz, so ~0.5 s
MIN_CONFIDENCE = 0.8
MAX_ROBOT_RETRIES = 2
MAX_LEVEL = 2  # levels 0..MAX_LEVEL, so stacks hold at most MAX_LEVEL + 1 cubes
LEVELS = list(range(MAX_LEVEL + 1))

# Paths, anchored at the repo root so they work from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_DIR = REPO_ROOT / "lessons"
SESSIONS_DIR = REPO_ROOT / "data" / "sessions"
