"""Ready-to-demo check: one screen of health for perception (ws :8765), brain (ws :8766), robot (:8770).

    python scripts/check.py                 # on the laptop (through the tunnel) or on the Jetson
    python scripts/check.py --host 10.0.0.5

Read-only: it listens to the feeds and calls GET /status, and never sends a command.
Exits 0 when everything that is running looks ready, 1 otherwise.
Grid lock and tags come from the extra "grid" and "tags" keys perception.publish adds to each
GridState message. A feed without them (brain.fake_feed, an older publish) shows "not reported".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from websockets.asyncio.client import connect  # noqa: E402

from shared.config import (BRAIN_WS_PORT, CELLS, COLUMNS, GRID_WS_PORT, MIN_CONFIDENCE,  # noqa: E402
                           ROBOT_HTTP_PORT, ROWS, UNKNOWN)
from shared.schemas import GridState, SessionState  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
TAG_IDS = [0, 1, 2, 3]  # corner tags on the sheet (README section 5: 0 at A1, 1 at A3, 2 at C3, 3 at C1)
LISTEN_S = 2.0  # how long to watch the grid feed (rate, and the newest frame)


def mark(ok: Optional[bool]) -> str:
    if ok is None:
        return f"{YELLOW}--{RESET}"
    return f"{GREEN}ok{RESET}" if ok else f"{RED}NO{RESET}"


async def read_grid(url: str) -> tuple[list[GridState], dict, Optional[str]]:
    """(frames, newest raw message, error). The raw message keeps publish's extra keys."""
    frames: list[GridState] = []
    raw: dict = {}
    try:
        async with connect(url, open_timeout=3) as ws:
            end = time.monotonic() + LISTEN_S
            while (left := end - time.monotonic()) > 0:
                try:
                    raw = json.loads(await asyncio.wait_for(ws.recv(), left))
                    frames.append(GridState.from_dict(raw))
                except asyncio.TimeoutError:
                    break
    except Exception as exc:  # report, never crash
        hint = " (port open but no ws server behind it: tunnel up, publish down?)" if "HTTP response" in str(exc) else ""
        return frames, raw, f"{type(exc).__name__}: {exc}{hint}"
    return frames, raw, None if frames else f"connected, but no GridState in {LISTEN_S:g} s"


async def read_brain(url: str) -> tuple[Optional[SessionState], Optional[str]]:
    try:
        async with connect(url, open_timeout=3) as ws:  # the server sends its current state on connect
            return SessionState.from_dict(json.loads(await asyncio.wait_for(ws.recv(), 3))), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_robot(url: str) -> tuple[Optional[dict], Optional[str], bool]:
    """(status, error, running). running is False when nothing listens on the port."""
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.loads(r.read()), None, True
    except urllib.error.URLError as exc:
        refused = isinstance(exc.reason, ConnectionRefusedError)
        return None, f"{exc.reason}", not refused
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", True


def board_lines(gs: GridState) -> list[str]:
    width = max([len("/".join(gs.board[c]) or ".") for c in CELLS] + [6]) + 2
    lines = ["      " + "".join(n.ljust(width) for n in ROWS) + "(A = arm edge)"]
    for letter in COLUMNS:
        cells = []
        for n in ROWS:
            cell = letter + n
            text = "/".join(gs.board[cell]) or "."
            bad = UNKNOWN in gs.board[cell] or gs.confidence.get(cell, 1.0) < MIN_CONFIDENCE
            cells.append((RED if bad else "") + text.ljust(width) + (RESET if bad else ""))
        lines.append(f"    {letter} " + "".join(cells))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()
    if sys.platform == "win32":
        import os
        os.system("")  # turns on ANSI colors in the classic Windows console

    async def both():
        return await asyncio.gather(read_grid(f"ws://{args.host}:{GRID_WS_PORT}"),
                                    read_brain(f"ws://{args.host}:{BRAIN_WS_PORT}"))

    (frames, raw, grid_err), (state, brain_err) = asyncio.run(both())
    robot, robot_err, robot_running = read_robot(f"http://{args.host}:{ROBOT_HTTP_PORT}/status")
    problems: list[str] = []

    print(f"perception  ws://{args.host}:{GRID_WS_PORT}")
    if grid_err:
        print(f"  {mark(False)} {grid_err}")
        problems.append("perception: no GridState")
    else:
        gs = frames[-1]
        unknown = [c for c in CELLS if UNKNOWN in gs.board[c]]
        low = [c for c, v in gs.confidence.items() if v < MIN_CONFIDENCE]
        print(f"  {mark(True)} {len(frames) / LISTEN_S:.1f} Hz, frame {gs.frame_id}")
        grid, tags = raw.get("grid"), raw.get("tags")
        if grid is None:
            print(f"  {mark(None)} grid lock: not reported by this feed")
        else:
            print(f"  {mark(grid == 'locked')} grid {grid}")
        if tags is None:
            print(f"  {mark(None)} tags: not reported by this feed")
        else:
            missing = [t for t in TAG_IDS if t not in tags]
            print(f"  {mark(True if not missing else None)} tags seen {tags or 'none'}"
                  + (f"   hidden: {missing} (fine while a block or hand covers them)" if missing else ""))
        print(f"  {mark(gs.hands_clear)} hands_clear   {mark(gs.arm_clear)} arm_clear")
        print(f"  {mark(not unknown and not low)} readable"
              + (f"   unknown: {', '.join(unknown)}" if unknown else "")
              + (f"   low confidence: {', '.join(low)}" if low else ""))
        for line in board_lines(gs):
            print(line)
        if grid is not None and grid != "locked":
            problems.append(f"perception: grid {grid}")
        if not gs.hands_clear:
            problems.append("perception: hands_clear false (hand on the sheet, or grid not locked)")
        if not gs.arm_clear:
            problems.append("perception: arm_clear false")
        if unknown or low:
            problems.append("perception: unreadable cells")

    print(f"\nbrain       ws://{args.host}:{BRAIN_WS_PORT}")
    if brain_err:
        print(f"  {mark(False)} {brain_err}")
        problems.append("brain: not reachable")
    else:
        s = state
        total = len(s.progress.human) or len(s.progress.robot)
        step = s.expected_step
        print(f"  {mark(s.phase == 'ready')} phase {s.phase}, mode {s.mode}, lesson {s.lesson_id}"
              + (f" ({s.lesson_title})" if s.lesson_title else ""))
        print(f"  {mark(s.board_seen)} board_seen   "
              + (f"step {s.step_index + 1}/{total or '?'}: {step.color} {step.cell} L{step.level}" if step
                 else f"all {total} steps done")
              + f"   verdict {s.verdict.value}   robot_status {s.robot_status}")
        if s.hint_text:
            print(f"     hint: {s.hint_text}")
        if s.phase != "ready":
            problems.append(f"brain: phase {s.phase}, not ready (reset with R in the UI)")
        if not s.board_seen:
            problems.append("brain: no stable board yet")

    print(f"\nrobot       http://{args.host}:{ROBOT_HTTP_PORT}/status")
    robot_needed = state is not None and state.mode == "robot"
    if robot is not None:
        good = bool(robot.get("at_home")) and not robot.get("busy") and not robot.get("last_error")
        print(f"  {mark(good)} busy={robot.get('busy')} at_home={robot.get('at_home')} "
              f"last_error={robot.get('last_error')}")
        if not good:
            problems.append("robot: not parked and idle")
    elif not robot_running and not robot_needed:
        print(f"  {mark(None)} not running ({robot_err})")
    else:
        print(f"  {mark(False)} {robot_err}")
        problems.append("robot: no status")

    print()
    if problems:
        print(f"{RED}NOT READY{RESET}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{GREEN}READY TO DEMO{RESET}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
