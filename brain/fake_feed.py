"""Fake perception feed: publishes GridState JSON on GRID_WS_PORT at ~10 Hz.

Scripted:     python -m brain.fake_feed --script tests/scripts/staircase_clean.json [--loop]
Interactive:  python -m brain.fake_feed --interactive

A script is a JSON list of segments:
  {"board": {...}, "duration_s": 1.5, "hands_clear": true, "arm_clear": true, "jitter": false}
Cells left out of a segment's board are treated as empty. Only "board" and
"duration_s" are required; the flags default to true, true, false.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from websockets.asyncio.server import ServerConnection, broadcast, serve

from shared.config import ALL_COLORS, CELLS, COLUMNS, GRID_WS_PORT, MAX_LEVEL, ROWS
from shared.schemas import Board, GridState, copy_board, empty_board, validate_board

HZ = 10.0
JITTER_PROB = 0.2
CONF_OK = 0.95
CONF_JITTER = 0.5


@dataclass
class Segment:
    board: Board
    duration_s: float
    hands_clear: bool = True
    arm_clear: bool = True
    jitter: bool = False


# ---------------------------------------------------------------- pure parts


def parse_script(data: Any) -> list[Segment]:
    if not isinstance(data, list) or not data:
        raise ValueError("script must be a non-empty JSON list of segments")
    segments = []
    for i, seg in enumerate(data):
        if "board" not in seg or "duration_s" not in seg:
            raise ValueError(f"segment {i}: needs 'board' and 'duration_s'")
        board = empty_board()
        board.update({cell: list(stack) for cell, stack in seg["board"].items()})
        validate_board(board)
        segments.append(
            Segment(
                board=board,
                duration_s=float(seg["duration_s"]),
                hands_clear=bool(seg.get("hands_clear", True)),
                arm_clear=bool(seg.get("arm_clear", True)),
                jitter=bool(seg.get("jitter", False)),
            )
        )
    return segments


def make_frame(
    board: Board,
    hands_clear: bool,
    arm_clear: bool,
    jitter: bool,
    frame_id: int,
    rng: random.Random,
    t: Optional[float] = None,
) -> GridState:
    """Build one GridState. With jitter, ~20% of frames flip one occupied
    cell's top color to another color or "unknown" at confidence 0.5."""
    out = copy_board(board)
    confidence = {cell: CONF_OK for cell in CELLS}
    occupied = [cell for cell in CELLS if out[cell]]
    if jitter and occupied and rng.random() < JITTER_PROB:
        cell = rng.choice(occupied)
        top = out[cell][-1]
        out[cell][-1] = rng.choice([c for c in ALL_COLORS if c != top])
        confidence[cell] = CONF_JITTER
    return GridState(
        t=time.time() if t is None else t,
        board=out,
        arm_clear=arm_clear,
        hands_clear=hands_clear,
        confidence=confidence,
        frame_id=frame_id,
    )


def render_board(board: Board) -> str:
    """3x3 text grid, back row (3) at the top, front row (1) at the bottom."""
    width = max(9, max((len("/".join(board[c])) for c in CELLS), default=0))
    lines = ["     " + " ".join(col.center(width) for col in COLUMNS)]
    for row in reversed(ROWS):
        cells = ["/".join(board[col + row]) or "." for col in COLUMNS]
        lines.append(f"  {row}  " + " ".join(c.center(width) for c in cells))
    lines.append("     (front / operator side)")
    return "\n".join(lines)


# ---------------------------------------------------------------- server


class Feed:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.clients: set[ServerConnection] = set()
        self.frame_id = 0
        self.board = empty_board()
        self.hands_clear = True
        self.arm_clear = True
        self.jitter = False

    async def handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        print(f"[feed] client connected ({len(self.clients)} total)", flush=True)
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)
            print(f"[feed] client disconnected ({len(self.clients)} total)", flush=True)

    def publish(self) -> None:
        frame = make_frame(
            self.board, self.hands_clear, self.arm_clear, self.jitter, self.frame_id, self.rng
        )
        self.frame_id += 1
        # "source" is an extra key GridState.from_dict ignores; brain.server records it as provenance.
        broadcast(self.clients, json.dumps({**frame.to_dict(), "source": "fake_feed"}))

    async def tick_forever(self, until: Optional[float] = None) -> None:
        """Publish at HZ on absolute deadlines (no drift) until `until`, or forever."""
        period = 1.0 / HZ
        next_t = time.monotonic()
        while until is None or next_t < until:
            self.publish()
            next_t += period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    async def run_script(self, segments: list[Segment], loop: bool) -> None:
        while True:
            for i, seg in enumerate(segments):
                self.board = copy_board(seg.board)
                self.hands_clear, self.arm_clear, self.jitter = seg.hands_clear, seg.arm_clear, seg.jitter
                print(
                    f"[feed] segment {i + 1}/{len(segments)} for {seg.duration_s}s "
                    f"hands_clear={seg.hands_clear} arm_clear={seg.arm_clear} jitter={seg.jitter}",
                    flush=True,
                )
                await self.tick_forever(until=time.monotonic() + seg.duration_s)
            if not loop:
                break
            print("[feed] looping script", flush=True)
        print("[feed] script finished, holding the final segment (Ctrl+C to stop)", flush=True)
        await self.tick_forever()

    async def run_interactive(self) -> None:
        ticker = asyncio.create_task(self.tick_forever())
        print(HELP, flush=True)
        try:
            while True:
                try:
                    line = await asyncio.to_thread(input, "feed> ")
                except EOFError:
                    break
                if self.handle_command(line.strip()) is False:
                    break
        finally:
            ticker.cancel()

    def handle_command(self, line: str) -> Optional[bool]:
        """Apply one keyboard command. Returns False to quit."""
        parts = line.split()
        if not parts:
            return None
        cmd, args = parts[0].lower(), parts[1:]
        if cmd == "place" and len(args) == 2:
            color, cell = args[0].lower(), args[1].upper()
            if color not in ALL_COLORS:
                print(f"bad color {color!r}; use one of {ALL_COLORS}")
            elif cell not in CELLS:
                print(f"bad cell {cell!r}")
            elif len(self.board[cell]) > MAX_LEVEL:
                print(f"{cell} is full ({MAX_LEVEL + 1} cubes)")
            else:
                self.board[cell].append(color)
                print(f"placed {color} at {cell} level {len(self.board[cell]) - 1}")
        elif cmd == "remove" and len(args) == 1:
            cell = args[0].upper()
            if cell not in CELLS:
                print(f"bad cell {cell!r}")
            elif not self.board[cell]:
                print(f"{cell} is already empty")
            else:
                print(f"removed {self.board[cell].pop()} from {cell}")
        elif cmd == "hands":
            self.hands_clear = not self.hands_clear
            print(f"hands_clear = {self.hands_clear}")
        elif cmd == "arm":
            self.arm_clear = not self.arm_clear
            print(f"arm_clear = {self.arm_clear}")
        elif cmd == "jitter":
            self.jitter = not self.jitter
            print(f"jitter = {self.jitter}")
        elif cmd == "clear":
            self.board = empty_board()
            print("board cleared")
        elif cmd == "show":
            print(render_board(self.board))
            print(f"hands_clear={self.hands_clear} arm_clear={self.arm_clear} jitter={self.jitter}")
        elif cmd in ("help", "?"):
            print(HELP)
        elif cmd in ("quit", "exit"):
            return False
        else:
            print(f"unknown command {line!r}; type 'help'")
        return None


HELP = f"""commands:
  place <color> <cell>   push a cube onto a cell (colors: {', '.join(ALL_COLORS)})
  remove <cell>          pop the top cube off a cell
  hands | arm | jitter   toggle hands_clear / arm_clear / jitter
  clear                  empty the board
  show                   print the board (front row at the bottom)
  quit"""


async def main_async(args: argparse.Namespace) -> None:
    feed = Feed(random.Random(args.seed))
    segments = None
    if args.script:
        segments = parse_script(json.loads(Path(args.script).read_text()))
    async with serve(feed.handler, args.host, args.port):
        print(f"[feed] publishing GridState on ws://{args.host}:{args.port} at {HZ:g} Hz", flush=True)
        if segments is not None:
            await feed.run_script(segments, args.loop)
        else:
            await feed.run_interactive()


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--script", help="path to a tests/scripts/*.json segment list")
    mode.add_argument("--interactive", action="store_true", help="type commands to edit the board")
    p.add_argument("--loop", action="store_true", help="repeat the script forever")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=GRID_WS_PORT)
    p.add_argument("--seed", type=int, default=None, help="seed jitter for repeatable runs")
    args = p.parse_args(argv)
    if args.loop and not args.script:
        p.error("--loop only applies to --script")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
