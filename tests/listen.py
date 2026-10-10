"""Print each GridState from the feed on one line.

python -m tests.listen [--url ws://localhost:8765]
"""

import argparse
import asyncio
import json

from websockets.asyncio.client import connect

from shared.config import CELLS, GRID_WS_PORT, MIN_CONFIDENCE
from shared.schemas import GridState


def one_line(gs: GridState) -> str:
    cells = " ".join(f"{c}={'/'.join(gs.board[c])}" for c in CELLS if gs.board[c]) or "(empty)"
    low = [c for c, v in gs.confidence.items() if v < MIN_CONFIDENCE]
    flags = f"hands={'ok' if gs.hands_clear else 'IN'} arm={'ok' if gs.arm_clear else 'OUT'}"
    return f"#{gs.frame_id:<6} {flags} {cells}" + (f"  low_conf={low}" if low else "")


async def main(url: str) -> None:
    async with connect(url) as ws:
        print(f"connected to {url}")
        async for msg in ws:
            print(one_line(GridState.from_dict(json.loads(msg))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=f"ws://localhost:{GRID_WS_PORT}")
    try:
        asyncio.run(main(p.parse_args().url))
    except KeyboardInterrupt:
        pass
