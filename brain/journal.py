"""Journal: append-only JSONL log of one session (README section 6.6).

One line per event at data/sessions/<session_id>.jsonl:

    {"t": 1728580001.2, "session": "s-0042", "mode": "robot", "kind": "verdict", "data": {...}}

The file is created exclusively (an existing session id raises instead of appending to
someone else's log), every line is flushed as soon as it is written, and nothing is ever
rewritten or deleted. Unknown kinds raise: a typo should fail in tests, not vanish.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from shared.config import SESSIONS_DIR
from shared.schemas import MODES

KINDS = (
    "session_start",
    "board",
    "verdict",
    "step_advance",
    "step_back",
    "hint",
    "robot_cmd",
    "robot_result",
    "gate",
    "note",
    "session_end",
)


def new_session_id(now: Optional[datetime] = None) -> str:
    return "s-" + (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


class Journal:
    def __init__(
        self,
        session_id: str,
        mode: str,
        directory: Path = SESSIONS_DIR,
        clock: Callable[[], float] = time.time,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.session_id = session_id
        self.mode = mode
        self.clock = clock
        self.path = Path(directory) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "x" refuses to open an existing file; the handle then only ever appends.
        self._f = open(self.path, "x", encoding="utf-8", newline="\n")

    @classmethod
    def open_new(cls, mode: str, directory: Path = SESSIONS_DIR, clock: Callable[[], float] = time.time) -> Journal:
        """Open a journal under a fresh timestamped id, adding -2, -3, ... if that second is taken."""
        base = new_session_id()
        for n in range(1, 100):
            try:
                return cls(base if n == 1 else f"{base}-{n}", mode, directory, clock)
            except FileExistsError:
                continue
        raise FileExistsError(f"no free session id for {base} in {directory}")

    def write(self, kind: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"unknown journal kind {kind!r}; expected one of {KINDS}")
        if self._f.closed:
            raise ValueError(f"journal {self.path} is closed")
        line = {"t": self.clock(), "session": self.session_id, "mode": self.mode, "kind": kind, "data": data or {}}
        self._f.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._f.flush()
        return line

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_journal(path: Path) -> list[dict[str, Any]]:
    """Every line of a journal, in order. For tests, replay, and analytics."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
