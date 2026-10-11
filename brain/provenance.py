"""Session provenance: where the boards and robot moves came from, and which code ran.

Recorded in every session_start journal event so eval numbers from a fake feed or fake arm
can never pass for real ones. Feeds tag each GridState message with an extra "source" key
("fake_feed", "perception"); GridState.from_dict ignores unknown keys, so the contract is
unchanged and older feeds without the key still parse.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from shared.config import REPO_ROOT

UNKNOWN = "unknown"
GIT_TIMEOUT_S = 2.0


def grid_source(grid: str, seen_source: Optional[str]) -> str:
    """"sim", "fake_feed", or "perception:<url>". A ws feed that has not sent a frame yet, or
    sends an unrecognized source, is "unknown:<url>" rather than a guess."""
    if grid == "sim":
        return "sim"
    if seen_source == "fake_feed":
        return "fake_feed"
    if seen_source == "perception":
        return f"perception:{grid}"
    return f"{UNKNOWN}:{grid}"


def robot_source(spec: Optional[str]) -> str:
    if spec is None:
        return "none"
    if spec == "fake":
        return "fake"
    return f"http:{spec}"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_info() -> dict[str, object]:
    """{"git_commit": hash | "unknown", "git_dirty": bool | "unknown"}. Untracked files do not
    count as dirty (journals and lessons are untracked by design)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit.strip() if commit else UNKNOWN,
        "git_dirty": UNKNOWN if status is None else bool(status.strip()),
    }
