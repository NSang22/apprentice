"""Robot clients for the README 5.3 API (HTTP JSON on ROBOT_HTTP_PORT).

RobotClient is the interface the session talks to. HttpRobotClient calls the real API;
brain/fake_robot.py implements the same interface on a fake board.

Every call returns {"ok": true} or {"ok": false, "error": "<reason>"} and never raises:
timeouts, refused connections, HTTP errors, and malformed replies all come back as
{"ok": false, ...} so the session can journal them and count a failed attempt.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol

from brain.settings import ROBOT_MOTION_TIMEOUT_S, ROBOT_STATUS_TIMEOUT_S

RobotReply = dict[str, Any]


class RobotClient(Protocol):
    def place(self, color: str, cell: str, level: int) -> RobotReply: ...
    def pick(self, cell: str, level: int) -> RobotReply: ...
    def hover(self, cell: str) -> RobotReply: ...
    def home(self) -> RobotReply: ...
    def status(self) -> RobotReply: ...


def fail(error: str) -> RobotReply:
    return {"ok": False, "error": error}


class HttpRobotClient:
    def __init__(
        self,
        base_url: str,
        motion_timeout_s: float = ROBOT_MOTION_TIMEOUT_S,
        status_timeout_s: float = ROBOT_STATUS_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.motion_timeout_s = motion_timeout_s
        self.status_timeout_s = status_timeout_s

    def place(self, color: str, cell: str, level: int) -> RobotReply:
        return self._call("POST", "/place", {"color": color, "cell": cell, "level": level})

    def pick(self, cell: str, level: int) -> RobotReply:
        return self._call("POST", "/pick", {"cell": cell, "level": level})

    def hover(self, cell: str) -> RobotReply:
        return self._call("POST", "/hover", {"cell": cell})

    def home(self) -> RobotReply:
        return self._call("POST", "/home", {})

    def status(self) -> RobotReply:
        reply = self._call("GET", "/status", None, self.status_timeout_s, needs_ok=False)
        reply.setdefault("ok", True)
        return reply

    def _call(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]],
        timeout_s: Optional[float] = None,
        needs_ok: bool = True,
    ) -> RobotReply:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.motion_timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()  # the API may still explain itself as {"ok": false, "error": ...}
            if not raw:
                return fail(f"{method} {path}: HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return fail(f"{method} {path}: {getattr(e, 'reason', e)}")
        try:
            reply = json.loads(raw)
        except ValueError:
            return fail(f"{method} {path}: reply is not JSON")
        if not isinstance(reply, dict) or (needs_ok and not isinstance(reply.get("ok"), bool)):
            return fail(f"{method} {path}: reply has no ok field")
        return reply
