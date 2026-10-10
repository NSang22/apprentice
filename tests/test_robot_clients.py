import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from brain.fake_robot import FakeRobot
from brain.robot_client import HttpRobotClient
from shared.schemas import empty_board


# ---------------------------------------------------------------- FakeRobot


def test_fake_place_and_pick_edit_the_board():
    r = FakeRobot()
    assert r.place("red", "A1", 0) == {"ok": True}
    assert r.place("blue", "A1", 1) == {"ok": True}
    assert r.board["A1"] == ["red", "blue"]
    assert r.pick("A1", 1) == {"ok": True}
    assert r.board["A1"] == ["red"]


def test_fake_shares_the_board_it_was_given():
    board = empty_board()
    FakeRobot(board).place("green", "B2", 0)
    assert board["B2"] == ["green"]


def test_fake_arm_leaves_home_until_home_is_called():
    r = FakeRobot()
    r.place("red", "A1", 0)
    assert r.status()["at_home"] is False
    r.hover("B2")
    assert r.snapshot()[1] is False
    r.home()
    assert r.status() == {"ok": True, "busy": False, "at_home": True, "last_error": None}


def test_fake_wrong_cell_on_attempt_n():
    r = FakeRobot(wrong_cell_on={2})
    r.place("red", "A1", 0)
    assert r.place("green", "B1", 0) == {"ok": True}
    assert r.board["B1"] == []
    assert sum(len(s) for s in r.board.values()) == 2  # landed on a neighbor instead


def test_fake_fail_on_attempt_n_moves_nothing():
    r = FakeRobot(fail_on={1})
    reply = r.place("red", "A1", 0)
    assert reply["ok"] is False and "injected" in reply["error"]
    assert r.board == empty_board()
    assert r.status()["last_error"] == reply["error"]
    assert r.place("red", "A1", 0) == {"ok": True}


def test_fake_rejects_impossible_moves():
    r = FakeRobot()
    assert r.place("red", "A1", 1)["ok"] is False  # floating cube
    assert r.pick("A1", 0)["ok"] is False          # nothing there
    assert r.place("purple", "A1", 0)["ok"] is False
    assert r.hover("D4")["ok"] is False
    r.place("red", "A1", 0)
    r.place("blue", "A1", 1)
    assert r.pick("A1", 0)["ok"] is False          # not the top cube


def test_fake_records_calls():
    r = FakeRobot()
    r.place("red", "A1", 0)
    r.home()
    assert r.calls == [("place", "red", "A1", 0), ("home",)]


# ---------------------------------------------------------------- HttpRobotClient


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    received: list = []

    def log_message(self, *args):
        pass

    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        Handler.received.append((self.command, self.path, body))
        status, payload, delay = Handler.routes.get(self.path, (404, b"", 0))
        time.sleep(delay)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply


@pytest.fixture
def api():
    Handler.routes, Handler.received = {}, []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", Handler
    server.shutdown()
    server.server_close()


def test_http_place_sends_readme_body(api):
    url, h = api
    h.routes["/place"] = (200, b'{"ok": true}', 0)
    assert HttpRobotClient(url).place("blue", "B2", 1) == {"ok": True}
    assert h.received == [("POST", "/place", {"color": "blue", "cell": "B2", "level": 1})]


def test_http_all_calls_hit_their_routes(api):
    url, h = api
    for path in ("/pick", "/hover", "/home"):
        h.routes[path] = (200, b'{"ok": true}', 0)
    h.routes["/status"] = (200, b'{"busy": false, "at_home": true, "last_error": null}', 0)
    c = HttpRobotClient(url)
    assert c.pick("A1", 0)["ok"] and c.hover("C3")["ok"] and c.home()["ok"]
    assert c.status() == {"ok": True, "busy": False, "at_home": True, "last_error": None}
    assert [(m, p) for m, p, _ in h.received] == [("POST", "/pick"), ("POST", "/hover"), ("POST", "/home"), ("GET", "/status")]


def test_http_ok_false_passes_through(api):
    url, h = api
    h.routes["/place"] = (200, b'{"ok": false, "error": "gripper slipped"}', 0)
    assert HttpRobotClient(url).place("red", "A1", 0) == {"ok": False, "error": "gripper slipped"}


def test_http_error_with_json_body_passes_through(api):
    url, h = api
    h.routes["/pick"] = (500, b'{"ok": false, "error": "arm fault"}', 0)
    assert HttpRobotClient(url).pick("A1", 0) == {"ok": False, "error": "arm fault"}


def test_http_error_without_body_is_a_failure(api):
    url, _ = api
    reply = HttpRobotClient(url).home()  # no route -> 404, empty body
    assert reply["ok"] is False and "404" in reply["error"]


def test_http_non_json_reply_is_a_failure(api):
    url, h = api
    h.routes["/home"] = (200, b"<html>", 0)
    assert HttpRobotClient(url).home()["ok"] is False


def test_http_reply_without_ok_is_a_failure(api):
    url, h = api
    h.routes["/home"] = (200, b'{"done": true}', 0)
    assert HttpRobotClient(url).home()["ok"] is False


def test_http_timeout_is_a_failure(api):
    url, h = api
    h.routes["/place"] = (200, b'{"ok": true}', 1.0)
    reply = HttpRobotClient(url, motion_timeout_s=0.2).place("red", "A1", 0)
    assert reply["ok"] is False


def test_http_connection_refused_is_a_failure():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    reply = HttpRobotClient(f"http://127.0.0.1:{port}", motion_timeout_s=1.0).home()
    assert reply["ok"] is False
