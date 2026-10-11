"""Pip's help on the table: the fix timer never runs while the arm is still moving."""

from buddy import recall, session
from buddy.session import BuddySession
from shared.config import CELLS


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class SlowPip:
    """The arm takes as long as the test says: busy until finish() is called."""

    def __init__(self):
        self.busy, self.step, self.error, self.left_over, self.calls = False, "", None, [], []

    def reachable(self):
        return True

    def point(self, cell, should_stop=None):
        self.calls.append(("point", cell))
        self.busy, self.step = True, "moving"

    def remove(self, cells, should_stop=None):
        self.calls.append(("remove", list(cells)))
        self.busy, self.step = True, "removing"

    def finish(self):
        self.busy, self.step = False, "done"


class Table:
    def ready(self):
        return True

    def hands_away(self):
        return True

    def status(self):
        return {}

    latest = None


def missed_round(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    clock, pip = Clock(), SlowPip()
    s = BuddySession(table=Table(), pip=pip, clock=clock)
    s.profile.data["source"] = "table"
    s.session_id, s.round, s.phase, s.attempt = "test", 1, "choice", 1
    s.pattern = {"C1": "red", "B2": "teal"}
    board = {c: [] for c in CELLS}
    board["B2"], board["A3"] = ["teal"], ["purple"]
    s.grade = recall.grade(s.pattern, board)
    return s, clock, pip


def test_fix_timer_waits_for_the_arm(tmp_path, monkeypatch):
    s, clock, pip = missed_round(tmp_path, monkeypatch)
    s.choose("hint")
    assert pip.calls == [("point", "C1")] and s.phase == "helping"
    clock.t += 60  # a very slow arm: a minute later it still hasn't arrived
    s.tick()
    assert s.phase == "helping"
    pip.finish()
    s.tick()
    assert s.phase == "build" and s.attempt == 2
    assert s.until - clock.t == recall.RETRY_S  # the full 20 s, counted from when the arm is done


def test_fix_removes_only_wrong_blocks_before_the_timer(tmp_path, monkeypatch):
    s, clock, pip = missed_round(tmp_path, monkeypatch)
    s.choose("fix")
    assert pip.calls == [("remove", ["A3"])]
    clock.t += 30
    s.tick()
    assert s.phase == "helping"
    pip.finish()
    s.tick()
    assert s.phase == "build" and s.until - clock.t == recall.RETRY_S and s.removing == []
