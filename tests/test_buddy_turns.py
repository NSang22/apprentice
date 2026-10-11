import random

from buddy import session, turns, turns_session
from buddy.turns_session import TurnsSession
from shared.config import CELLS

TARGET = {"A1": "red", "A3": "teal", "C1": "yellow", "C3": "purple"}


def board(**cells):
    b = {c: [] for c in CELLS}
    for cell, color in cells.items():
        b[cell] = [color]
    return b


def test_pip_starts_on_its_own_edge():
    assert turns.pip_next(TARGET, board()) == ("A1", "red")
    assert turns.pip_next(TARGET, board(A1="red", A3="teal")) == ("C1", "yellow")
    assert turns.pip_next(TARGET, board(A1="red", A3="teal", C1="yellow", C3="purple")) is None


def test_child_turn_judgement():
    before = board(A1="red")
    assert turns.child_turn(TARGET, before, right_before=1)["result"] == "none"
    assert turns.child_turn(TARGET, board(A1="red", C3="purple"), 1)["result"] == "ok"
    wrong_spot = turns.child_turn(TARGET, board(A1="red", B2="purple"), 1)
    assert wrong_spot["result"] == "wrong" and wrong_spot["point"] == "C3"      # where purple belongs
    wrong_color = turns.child_turn(TARGET, board(A1="red", C3="teal"), 1)
    assert wrong_color["result"] == "wrong" and wrong_color["point"] == "C3"    # this square needs purple


# ---------------------------------------------------------------- the game loop


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakePip:
    def __init__(self, reachable=True):
        self._reachable, self.busy, self.step, self.error, self.calls = reachable, False, "", None, []

    def reachable(self):
        return self._reachable

    def place(self, cell, color, should_stop=None):
        self.calls.append(("place", cell, color))
        self.busy = True

    def point(self, cell, should_stop=None):
        self.calls.append(("point", cell))
        self.busy = True

    def finish(self, error=None):
        self.busy, self.error = False, error


class FakeTable:
    latest = None

    def ready(self):
        return True

    def hands_away(self):
        return True

    def scanned(self, since):
        return board()

    def status(self):
        return {}


class FakeEyes:
    info = {}          # no Gemini: a failed place goes straight to asking for the color

    def warm(self):
        pass


class FakeScanner:
    """Returns whatever board the test puts in `next_board` as soon as a scan starts."""

    def __init__(self):
        self.next_board, self.active, self.last_sync = None, False, None

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def poll(self):
        if not self.active:
            return {"status": "idle"}
        self.active = False
        return {"status": "done", "board": self.next_board}


def game(tmp_path, monkeypatch, source, pip=None):
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(turns_session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(turns, "BLOCKS_PER_ROUND", [4])
    clock = Clock()
    profile = session.Profile(tmp_path / "profile.json")
    profile.data.update({"name": "Kai", "source": source})
    t = TurnsSession(table=FakeTable(), eyes=FakeEyes(), pip=pip or FakePip(), profile=profile, clock=clock,
                     rng=random.Random(1))
    t.scanner = FakeScanner()
    t.start()
    t.target = dict(TARGET)                 # a known pattern instead of the random one
    t.pip_move = turns.pip_next(t.target, t.current())
    t.glow = t.pip_move[0]
    return t, clock


def test_practice_round_alternates_pip_first(tmp_path, monkeypatch):
    t, clock = game(tmp_path, monkeypatch, "practice")
    assert t.phase == "pip_turn"
    clock.t += turns_session.PRACTICE_PIP_S
    t.tick()
    assert t.board["A1"] == ["red"] and t.phase == "child_turn"
    t.your_turn_pip()                                   # tapped without placing anything
    assert t.phase == "child_turn" and "first" in t.say
    t.set_cell("B2", "purple")                          # wrong square: Pip shows where purple goes
    t.your_turn_pip()
    assert t.phase == "child_turn" and t.glow == "C3" and t.mistakes == 1
    t.set_cell("B2", None)
    t.set_cell("C3", "purple")
    t.your_turn_pip()
    assert t.phase == "pip_turn" and t.pip_move == ("A3", "teal")
    clock.t += turns_session.PRACTICE_PIP_S
    t.tick()
    t.set_cell("C1", "yellow")
    t.your_turn_pip()
    assert t.phase == "round_done" and t.stars == [2]  # one slip this round


def test_table_pip_miss_asks_the_child_to_move_its_block(tmp_path, monkeypatch):
    pip = FakePip()
    t, clock = game(tmp_path, monkeypatch, "table", pip)
    t.arm_used = True
    t.pip.place(*t.pip_move)                            # (game() swapped the pattern after start)
    pip.finish()
    t.tick()
    assert t.phase == "pip_check"
    t.scanner.next_board = board(B1="red")              # the arm put red one square off
    t.tick()
    assert t.phase == "child_turn" and t.glow == "A1" and t.pip_misses == 1 and "missed" in t.say
    t.your_turn_pip()
    t.scanner.next_board = board(A1="red", C3="purple")  # moved Pip's block and added their own
    t.tick()
    assert t.phase == "pip_turn" and t.mistakes == 0 and t.helped_pip == 1
    assert pip.calls[-1] == ("place", "A3", "teal")


def test_arm_cannot_find_a_block_in_the_supply(tmp_path, monkeypatch):
    pip = FakePip()
    t, clock = game(tmp_path, monkeypatch, "table", pip)
    t.arm_used = True
    pip.busy = True
    pip.finish(error="no red block in the supply")
    t.tick()
    assert t.phase == "need_supply" and "red block below the grid" in t.say
