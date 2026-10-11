"""robot.executor: retry a block that landed off target, aiming off by the miss, at most twice."""

import numpy as np
import pytest

pytest.importorskip("lerobot")  # robot.executor imports the arm skills

import robot.executor as executor  # noqa: E402
from robot.executor import MAX_RETRIES, Executor, TaskError, cell_center  # noqa: E402


@pytest.fixture(autouse=True)
def reports_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "ROOT", tmp_path)  # task reports go here, not robot/outputs


class FakeSkills:
    """Records moves; the block lands at aim + bias (a repeatable miss)."""

    def __init__(self, bias):
        self.bias, self.moves, self.landed = np.asarray(bias, float), [], None

    def plan_move(self, src_xy, src_yaw, aim, yaw, on, pick_neighbours=None, place_neighbours=None, src_base_mm=0.0):
        self.moves.append({"src": np.asarray(src_xy, float), "aim": np.asarray(aim, float), "src_base": src_base_mm})
        return None, None

    def pick(self, *a, **k):
        pass

    def place(self, aim, yaw, on, grip=None):
        self.landed = np.asarray(aim, float) + self.bias

    def park(self, **k):
        pass


class ScriptedExecutor(Executor):
    def __init__(self, skills, start_xy):
        super().__init__(skills, cam=None, loc=None, log=lambda *a: None)
        self.block = np.asarray(start_xy, float)

    def observe(self):
        self.camera_xyz = None  # no parallax: positions read as they are
        if self.sk.landed is not None:
            self.block = self.sk.landed
        return {"red_cube": {"name": "red_cube", "center_mm": self.block.tolist(), "yaw_deg": 0.0,
                             "height_mm": 48.0, "score": 0.9}}


def run(bias):
    sk = FakeSkills(bias)
    ex = ScriptedExecutor(sk, start_xy=cell_center("A1"))
    return sk, ex, ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])


def test_good_landing_needs_no_retry():
    sk, ex, report = run((3.0, -4.0))
    assert len(sk.moves) == 1 and report["steps"][0]["retries"] == 0


def test_repeatable_miss_is_fixed_by_aiming_off():
    sk, ex, report = run((0.0, -19.0))  # the B2 miss from runs 18:34 and 18:47
    step = report["steps"][0]
    assert len(sk.moves) == 2 and step["ok"] and step["error_mm"] < 1.0
    target = cell_center("B2")
    assert np.allclose(sk.moves[1]["aim"], target + (0.0, 19.0))           # aimed off by the miss
    # picked where it landed, corrected by the same offset (the arm is off there too)
    assert np.allclose(sk.moves[1]["src"], target + (0.0, -19.0) - (0.0, -19.0))


def test_aim_shift_is_capped_and_retries_stop_at_two():
    sk = FakeSkills((60.0, 0.0))  # a miss no aiming can fix
    ex = ScriptedExecutor(sk, start_xy=cell_center("A1"))
    with pytest.raises(TaskError, match="after 2 retries"):
        ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    assert len(sk.moves) == MAX_RETRIES + 1
    assert all(np.hypot(*(m["aim"] - cell_center("B2"))) <= 25.0 + 1e-6 for m in sk.moves)


def test_failed_step_puts_the_held_block_back():
    # run 19:42: a stacking step failed with the block still in the jaws, and rest tried to set
    # the arm down on the table with it
    from robot.skills import SkillError

    class FailingPlace(FakeSkills):
        holding, put_back_calls = None, 0

        def pick(self, *a, **k):
            self.holding = (np.zeros(2), 0.0, 0.0)

        def place(self, *a, **k):
            raise SkillError("blocked")

        def put_back(self):
            self.put_back_calls += 1
            self.holding = None

    sk = FailingPlace((0.0, 0.0))
    ex = ScriptedExecutor(sk, start_xy=cell_center("A1"))
    with pytest.raises(TaskError, match="blocked"):
        ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    assert sk.put_back_calls == 1 and sk.holding is None


class World(Executor):
    """Several blocks; the fake arm moves whichever block it picked to exactly where it aims."""

    def __init__(self, blocks):
        sk = FakeSkills((0.0, 0.0))
        super().__init__(sk, cam=None, loc=None, log=lambda *a: None)
        self.blocks = {f"{k}_cube": np.asarray(v, float) for k, v in blocks.items()}
        self.ons = []
        world = self

        def pick(src_xy, yaw, grip=None, base_mm=0.0):
            # the block at the commanded spot; on a stack, the top one
            world.held = min(world.blocks, key=lambda n: (round(float(np.hypot(*(world.blocks[n] - src_xy)))),
                                                          -world.base(n)))

        def place(aim, yaw, on, grip=None):
            world.blocks[world.held] = np.asarray(aim, float)
            world.ons.append(on)

        sk.pick, sk.place = pick, place

    def observe(self):
        self.camera_xyz = None
        return {n: {"name": n, "center_mm": xy.tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
                for n, xy in self.blocks.items()}

    def run_steps(self, *steps):
        for text in steps:
            obj, goal = text.split("->")
            self.moving = f"{obj}_cube"
            g = ({"type": "on_top", "reference": goal[3:]} if goal.startswith("on:")
                 else {"type": "cell", "cell": goal})
            self.run([{"object": obj, "goal": g}])


def test_three_block_tower_releases_at_each_level():
    w = World({"red": cell_center("A1"), "purple": cell_center("A3"), "teal": cell_center("C1")})
    w.run_steps("purple->on:red", "teal->on:purple")
    assert w.ons == [48.0, 96.0]
    assert w.base("teal_cube") == 96.0


def test_cannot_move_a_block_that_has_one_on_top():
    w = World({"red": cell_center("A1"), "purple": cell_center("A3")})
    w.run_steps("purple->on:red")
    with pytest.raises(TaskError, match="purple on top"):
        w.run_steps("red->C3")


def test_unstacking_frees_the_block_below():
    w = World({"red": cell_center("A1"), "purple": cell_center("A3")})
    w.run_steps("purple->on:red", "purple->C3", "red->B2")
    assert w.base("purple_cube") == 0.0 and np.allclose(w.blocks["red_cube"], cell_center("B2"))


def test_stack_miss_corrects_the_repick_too_and_is_remembered():
    # run 20:52: yellow landed ~20 mm off on top of a tower; the retry corrected only the place,
    # picked the block off-centre at the same height, and it toppled. Both moves get the offset.
    class HighError(FakeSkills):
        def place(self, aim, yaw, on, grip=None):
            self.landed = np.asarray(aim, float) + (self.bias if on > 0 else 0.0)

    sk = HighError((10.0, 17.0))
    w = World.__new__(World)
    Executor.__init__(w, sk, cam=None, loc=None, log=lambda *a: None)
    w.blocks = {"red_cube": cell_center("B2"), "purple_cube": cell_center("C1")}
    world_pick = {}
    sk.pick = lambda src_xy, yaw, grip=None, base_mm=0.0: world_pick.update(held="purple_cube")

    def place(aim, yaw, on, grip=None):
        HighError.place(sk, aim, yaw, on)
        w.blocks["purple_cube"] = sk.landed
    sk.place = place
    w.moving = "purple_cube"
    w.run([{"object": "purple", "goal": {"type": "on_top", "reference": "red"}}])
    target = cell_center("B2")
    assert len(sk.moves) == 2
    assert np.allclose(sk.moves[1]["aim"], target - (10.0, 17.0))
    assert np.allclose(sk.moves[1]["src"], target + (10.0, 17.0) - (10.0, 17.0))  # re-pick corrected too
    assert np.allclose(executor.load_level_offsets()[48.0], (10.0, 17.0))          # kept for the next run


def test_blocks_crowding_the_target_are_moved_aside_first():
    # run 21:09: purple sat 51/42 mm from B2 and caught red's corner on the way down
    from robot.executor import too_close
    w = World({"red": (61.5, 174.7), "purple": (159.4, 176.0), "yellow": (143.2, 78.5)})
    w.moving = "red_cube"
    report = w.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    b2 = cell_center("B2")
    assert np.allclose(w.blocks["red_cube"], b2)
    asides = [s for s in report["steps"] if s["step"] == "aside"]
    assert asides and all(s["ok"] for s in asides)
    for n in ("purple_cube", "yellow_cube"):  # nothing left crowding B2, and nothing crowding each other
        assert not too_close(w.blocks[n], b2)
    assert not too_close(w.blocks["purple_cube"], w.blocks["yellow_cube"])


def test_reference_block_is_never_moved_aside():
    w = World({"red": cell_center("B2"), "purple": cell_center("C1")})
    w.moving = "purple_cube"
    report = w.run([{"object": "purple", "goal": {"type": "beside", "reference": "red", "side": "right"}}])
    assert np.allclose(w.blocks["red_cube"], cell_center("B2"))
    assert not [s for s in report["steps"] if s["step"] == "aside"]


def test_free_spot_stays_on_the_grid_and_clear():
    from robot.executor import grid_box, too_close
    w = World({"red": cell_center("B2"), "purple": cell_center("B1"), "yellow": cell_center("A2")})
    objs = w.observe()
    spot = w.free_spot("purple_cube", objs, avoid=[cell_center("C1")])
    x0, y0, x1, y1 = grid_box()
    assert x0 + 24 <= spot[0] <= x1 - 24 and y0 + 24 <= spot[1] <= y1 - 24
    assert not any(too_close(spot, p) for p in (cell_center("B2"), cell_center("A2"), cell_center("C1")))


def test_hemmed_in_block_gets_a_neighbour_moved_then_retries():
    from robot.skills import SkillError
    w = World({"red": cell_center("B2"), "purple": cell_center("B3"), "yellow": cell_center("A1")})
    calls = {"n": 0}
    plan = w.sk.plan_move

    def picky_plan(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:  # first try at moving red: no grip fits around it
            raise SkillError("every side of the block is blocked by a neighbouring block (best gap -3 mm)")
        return plan(*a, **k)
    w.sk.plan_move = picky_plan
    report = w.run([{"object": "red", "goal": {"type": "cell", "cell": "C1"}}])
    assert np.allclose(w.blocks["red_cube"], cell_center("C1"))
    aside = [s for s in report["steps"] if s["step"] == "aside"]
    assert len(aside) == 1 and aside[0]["description"].startswith("purple")  # red's nearest neighbour
