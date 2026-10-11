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

    def observe(self, source=None):
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


def test_slip_is_not_learned_so_the_repick_goes_where_the_block_is():
    # Run 00:12: the block slipped 41 mm in the jaws; the retry shifted the pick by the miss
    # and closed on nothing. A slip doesn't repeat, so pick it where it is and aim straight.
    sk = FakeSkills((0.0, -41.0))
    ex = ScriptedExecutor(sk, start_xy=cell_center("A1"))
    place = sk.place
    sk.place = lambda *a, **k: (place(*a, **k), setattr(sk, "bias", np.zeros(2)))  # slips once
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    target = cell_center("B2")
    assert len(sk.moves) == 2 and report["steps"][0]["ok"]
    assert np.allclose(sk.moves[1]["src"], target + (0.0, -41.0))  # where the camera saw it
    assert np.allclose(sk.moves[1]["aim"], target)


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

    def observe(self, source=None):
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


class SceneCam:
    def read(self):
        return None


def bank_executor(monkeypatch, objects):
    scene = {"sheet": {"camera_xyz_mm": None}, "objects": [
        {"name": n, "center_mm": list(xy), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9} for n, xy in objects]}
    monkeypatch.setattr(executor, "build_scene", lambda frame, loc: (scene, None, []))
    return Executor(FakeSkills((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)


def test_a_colour_missing_from_the_sheet_is_taken_from_the_bank(monkeypatch):
    ex = bank_executor(monkeypatch, [("purple_cube", cell_center("A1")), ("red_cube", (290.0, 120.0)),
                                     ("purple_cube_2", (290.0, 50.0))])
    objs = ex.observe(source="red_cube")
    assert np.allclose(objs["red_cube"]["center_mm"], (290.0, 120.0))   # building from the bank
    assert "red_cube" not in ex.bank and "purple_cube" in ex.bank       # the bank's purple, renumbered
    objs = ex.observe(source="purple_cube")
    assert np.allclose(objs["purple_cube"]["center_mm"], cell_center("A1"))  # the sheet's block wins


def test_bank_blocks_count_as_neighbours_but_not_as_sheet_blocks(monkeypatch):
    ex = bank_executor(monkeypatch, [("red_cube", (290.0, 120.0)), ("yellow_cube", (290.0, 190.0))])
    objs = ex.observe(source="red_cube")
    assert set(objs) == {"red_cube"}                                    # yellow stays a spare
    assert [n[:2] for n in ex.neighbours("red_cube", objs)] == [(290.0, 190.0)]  # but the jaws avoid it


def test_a_missing_block_outside_the_bank_is_reported_with_where_it_is(monkeypatch):
    # Run 01:47: purple sat above the bank, out of reach, and the task only said "can't see 'purple'".
    def scene(objects):
        return {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": list(xy), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9} for n, xy in objects]}
    seen = [("red_cube", (290.0, 120.0))]
    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (
        scene(seen + ([("purple_cube", (247.0, -13.0))] if bank_zone else [])), None, []))
    monkeypatch.setattr(executor, "MISSING_WAIT_S", 0.0)
    ex = Executor(FakeSkills((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    with pytest.raises(TaskError, match=r"purple.*there is one at \(247, -13\) mm, outside the bank"):
        ex.run([{"object": "purple", "goal": {"type": "cell", "cell": "B2"}}])


def test_a_bank_block_boxed_in_by_its_neighbour_gets_the_neighbour_moved_first(monkeypatch):
    # Run 02:30: red at the top of the bank had purple almost beside it; no grip fit, and only
    # sheet blocks were considered for moving, so the task stopped.
    from robot.skills import SkillError
    state = {"red_cube": np.array([246.0, 37.0]), "purple_cube": np.array([297.0, 62.0])}
    start = {k: v.copy() for k, v in state.items()}

    class BoxedSkills(FakeSkills):
        def plan_move(self, src_xy, *a, **k):
            red_at_start = np.allclose(src_xy, start["red_cube"], atol=1)
            if red_at_start and np.allclose(state["purple_cube"], start["purple_cube"]):
                raise SkillError("every side of the block is blocked by a neighbouring block (best gap -14 mm)")
            return super().plan_move(src_xy, *a, **k)

        def pick(self, xy, *a, **k):
            self.held = min(state, key=lambda n: np.hypot(*(state[n] - np.asarray(xy))))

        def place(self, aim, yaw, on, grip=None):
            state[self.held] = np.asarray(aim, float)

    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (
        {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": xy.tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
            for n, xy in state.items()]}, None, []))
    ex = Executor(BoxedSkills((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done"
    assert np.allclose(state["red_cube"], cell_center("C2"))
    assert not np.allclose(state["purple_cube"], start["purple_cube"])        # moved out of the way
    assert [s["step"] for s in report["steps"]] == [1, "aside"] or [s["step"] for s in report["steps"]] == ["aside", 1]


def test_a_miss_carrying_a_block_from_the_bank_is_not_learned_as_arm_error(monkeypatch):
    # Run 02:45: red picked in the bank, placed 23 mm off; the retry applied that miss to the
    # re-pick on the sheet, went 23 mm from the block, and closed on nothing.
    state = {"red_cube": np.array([266.0, 45.0])}

    class BankSkills(FakeSkills):
        def pick(self, xy, *a, **k):
            self.picked_at = np.asarray(xy, float)

        def place(self, aim, yaw, on, grip=None):
            bias = (0.0, -23.0) if np.hypot(*(self.picked_at - [266.0, 45.0])) < 1 else (0.0, 0.0)
            state["red_cube"] = np.asarray(aim, float) + bias

    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (
        {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": xy.tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
            for n, xy in state.items()]}, None, []))
    sk = BankSkills((0.0, 0.0))
    ex = Executor(sk, cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    target = cell_center("C2")
    assert report["steps"][0]["ok"] and report["steps"][0]["retries"] == 1
    assert np.allclose(sk.moves[1]["src"], target + (0.0, -23.0))   # re-picked where it landed
    assert np.allclose(sk.moves[1]["aim"], target)                  # and aimed straight


def test_a_block_added_to_the_bank_during_the_run_is_found_on_a_later_look(monkeypatch):
    # Two blocks at a time in the bank, refilled while the arm works: the step waits for it.
    looks = {"n": 0}
    state = {}  # the bank is empty when the step starts; red arrives before the third look

    class Placing(FakeSkills):
        def place(self, aim, yaw, on, grip=None):
            state["red_cube"] = np.asarray(aim, float)

    def scene(frame, loc, naming="auto", bank_zone=None):
        looks["n"] += 1
        if looks["n"] == 3:
            state["red_cube"] = np.array([266.0, 50.0])
        return {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": xy.tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
            for n, xy in state.items()]}, None, []

    monkeypatch.setattr(executor, "build_scene", scene)
    monkeypatch.setattr(executor, "MISSING_WAIT_S", 0.0)
    ex = Executor(Placing((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and report["steps"][0]["retries"] == 0
    assert np.allclose(state["red_cube"], cell_center("C2"))


def test_a_big_wrist_roll_miss_fixes_its_own_retry_but_is_not_remembered(tmp_path, monkeypatch):
    # Run 04:21: three identical 22 mm misses at roll 84 and no correction, because a big-roll
    # miss was not learned at all. It must steer the retry (it repeats to ~1 mm), and nothing more:
    # the 01:00 lesson from a -77 deg pick, kept for good, pushed purple off its tower at 03:21.
    class RolledGrip(FakeSkills):
        def plan_move(self, src_xy, *a, **k):
            super().plan_move(src_xy, *a, **k)
            return {"roll": -77.0}, {"roll": -2.0}

    sk = RolledGrip((0.0, -19.0))
    ex = ScriptedExecutor(sk, start_xy=cell_center("A1"))
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    assert report["steps"][0]["ok"] and report["steps"][0]["retries"] == 1
    assert np.allclose(sk.moves[1]["aim"], cell_center("B2") + (0.0, 19.0))  # the retry aimed off by the miss
    assert 0.0 not in ex.level_err and executor.load_level_offsets() == {}  # and nothing was kept


def test_an_offset_that_makes_the_landing_worse_is_replaced(monkeypatch):
    ex = ScriptedExecutor(FakeSkills((0.0, 2.0)), start_xy=cell_center("A1"))
    ex.level_err[0.0] = np.array([0.0, 19.0])  # a stale lesson: aim 19 mm off
    ex.run([{"object": "red", "goal": {"type": "cell", "cell": "B2"}}])
    assert np.allclose(ex.level_err[0.0], (0.0, 2.0))  # this landing's real 2 mm error, not an average


def test_jaws_closing_on_nothing_looks_again_and_picks_where_the_block_is(monkeypatch):
    # One empty grab used to end the task. Here the jaws nudged the block 15 mm; the retry
    # must pick it up where the camera sees it afterwards.
    from robot.skills import SkillError
    state = {"red_cube": np.array([266.0, 50.0])}

    class PushySkills(FakeSkills):
        grabs = 0

        def pick(self, xy, *a, **k):
            PushySkills.grabs += 1
            if PushySkills.grabs == 1:
                state["red_cube"] = state["red_cube"] + (15.0, 0.0)
                raise SkillError("grasp failed: jaws closed on nothing")

        def place(self, aim, yaw, on, grip=None):
            state["red_cube"] = np.asarray(aim, float)

    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (
        {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": xy.tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
            for n, xy in state.items()]}, None, []))
    sk = PushySkills((0.0, 0.0))
    ex = Executor(sk, cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and report["steps"][0]["grasp_retries"] == 1
    assert np.allclose(sk.moves[1]["src"], (281.0, 50.0))   # picked where it was seen after the push
    assert np.allclose(state["red_cube"], cell_center("C2"))


def test_a_block_behind_a_tower_is_placed_without_taking_the_tower_apart(monkeypatch):
    # Run 04:48: purple (on red at B2) had landed 9 mm toward C2, so yellow -> C2 saw it as "too
    # close", lifted it off, and set it down into its own tower (red, hidden, "wasn't there").
    state = {"red_cube": np.array(cell_center("B2")), "purple_cube": np.array(cell_center("B2")) + (-4.0, 9.0),
             "yellow_cube": np.array([266.0, 50.0])}
    stacked = {"purple_cube"}

    class Placing(FakeSkills):
        def pick(self, xy, *a, **k):
            self.held = min(state, key=lambda n: np.hypot(*(state[n] - np.asarray(xy))))

        def place(self, aim, yaw, on, grip=None):
            state[self.held] = np.asarray(aim, float)

    def scene(frame, loc, naming="auto", bank_zone=None):
        seen = [n for n in state if n != "red_cube"]  # red is under purple: the camera can't see it
        return {"sheet": {"camera_xyz_mm": None}, "objects": [
            {"name": n, "center_mm": state[n].tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
            for n in seen]}, None, []

    monkeypatch.setattr(executor, "build_scene", scene)
    sk = Placing((0.0, 0.0))
    ex = Executor(sk, cam=SceneCam(), loc=None, log=lambda *a: None)
    ex.under["purple_cube"] = "red_cube"                      # as the earlier "purple on red" step left it
    ex.last_xy["red_cube"] = cell_center("B2").tolist()
    report = ex.run([{"object": "yellow", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and [s["step"] for s in report["steps"]] == [1]   # no "aside"
    assert np.allclose(state["yellow_cube"], cell_center("C2"))
    assert np.allclose(state["purple_cube"], cell_center("B2") + (-4.0, 9.0))           # the tower stands


def _scene_of(state, hide=()):
    return {"sheet": {"camera_xyz_mm": None}, "objects": [
        {"name": n, "center_mm": np.asarray(xy, float).tolist(), "yaw_deg": 0.0, "height_mm": 48.0, "score": 0.9}
        for n, xy in state.items() if n not in hide]}


def test_a_hidden_sheet_gets_the_arm_parked_and_more_looks(monkeypatch):
    looks = {"n": 0}
    state = {"red_cube": np.array([266.0, 50.0])}

    class Placing(FakeSkills):
        parks = 0

        def park(self, **k):
            Placing.parks += 1

        def place(self, aim, yaw, on, grip=None):
            state["red_cube"] = np.asarray(aim, float)

    def scene(frame, loc, naming="auto", bank_zone=None):
        looks["n"] += 1
        return (None, None, []) if looks["n"] <= 2 else (_scene_of(state), None, [])  # tags hidden at first

    monkeypatch.setattr(executor, "build_scene", scene)
    monkeypatch.setattr(executor, "MISSING_WAIT_S", 0.0)
    ex = Executor(Placing((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and Placing.parks >= 2


def test_a_reference_block_hidden_for_a_moment_is_waited_for(monkeypatch):
    looks = {"n": 0}
    state = {"red_cube": np.array(cell_center("B2")), "purple_cube": np.array([266.0, 50.0])}

    class Placing(FakeSkills):
        def place(self, aim, yaw, on, grip=None):
            state["purple_cube"] = np.asarray(aim, float)

    def scene(frame, loc, naming="auto", bank_zone=None):
        looks["n"] += 1
        return _scene_of(state, hide=("red_cube",) if looks["n"] <= 2 else ()), None, []

    monkeypatch.setattr(executor, "build_scene", scene)
    monkeypatch.setattr(executor, "MISSING_WAIT_S", 0.0)
    ex = Executor(Placing((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "purple", "goal": {"type": "on_top", "reference": "red"}}])
    assert report["result"] == "done" and np.allclose(state["purple_cube"], cell_center("B2"))


def test_a_slip_while_lifting_is_retried_from_a_fresh_look(monkeypatch):
    from robot.skills import SkillError
    state = {"red_cube": np.array([266.0, 50.0])}

    class Slippy(FakeSkills):
        grabs = 0

        def pick(self, xy, *a, **k):
            Slippy.grabs += 1
            if Slippy.grabs == 1:
                state["red_cube"] = state["red_cube"] + (0.0, 20.0)  # dropped a little further on
                raise SkillError("cube slipped out while lifting")

        def place(self, aim, yaw, on, grip=None):
            state["red_cube"] = np.asarray(aim, float)

    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (_scene_of(state), None, []))
    sk = Slippy((0.0, 0.0))
    ex = Executor(sk, cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and report["steps"][0]["grasp_retries"] == 1
    assert np.allclose(sk.moves[1]["src"], (266.0, 70.0))


def test_a_place_that_hits_something_puts_the_block_back_and_retries(monkeypatch):
    from robot.skills import SkillError
    state = {"red_cube": np.array([266.0, 50.0])}

    class Bumpy(FakeSkills):
        places = 0
        holding = None

        def pick(self, xy, *a, **k):
            self.holding, self.picked = "red_cube", np.asarray(xy, float)

        def put_back(self):
            state["red_cube"], self.holding = self.picked, None

        def place(self, aim, yaw, on, grip=None):
            Bumpy.places += 1
            if Bumpy.places == 1:
                raise SkillError("blocked: tips 16 mm high, 2 mm sideways; backed off")
            state["red_cube"], self.holding = np.asarray(aim, float), None

    monkeypatch.setattr(executor, "build_scene", lambda frame, loc, naming="auto", bank_zone=None: (_scene_of(state), None, []))
    monkeypatch.setattr(executor, "MISSING_WAIT_S", 0.0)
    ex = Executor(Bumpy((0.0, 0.0)), cam=SceneCam(), loc=None, log=lambda *a: None)
    report = ex.run([{"object": "red", "goal": {"type": "cell", "cell": "C2"}}])
    assert report["result"] == "done" and report["steps"][0]["retries"] == 1
    assert np.allclose(state["red_cube"], cell_center("C2"))
