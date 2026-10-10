import json
import random
from pathlib import Path

from brain.fake_feed import make_frame, parse_script
from brain.stabilizer import Stabilizer, StabilizerState, frame_usable, step
from shared.config import CELLS, MIN_CONFIDENCE, STABLE_FRAMES
from shared.schemas import GridState, empty_board

SCRIPTS_DIR = Path(__file__).parent / "scripts"

A = {**empty_board(), "A1": ["red"]}
B = {**empty_board(), "A1": ["red"], "B1": ["green"]}


def gs(board, hands=True, arm=True, conf=None, frame_id=0):
    confidence = {c: 0.95 for c in CELLS}
    confidence.update(conf or {})
    return GridState(t=0.0, board={c: list(board[c]) for c in CELLS}, arm_clear=arm, hands_clear=hands,
                     confidence=confidence, frame_id=frame_id)


def feed_all(stab, frames):
    return [b for b in (stab.feed(f) for f in frames) if b is not None]


# ---------------------------------------------------------------- basics


def test_emits_after_exactly_stable_frames():
    stab = Stabilizer()
    for _ in range(STABLE_FRAMES - 1):
        assert stab.feed(gs(A)) is None
    assert stab.feed(gs(A)) == A


def test_emits_once_while_board_stays_stable():
    assert feed_all(Stabilizer(), [gs(A)] * (STABLE_FRAMES * 5)) == [A]


def test_new_board_emits_again():
    out = feed_all(Stabilizer(), [gs(A)] * STABLE_FRAMES + [gs(B)] * STABLE_FRAMES)
    assert out == [A, B]


def test_returning_to_previous_board_emits_it_again():
    out = feed_all(Stabilizer(), [gs(A)] * STABLE_FRAMES + [gs(B)] * STABLE_FRAMES + [gs(A)] * STABLE_FRAMES)
    assert out == [A, B, A]


def test_same_board_after_hands_is_re_emitted():
    # Wrong board, hands reach in and pull back without changing it: a second settle.
    frames = [gs(A)] * STABLE_FRAMES + [gs(A, hands=False)] * 3 + [gs(A)] * STABLE_FRAMES
    assert feed_all(Stabilizer(), frames) == [A, A]


def test_same_board_after_arm_is_re_emitted():
    frames = [gs(A)] * STABLE_FRAMES + [gs(A, arm=False)] * 3 + [gs(A)] * STABLE_FRAMES
    assert feed_all(Stabilizer(), frames) == [A, A]


def test_interruption_mid_run_still_re_emits_once():
    # The interruption lands after the board went stable; the repeat comes only once it resettles.
    frames = [gs(A)] * (STABLE_FRAMES + 2) + [gs(A, hands=False)] + [gs(A)] * (STABLE_FRAMES * 3)
    assert feed_all(Stabilizer(), frames) == [A, A]


def test_jitter_without_interruption_does_not_re_emit():
    frames = [gs(A)] * STABLE_FRAMES + [gs(A, conf={"B2": 0.1})] * 3 + [gs({**A, "A1": ["unknown"]})] + [gs(A)] * STABLE_FRAMES
    assert feed_all(Stabilizer(), frames) == [A]


def test_interruption_flag_clears_after_emit():
    frames = [gs(A)] * STABLE_FRAMES + [gs(A, hands=False)] + [gs(A)] * STABLE_FRAMES + [gs(A, conf={"B2": 0.1})] + [gs(A)] * STABLE_FRAMES
    assert feed_all(Stabilizer(), frames) == [A, A]


def test_reset_re_emits_current_board():
    stab = Stabilizer()
    feed_all(stab, [gs(A)] * STABLE_FRAMES)
    stab.reset()
    assert feed_all(stab, [gs(A)] * STABLE_FRAMES) == [A]


# ---------------------------------------------------------------- violations reset the count


def interrupted(bad_frame):
    """STABLE_FRAMES - 1 good frames, one bad frame, STABLE_FRAMES - 1 good frames: never stable."""
    return [gs(A)] * (STABLE_FRAMES - 1) + [bad_frame] + [gs(A)] * (STABLE_FRAMES - 1)


def test_hands_in_frame_resets():
    stab = Stabilizer()
    assert feed_all(stab, interrupted(gs(A, hands=False))) == []
    assert stab.feed(gs(A)) == A  # one more good frame completes a fresh run


def test_arm_not_clear_resets():
    assert feed_all(Stabilizer(), interrupted(gs(A, arm=False))) == []


def test_low_confidence_resets():
    assert feed_all(Stabilizer(), interrupted(gs(A, conf={"C3": MIN_CONFIDENCE - 0.01}))) == []


def test_confidence_exactly_at_threshold_is_usable():
    assert frame_usable(gs(A, conf={"B2": MIN_CONFIDENCE}))


def test_missing_confidence_entry_counts_as_low():
    frame = gs(A)
    del frame.confidence["B2"]
    assert not frame_usable(frame)


def test_unknown_color_resets():
    unknown = {**A, "A1": ["red", "unknown"]}
    assert feed_all(Stabilizer(), interrupted(gs(unknown))) == []


def test_jittery_frames_never_settle_on_the_jitter():
    # One cell flips color every third frame: runs never reach STABLE_FRAMES (5).
    flipped = {**A, "A1": ["blue"]}
    frames = [gs(flipped if i % 3 == 2 else A) for i in range(30)]
    assert feed_all(Stabilizer(), frames) == []


def test_different_board_restarts_count():
    frames = [gs(A)] * (STABLE_FRAMES - 1) + [gs(B)] * (STABLE_FRAMES - 1)
    assert feed_all(Stabilizer(), frames) == []


# ---------------------------------------------------------------- purity


def test_step_is_pure():
    s0 = StabilizerState()
    frame = gs(A)
    s1, _ = step(s0, frame)
    assert s0 == StabilizerState()
    assert s1.count == 1
    assert frame.board == A


def test_returned_board_is_a_copy():
    stab = Stabilizer()
    board = feed_all(stab, [gs(A)] * STABLE_FRAMES)[0]
    board["C3"].append("green")
    assert stab.state.emitted == A


# ---------------------------------------------------------------- scripted feeds


def run_script(name, seed=0):
    rng, frames = random.Random(seed), []
    for seg in parse_script(json.loads((SCRIPTS_DIR / name).read_text())):
        for _ in range(round(seg.duration_s * 10)):
            frames.append(make_frame(seg.board, seg.hands_clear, seg.arm_clear, seg.jitter, len(frames), rng))
    return feed_all(Stabilizer(), frames)


def test_jitter_and_hands_script_repeats_only_after_interruptions():
    # Settles once, survives jitter without repeating, then repeats after the hands and arm segments.
    board = {**empty_board(), "A1": ["red", "blue"], "B2": ["green"], "C3": ["yellow"]}
    for seed in range(20):
        assert run_script("jitter_and_hands.json", seed) == [board] * 3, seed


def test_staircase_script_yields_each_board_once():
    out = run_script("staircase_clean.json")
    assert [sum(len(s) for s in b.values()) for b in out] == [0, 1, 2, 3, 4, 5, 6]
