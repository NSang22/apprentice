"""The synthetic journals must use the same event data keys as the real brain.

tests/fixtures/brain_journal_{human,robot}.jsonl are the brain's own journals, recorded by
    python -m analytics.record_fixtures
Re-run it whenever the brain's journal changes. Never edit the fixtures by hand.
"""

import json
from pathlib import Path

from analytics.fake_journal import generate
from brain.settings import DEMO_LESSON
from shared.schemas import Lesson

FIXTURES = Path(__file__).parent / "fixtures"


def key_sets(lines):
    """{(mode, kind, cmd): {frozenset of data keys, ...}} over every complete line."""
    out = {}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # the one cut-off synthetic line
        data = event["data"]
        group = (event["mode"], event["kind"], data.get("cmd"))
        out.setdefault(group, set()).add(frozenset(data) - {"synthetic"})
    return out


def keys_by_kind(sets):
    """{(mode, kind): every data key written for that kind in that mode}."""
    out = {}
    for (mode, kind, _), shapes in sets.items():
        out.setdefault((mode, kind), set()).update(*shapes)
    return out


def test_generator_and_brain_write_the_same_keys():
    brain = key_sets(l for p in FIXTURES.glob("brain_journal_*.jsonl") for l in p.read_text().splitlines())
    lesson = Lesson.from_dict(json.loads(DEMO_LESSON.read_text()))
    ours = key_sets(l for text in generate(lesson, seed=7).values() for l in text.splitlines())
    # Both ways, per kind and per mode: the generator writes no kind or key the brain doesn't,
    # and misses none the brain writes (a new brain key fails here until the fixtures are re-recorded).
    assert keys_by_kind(ours) == keys_by_kind(brain)
    # And each exact shape the brain writes shows up in the generator's output.
    for group, sets in brain.items():
        assert sets <= ours[group], f"{group}: brain {sorted(map(sorted, sets))} vs ours {sorted(map(sorted, ours[group]))}"
