"""The synthetic journals must use the same event data keys as the real brain.

tests/fixtures/brain_journal_{human,robot}.jsonl were written by brain.server on main:
    python -m brain.server --mode robot --robot fake --grid sim --fake-wrong-on 2
    python -m brain.fake_feed --script tests/scripts/staircase_with_mistake.json   # with:
    python -m brain.server --lesson tests/fixtures/staircase-v1.json --mode human --robot fake
Regenerate them the same way when the brain's journal changes.
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


def test_generator_writes_every_key_set_the_brain_writes():
    brain = key_sets(l for p in FIXTURES.glob("brain_journal_*.jsonl") for l in p.read_text().splitlines())
    lesson = Lesson.from_dict(json.loads(DEMO_LESSON.read_text()))
    ours = key_sets(l for text in generate(lesson, seed=7).values() for l in text.splitlines())
    for group, sets in brain.items():
        assert group in ours, f"generator never writes {group}"
        assert sets <= ours[group], f"{group}: brain {sorted(map(sorted, sets))} vs ours {sorted(map(sorted, ours[group]))}"
