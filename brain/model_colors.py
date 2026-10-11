"""Ask the vision model the color of one block crop: the extractor's last color source, offline only.

Replaced gemini_colors.py on 2026-10-11. brain.extractor calls this after a recording, never in a
live session (AGENTS.md rule 5), and only for a step no other source could name. The answer is
closed: one of the shared COLORS or "none". The extractor decides whether to accept it; nothing here
writes the robot's color memory.

The model, key and retries are robot/perception/openai_vision.py's (OPENAI_API_KEY, OPENAI_MODEL,
default gpt-5.6-sol). The key goes in the Authorization header only and is never printed.
"""

from __future__ import annotations

import json
from typing import Optional

from robot.perception import openai_vision
from robot.perception.openai_vision import ModelUnavailable  # noqa: F401  (raised to the extractor)
from shared.config import COLORS

NONE = "none"
PROMPT = (
    "This is a close-up photo, taken from above, of one toy block on a printed sheet (a hand or a "
    "neighbouring block may show at the edges). What is the main color of the block in the middle? "
    f"Answer with exactly one word from this list: {', '.join(COLORS)}, or {NONE} if it is not clearly "
    'one of them. Ignore glare, shadows and small printed patterns. Reply with JSON only, like {"color": "red"}.'
)


def api_key() -> Optional[str]:
    return openai_vision.api_key()


def parse_answer(text: str) -> str:
    """The closed answer: one of COLORS, or "none" for anything else."""
    try:
        color = str(json.loads(text).get("color", "")).strip().lower()
    except (ValueError, AttributeError):
        return NONE
    return color if color in COLORS else NONE


def ask_color(jpeg: bytes, key: Optional[str] = None) -> dict:
    """{"answer": a COLORS name or "none", "model": ..., "raw": the reply text} for one crop."""
    r = openai_vision.ask_json(PROMPT, [jpeg])
    return {"answer": parse_answer(r["raw"]), "model": r["model"], "raw": r["raw"]}
