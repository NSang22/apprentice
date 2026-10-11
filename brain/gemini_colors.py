"""Ask Gemini the color of one block crop: the extractor's last color source, offline only.

brain.extractor calls this after a recording, never in a live session (AGENTS.md rule 5), and only
for a step no other source could name. The answer is closed: one of the shared COLORS or "none".
The extractor decides whether to accept it; nothing here writes the robot's color memory.

Same key and model as robot/perception/gemini_namer.py: GEMINI_API_KEY from the environment or the
repo's .env file, GEMINI_MODEL if set, else the newest "flash" model the key can use. That module
sends a whole numbered frame and takes open answers, so this is a small closed variant. Standard
library only. The key goes in the x-goog-api-key header, never in a URL, and is never printed.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from shared.config import COLORS, REPO_ROOT

API = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT_S = 30
RETRIES, RETRY_WAIT_S = 2, 2.0   # a timeout or a 5xx answer is retried twice, 2 s apart
NONE = "none"
PROMPT = (
    "This is a close-up photo, taken from above, of one toy block on a printed sheet (a hand or a "
    "neighbouring block may show at the edges). What is the main color of the block in the middle? "
    f"Answer with exactly one word from this list: {', '.join(COLORS)}, or {NONE} if it is not clearly "
    'one of them. Ignore glare, shadows and small printed patterns. Reply with JSON only, like {"color": "red"}.'
)


class GeminiUnavailable(RuntimeError):
    """No answer: no key, no model, or the request failed. Messages never contain the key."""


def api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    env = REPO_ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*GEMINI_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(1)
    return None


def _request(url: str, key: str, body: Optional[dict] = None) -> dict:
    """One request, retried RETRIES times after RETRY_WAIT_S on a timeout or a 5xx answer."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            failure, retry = f"Gemini answered HTTP {e.code}", e.code >= 500
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            failure = f"Gemini request failed: {type(reason).__name__}"
            retry = isinstance(reason, TimeoutError) or isinstance(e, TimeoutError)
        except ValueError as e:
            raise GeminiUnavailable(f"Gemini request failed: {type(e).__name__}") from None
        if not retry or attempt == RETRIES:
            raise GeminiUnavailable(failure + (f" (after {attempt + 1} tries)" if attempt else "")) from None
        time.sleep(RETRY_WAIT_S)
    raise GeminiUnavailable("unreachable")


def model_name(key: str) -> str:
    """GEMINI_MODEL, else the newest flash model this key can call (as robot/perception/gemini_namer.py)."""
    if os.environ.get("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    models = _request(f"{API}/models?pageSize=200", key).get("models", [])
    flash = [m["name"].split("/", 1)[1] for m in models
             if "flash" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", [])
             and not any(x in m["name"] for x in ("lite", "image", "tts", "audio", "live", "exp", "preview"))]
    if not flash:
        raise GeminiUnavailable("no Gemini flash model available to this key")

    def version(name: str) -> tuple[int, int]:
        v = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
        return (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0)

    return max(flash, key=version)


def parse_answer(text: str) -> str:
    """The closed answer: one of COLORS, or "none" for anything else."""
    try:
        color = str(json.loads(text).get("color", "")).strip().lower()
    except (ValueError, AttributeError):
        return NONE
    return color if color in COLORS else NONE


def ask_color(jpeg: bytes, key: str) -> dict:
    """{"answer": a COLORS name or "none", "model": ..., "raw": the reply text} for one crop."""
    model = model_name(key)
    body = {
        "contents": [{"parts": [{"text": PROMPT},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    reply = _request(f"{API}/models/{model}:generateContent", key, body)
    try:
        text = reply["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise GeminiUnavailable("Gemini sent no answer") from None
    return {"answer": parse_answer(text), "model": model, "raw": text}
