"""OpenAI for every vision-model job in the project. Replaced Gemini on 2026-10-11 (rate limits and
unstable answers during the hackathon).

Used by:
  robot/perception/vision_namer.py   names block colors the robot's color memory is unsure of
  brain/model_colors.py              the extractor's last color source, one block crop at a time
  buddy/eyes_worker.py               what is in the robot's cube bank; the narration's placements

Key: OPENAI_API_KEY from the environment, or from the repo's .env file (gitignored). It goes in the
Authorization header only and is never printed or logged.
Model: OPENAI_MODEL, else DEFAULT_MODEL. Measured on the real bank check (one teal block in the green
box, 2 runs each): gpt-5.6-sol 1.3/1.8 s and right; gpt-6-sol 2.6/2.5 s; gpt-6.1-sol 4.2/1.7 s;
gpt-5.6-terra and gpt-4.1-mini answered sky blue. Transcription: OPENAI_TRANSCRIBE_MODEL, else
DEFAULT_TRANSCRIBE.

Standard library only, so robot.perception stays importable with just OpenCV and numpy. A timeout,
a 429 or a 5xx answer is retried RETRIES times, RETRY_WAIT_S apart; anything else raises
ModelUnavailable with a message that never contains the key.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

API = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_TRANSCRIBE = "gpt-4o-transcribe"
TIMEOUT_S = 30
RETRIES, RETRY_WAIT_S = 2, 1.5
REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelUnavailable(RuntimeError):
    """No answer: no key, or the request failed. Messages never contain the key."""


def api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    env = REPO_ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*OPENAI_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(1)
    return None


def available() -> bool:
    return api_key() is not None


def model() -> str:
    return os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def transcribe_model() -> str:
    return os.environ.get("OPENAI_TRANSCRIBE_MODEL") or DEFAULT_TRANSCRIBE


def _send(req_factory, what: str) -> dict:
    """One request (req_factory builds a fresh urllib Request), retried on timeouts, 429 and 5xx."""
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req_factory(), timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                detail = ""
            failure, retry = f"OpenAI answered HTTP {e.code}{': ' + detail[:160] if detail else ''}", e.code == 429 or e.code >= 500
            if e.code == 400 and "reasoning" in detail.lower():
                raise _NoReasoning(detail) from None
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            failure, retry = f"OpenAI request failed: {type(reason).__name__}", True
        except ValueError as e:
            raise ModelUnavailable(f"{what}: bad reply ({type(e).__name__})") from None
        if not retry or attempt == RETRIES:
            raise ModelUnavailable(failure + (f" (after {attempt + 1} tries)" if attempt else "")) from None
        time.sleep(RETRY_WAIT_S)
    raise ModelUnavailable("unreachable")


class _NoReasoning(ModelUnavailable):
    """The model refused the reasoning setting: ask again without it."""


def ask_json(prompt: str, images: tuple[bytes, ...] | list[bytes] = (), effort: Optional[str] = "low") -> dict:
    """One question with optional JPEG images; the prompt must ask for JSON.
    Returns {"answer": the parsed JSON, "model", "seconds", "raw"}."""
    key = api_key()
    if not key:
        raise ModelUnavailable("no OPENAI_API_KEY (environment or repo .env)")
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(j).decode()} for j in images]
    name = model()

    def body(with_effort: bool) -> bytes:
        b = {"model": name, "input": [{"role": "user", "content": content}], "text": {"format": {"type": "json_object"}}}
        if with_effort and effort:
            b["reasoning"] = {"effort": effort}
        return json.dumps(b).encode("utf-8")

    def request(with_effort: bool):
        return lambda: urllib.request.Request(f"{API}/responses", data=body(with_effort), method="POST",
                                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        reply = _send(request(True), "OpenAI")
    except _NoReasoning:
        reply = _send(request(False), "OpenAI")
    text = "".join(c.get("text", "") for o in reply.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if c.get("type") == "output_text")
    try:
        answer = json.loads(text)
    except ValueError:
        raise ModelUnavailable(f"{name} did not answer with JSON") from None
    return {"answer": answer, "model": name, "seconds": round(time.monotonic() - started, 2), "raw": text}


def transcribe(audio: Path, prompt: Optional[str] = None) -> dict:
    """Speech to text for a recording (any format the API takes: ogg/opus, wav, mp3, webm).
    Returns {"text", "model", "seconds"}."""
    key = api_key()
    if not key:
        raise ModelUnavailable("no OPENAI_API_KEY (environment or repo .env)")
    audio = Path(audio)
    boundary = uuid.uuid4().hex
    fields = {"model": transcribe_model(), "response_format": "json"}
    if prompt:
        fields["prompt"] = prompt
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items()]
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio.name}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + audio.read_bytes() + b"\r\n")
    data = b"".join(parts) + f"--{boundary}--\r\n".encode()
    started = time.monotonic()
    reply = _send(lambda: urllib.request.Request(f"{API}/audio/transcriptions", data=data, method="POST",
                                                 headers={"Authorization": f"Bearer {key}",
                                                          "Content-Type": f"multipart/form-data; boundary={boundary}"}),
                  "OpenAI transcription")
    return {"text": str(reply.get("text", "")).strip(), "model": transcribe_model(),
            "seconds": round(time.monotonic() - started, 2)}
