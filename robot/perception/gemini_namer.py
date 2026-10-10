"""Ask Gemini what colour each segmented block is.

One request per call: the overhead frame with every segment outlined and numbered, and a JSON
reply mapping number -> colour words. Used to teach ColorMemory at the start of a session or
when a block looks unfamiliar; everyday recognition is local.

Key: GEMINI_API_KEY from the environment, or from a `.env` file at the repo root (gitignored).
Model: GEMINI_MODEL if set, else the newest available "flash" model.
"""

import base64
import json
import os
import re
import time

import cv2
import numpy as np

from .config import BLOCK_COLORS, ROOT

# `requests` is imported only when the model is actually called: robot.perception must stay
# importable with just OpenCV and numpy (perception/publish.py reuses it; see
# tests/test_perception_publish.py).

API = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT_S = 30
PROMPT = (
    "This is an overhead photo of toy blocks on a printed sheet. Each block is outlined in magenta "
    "and labelled with a white number in a black box. For every number, name the block's colour, "
    f"choosing from the colours this kit has: {', '.join(BLOCK_COLORS)}. Only if a block clearly "
    "matches none of them, use another plain lowercase colour word. Judge the block's main colour, "
    "ignoring glare, shadows and small printed patterns. Reply with JSON only, like "
    '{"1": "red", "2": "light blue"}.'
)


class NamerUnavailable(RuntimeError):
    pass


def _api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*GEMINI_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(1)
    return None


def available():
    return _api_key() is not None


_model_cache = {}


def _model(key, requests):
    if os.environ.get("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    if "name" not in _model_cache:
        r = requests.get(f"{API}/models", params={"key": key, "pageSize": 200}, timeout=TIMEOUT_S)
        r.raise_for_status()
        flash = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                 if "flash" in m["name"] and "generateContent" in m.get("supportedGenerationMethods", [])
                 and not any(x in m["name"] for x in ("lite", "image", "tts", "audio", "live", "exp", "preview"))]
        if not flash:
            raise NamerUnavailable("no Gemini flash model available to this key")

        def version(name):  # gemini-3.8-flash -> (3, 8)
            v = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
            return (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0)

        _model_cache["name"] = max(flash, key=version)
    return _model_cache["name"]


def numbered_image(frame_bgr, contours, max_width=1280):
    """The frame with each contour outlined and numbered 1..n (what the model sees)."""
    img = frame_bgr.copy()
    for i, c in enumerate(contours, 1):
        c = np.asarray(c, np.int32).reshape(-1, 1, 2)
        cv2.drawContours(img, [c], -1, (255, 0, 255), 3)
        m = cv2.moments(c)
        u, v = int(m["m10"] / max(m["m00"], 1)), int(m["m01"] / max(m["m00"], 1))
        label = str(i)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        cv2.rectangle(img, (u - w // 2 - 8, v - h // 2 - 8), (u + w // 2 + 8, v + h // 2 + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (u - w // 2, v + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])))
    return img


def name_segments(frame_bgr, contours):
    """{index (0-based): colour name} for each contour. Raises NamerUnavailable on any failure."""
    key = _api_key()
    if not key:
        raise NamerUnavailable("no GEMINI_API_KEY (environment or repo .env)")
    if not len(contours):
        return {}
    ok, jpg = cv2.imencode(".jpg", numbered_image(frame_bgr, contours), [cv2.IMWRITE_JPEG_QUALITY, 85])
    body = {
        "contents": [{"parts": [{"text": PROMPT},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    try:
        import requests
    except ImportError:
        raise NamerUnavailable("the 'requests' package isn't installed") from None
    t = time.time()
    try:
        model = _model(key, requests)
        r = requests.post(f"{API}/models/{model}:generateContent", params={"key": key}, json=body, timeout=TIMEOUT_S)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        reply = json.loads(text)
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise NamerUnavailable(f"Gemini request failed: {e}") from None
    names = {}
    for k, v in reply.items():
        if str(k).isdigit() and 1 <= int(k) <= len(contours) and isinstance(v, str) and v.strip():
            names[int(k) - 1] = v.strip().lower()
    name_segments.last = {"model": model, "seconds": round(time.time() - t, 2), "raw": reply}
    return names
