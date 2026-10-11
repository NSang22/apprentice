"""Ask the vision model (OpenAI, robot/perception/openai_vision.py) what colour each segmented block is.

Replaced gemini_namer.py on 2026-10-11 with the same interface: available(), name_segments(),
NamerUnavailable, name_segments.last. One request per call: the overhead frame with every segment
outlined and numbered, and a JSON reply mapping number -> colour words. Used to teach ColorMemory
at the start of a session or when a block looks unfamiliar; everyday recognition is local.
"""

import time

import cv2
import numpy as np

from . import openai_vision
from .config import BLOCK_COLORS

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


def available():
    return openai_vision.available()


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
    if not len(contours):
        return {}
    ok, jpg = cv2.imencode(".jpg", numbered_image(frame_bgr, contours), [cv2.IMWRITE_JPEG_QUALITY, 85])
    t = time.time()
    try:
        r = openai_vision.ask_json(PROMPT, [jpg.tobytes()])
    except openai_vision.ModelUnavailable as e:
        raise NamerUnavailable(str(e)) from None
    reply = r["answer"] if isinstance(r["answer"], dict) else {}
    names = {}
    for k, v in reply.items():
        if str(k).isdigit() and 1 <= int(k) <= len(contours) and isinstance(v, str) and v.strip():
            names[int(k) - 1] = v.strip().lower()
    name_segments.last = {"model": r["model"], "seconds": round(time.time() - t, 2), "raw": reply}
    return names
