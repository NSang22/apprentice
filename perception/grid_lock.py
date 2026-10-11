"""Grid lock: one trusted view of the sheet, kept for the whole session.

The camera and the sheet are taped down, so the image-to-sheet homography never has to change
during a session. The Playskool blocks are big enough that a block in a corner cell always
covers part of an AprilTag, so fitting the homography fresh every frame fails ("GRID NOT
FOUND") exactly when the board is in use. Instead:

  lock     With the board empty and all 4 tags in view, fit the homography once and save it
           with each tag's centre in pixels (perception/calib/grid_lock.json), plus the empty
           board warped to the sheet (perception/calib/empty_board.png, see perception/change.py).
  confirm  At start-up a saved lock counts only after MIN_CONFIRM_TAGS tags are seen where the
           lock predicts them.
  check    Every frame, each visible tag is compared with its locked centre. Hidden tags are
           simply not compared (a block on a tag is normal, never a hand). Any visible tag more
           than TAG_TOLERANCE_PX off means the camera or sheet moved: the grid is lost until all
           4 tags are visible on an empty board again and the lock is refit.

Pure functions over tag centres, plus save and load. No camera here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robot.perception.sheet import SheetLocalizer, SheetPose

CALIB_DIR = Path(__file__).resolve().parent / "calib"
LOCK_PATH = CALIB_DIR / "grid_lock.json"
TAG_IDS = (0, 1, 2, 3)
TAG_TOLERANCE_PX = 8.0   # a visible tag farther than this from its locked centre: the grid moved
MIN_CONFIRM_TAGS = 2     # tags that must agree to trust the lock


@dataclass
class GridLock:
    H: np.ndarray                          # image px -> sheet mm, table plane
    tag_centers: dict[int, tuple[float, float]]  # locked centre of each tag, image px
    camera_xyz: Optional[np.ndarray] = None  # camera centre in sheet mm (parallax for block tops)
    frame_size: tuple[int, int] = (0, 0)   # (width, height) the lock was taken at
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def pose(self) -> SheetPose:
        return SheetPose(self.H, sorted(self.tag_centers), 0.0, self.camera_xyz)

    def to_dict(self) -> dict:
        return {
            "H": np.asarray(self.H).tolist(),
            "tag_centers": {str(k): [round(v[0], 2), round(v[1], 2)] for k, v in self.tag_centers.items()},
            "camera_xyz": None if self.camera_xyz is None else np.asarray(self.camera_xyz).tolist(),
            "frame_size": list(self.frame_size),
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridLock":
        return cls(
            H=np.float64(d["H"]),
            tag_centers={int(k): (float(v[0]), float(v[1])) for k, v in d["tag_centers"].items()},
            camera_xyz=None if d.get("camera_xyz") is None else np.float64(d["camera_xyz"]),
            frame_size=tuple(d.get("frame_size", (0, 0))),
            created=d.get("created", ""),
        )


@dataclass(frozen=True)
class TagCheck:
    status: str                  # "ok": >= MIN_CONFIRM_TAGS visible, all in place; "moved"; "few": too few to tell
    visible: int                 # tags detected this frame
    max_error_px: float = 0.0    # largest offset of a visible tag from its locked centre
    moved: tuple[int, ...] = ()  # ids of the tags that moved


def tag_centers(found: dict[int, np.ndarray]) -> dict[int, tuple[float, float]]:
    """Detected tag corners (id -> 4x2 px) to centres."""
    out = {}
    for tag, corners in found.items():
        c = np.asarray(corners, np.float64).reshape(-1, 2).mean(axis=0)
        out[int(tag)] = (float(c[0]), float(c[1]))
    return out


def check_tags(locked: dict[int, tuple[float, float]], visible: dict[int, tuple[float, float]],
               tolerance_px: float = TAG_TOLERANCE_PX, min_tags: int = MIN_CONFIRM_TAGS) -> TagCheck:
    """Compare this frame's visible tag centres with the locked ones. A tag that is not visible
    is not evidence of anything (a block or a hand may be on it)."""
    errors = {t: float(np.hypot(v[0] - locked[t][0], v[1] - locked[t][1])) for t, v in visible.items() if t in locked}
    moved = tuple(sorted(t for t, e in errors.items() if e > tolerance_px))
    worst = max(errors.values(), default=0.0)
    if moved:
        return TagCheck("moved", len(errors), worst, moved)
    if len(errors) < min_tags:
        return TagCheck("few", len(errors), worst)
    return TagCheck("ok", len(errors), worst)


def lock_from_frame(frame_bgr: np.ndarray, localizer: SheetLocalizer) -> tuple[Optional[GridLock], str]:
    """A new lock from a frame that shows all 4 tags, or (None, why not)."""
    found = localizer.detect_tags(frame_bgr)
    missing = [t for t in TAG_IDS if t not in found]
    if missing:
        return None, f"tags {missing} not visible: clear the board and keep hands off the corners"
    pose = localizer.locate(frame_bgr)
    if pose is None:
        return None, "could not fit the sheet"
    h, w = frame_bgr.shape[:2]
    return GridLock(np.float64(pose.H), tag_centers(found), pose.camera_xyz, (w, h)), "ok"


def save_lock(lock: GridLock, path: Path = LOCK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock.to_dict(), indent=1), encoding="utf-8")
    tmp.replace(path)


def load_lock(path: Path = LOCK_PATH) -> Optional[GridLock]:
    if not path.exists():
        return None
    return GridLock.from_dict(json.loads(path.read_text(encoding="utf-8")))
