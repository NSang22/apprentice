"""Load session journals, lessons, and robot task reports into flat tables for metrics.

Offline only, standard library only. Nothing here fails on bad input: unreadable lines are
skipped and counted, unknown kinds and extra fields are ignored, and missing fields read as None.

Every journal name analytics depends on lives in FIELDS and VALUES below, so a rename in
brain/ is a one-line change here.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from shared.config import LESSONS_DIR, REPO_ROOT, SESSIONS_DIR
from shared.schemas import Lesson

# Journal "step" is a 0-based index into lesson.steps (confirmed with the brain owner).
STEP_INDEX_BASE = 0

# Our name -> the key the brain writes in event["data"], per event kind.
# None marks a field the brain has not named yet; it reads as missing until filled in.
FIELDS: dict[str, dict[str, Optional[str]]] = {
    "session_start": {
        "lesson_id": "lesson_id",
        "steps": "steps",                # a count today; a list of step dicts is used directly
        "robot": "robot",                # "FakeRobot" / "HttpRobotClient" today; "none" / "fake" / "http:<url>" soon
        "grid_source": "grid_source",    # coming: "sim", "fake_feed", "fake_feed_script:<name>", "perception:<url>"
        "commit": None,                  # coming: git commit hash, name not set yet
        "dirty": None,                   # coming: working tree dirty flag, name not set yet
        "hints_file_hash": None,         # coming: sha256 of the hints file the session loaded, name not set yet
        "synthetic": "synthetic",        # analytics/fake_journal.py only
    },
    "verdict": {
        "step": "step",
        "verdict": "verdict",
        "fault_cell": "fault_cell",
        "fault_level": "fault_level",
        "expected_cell": "expected_cell",
        "expected_level": "expected_level",
        "hint_level": "hint_level",      # human mode: hint level showing when the board was checked
        "retry": "retry",                # robot mode: retries already made on this step
    },
    "hint": {"step": "step", "hint_level": "hint_level"},
    "step_advance": {"from_step": "from_step", "to_step": "to_step", "first_try": "first_try"},
    "robot_cmd": {"step": "step", "cmd": "cmd"},
    "robot_result": {"step": "step", "cmd": "cmd", "ok": "ok", "error": "error"},
    "gate": {"gate": "gate"},
    "session_end": {"outcome": "outcome", "reason": "reason", "step": "step"},
}

VALUES = {
    "pass": "PASS",
    "waiting": "WAITING",
    "missing": "MISSING",
    "place_cmd": "place",
    "fake_robots": ("FakeRobot", "fake"),
    "perception_prefix": "perception:",
}

PROVENANCES = ("real", "unverified", "simulation", "synthetic")

DEFAULT_LESSON_DIRS = (LESSONS_DIR, REPO_ROOT / "tests" / "fixtures")
DEFAULT_TASKS_DIR = REPO_ROOT / "robot" / "outputs" / "tasks"


def field_of(event: dict[str, Any], name: str) -> Any:
    key = FIELDS.get(event.get("kind"), {}).get(name)
    return None if key is None else event["data"].get(key)


# ---------------------------------------------------------------- tables


@dataclass
class SessionRow:
    session: str
    mode: str
    provenance: str
    path: str
    lesson_id: Optional[str]
    n_steps: Optional[int]
    grid_source: Optional[str]
    robot: Optional[str]
    commit: Optional[str]
    dirty: Optional[bool]
    hints_file_hash: Optional[str]
    gate_passed: Optional[bool]  # None when the lesson had no gate
    t_start: float
    t_end: Optional[float]       # None when the journal has no session_end (cut off or crashed)
    outcome: Optional[str]
    end_reason: Optional[str]
    steps_completed: int
    skipped_lines: int


@dataclass
class Attempt:
    """One graded check of a lesson step by the student."""

    session: str
    mode: str
    provenance: str
    lesson_id: Optional[str]
    step: int
    step_color: Optional[str]   # the lesson step being worked on
    step_cell: Optional[str]
    step_level: Optional[int]
    verdict: str
    fault_cell: Optional[str]   # straight from the verdict
    fault_level: Optional[int]
    expected_cell: Optional[str]  # the slot that needs a cube; differs from step_cell for a knocked older cube
    expected_level: Optional[int]
    attempt_no: int             # 1, 2, ... within this session and step
    hint_level_before: Optional[int]  # human mode: hint showing at check time
    retry: Optional[int]        # robot mode: retries already made
    t: float
    seconds_since_step_start: float

    @property
    def passed(self) -> bool:
        return self.verdict == VALUES["pass"]


@dataclass
class HintShown:
    session: str
    provenance: str
    lesson_id: Optional[str]
    step: Optional[int]
    hint_level: Optional[int]
    t: float


@dataclass
class RobotCmd:
    session: str
    mode: str
    provenance: str
    step: Optional[int]
    cmd: Optional[str]
    ok: Optional[bool]  # from the matching robot_result; None if it never came back
    error: Optional[str]
    t: float


@dataclass
class TaskStep:
    """One step of a robot/outputs/tasks/*.json report from robot/executor.py."""

    file: str
    started: Optional[float]
    result: Optional[str]
    step: Optional[int]
    description: Optional[str]
    cell: Optional[str]  # robot naming: letter = row (A nearest the arm), number = column
    seconds: Optional[float]
    error_mm: Optional[float]
    ok: Optional[bool]   # None when the report never recorded it


@dataclass
class Tables:
    sessions: list[SessionRow] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    hints: list[HintShown] = field(default_factory=list)
    robot_cmds: list[RobotCmd] = field(default_factory=list)
    task_steps: list[TaskStep] = field(default_factory=list)
    skipped_lines: int = 0
    never_started: int = 0        # journal files with no session_start (only board lines)
    bad_lessons: int = 0
    unknown_lessons: list[str] = field(default_factory=list)

    def only(self, provenances: Iterable[str]) -> Tables:
        """The same tables restricted to sessions of the given provenances."""
        keep = set(provenances)
        return replace(
            self,
            sessions=[s for s in self.sessions if s.provenance in keep],
            attempts=[a for a in self.attempts if a.provenance in keep],
            hints=[h for h in self.hints if h.provenance in keep],
            robot_cmds=[r for r in self.robot_cmds if r.provenance in keep],
        )


# ---------------------------------------------------------------- journals


def read_journal(path: Path) -> tuple[list[dict], int]:
    """(events, count of skipped lines). A line is skipped unless it is a JSON object with a
    string kind, a numeric t, and an object for data. A cut-off last line is just skipped."""
    events, skipped = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            ok = (
                isinstance(event, dict)
                and isinstance(event.get("kind"), str)
                and isinstance(event.get("t"), (int, float))
                and isinstance(event.get("data", {}), dict)
            )
            if not ok:
                skipped += 1
                continue
            event.setdefault("data", {})
            events.append(event)
    return events, skipped


def _session_start(events: list[dict]) -> Optional[dict]:
    return next((e for e in events if e["kind"] == "session_start"), None)


def _after_start(events: list[dict]) -> list[dict]:
    """Events from session_start on. In the "ready" phase the brain journals boards first."""
    for i, e in enumerate(events):
        if e["kind"] == "session_start":
            return events[i:]
    return []


def provenance(start: dict) -> str:
    if field_of(start, "synthetic") is True:
        return "synthetic"
    robot = field_of(start, "robot")
    grid = field_of(start, "grid_source")
    if robot in VALUES["fake_robots"]:
        return "simulation"
    if not isinstance(grid, str):
        return "unverified"
    return "real" if grid.startswith(VALUES["perception_prefix"]) else "simulation"


def lesson_steps(start: dict, lessons: dict[str, Lesson]) -> Optional[list[dict]]:
    """The lesson's steps as dicts: from session_start if it carries the list, else by lesson_id."""
    steps = field_of(start, "steps")
    if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
        return steps
    lesson = lessons.get(field_of(start, "lesson_id"))
    return None if lesson is None else [s.to_dict() for s in lesson.steps]


def _step_info(steps: Optional[list[dict]], step: int) -> tuple[Any, Any, Any]:
    i = step - STEP_INDEX_BASE
    if steps is None or not 0 <= i < len(steps):
        return None, None, None
    s = steps[i]
    return s.get("color"), s.get("cell"), s.get("level")


def build_attempts(events: list[dict], lessons: dict[str, Lesson]) -> list[Attempt]:
    """One Attempt per graded check in one session's events."""
    events = _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, mode, prov = start.get("session", ""), start.get("mode", ""), provenance(start)
    lesson_id = field_of(start, "lesson_id")
    steps = lesson_steps(start, lessons)

    step_start_t = start["t"]
    attempt_no: dict[int, int] = defaultdict(int)
    placed: set[int] = set()  # steps with a place command already issued (robot mode)
    out = []
    for e in events[1:]:
        kind = e["kind"]
        if kind == "step_advance":
            step_start_t = e["t"]
        elif kind == "robot_cmd" and field_of(e, "cmd") == VALUES["place_cmd"]:
            placed.add(field_of(e, "step"))
        elif kind == "verdict":
            verdict, step = field_of(e, "verdict"), field_of(e, "step")
            if not isinstance(verdict, str) or not isinstance(step, int) or verdict == VALUES["waiting"]:
                continue
            if verdict == VALUES["missing"] and field_of(e, "fault_cell") is None:
                # "Still placing": the board is exactly as before this step. Human mode never counts
                # it (brain/session.py:63-67, is_mistake). Robot mode counts it only once a place for
                # this step was issued, i.e. the place failed (brain/session.py:274-282, placed_current).
                if mode != "robot" or step not in placed:
                    continue
            attempt_no[step] += 1
            color, cell, level = _step_info(steps, step)
            out.append(Attempt(
                session=sid, mode=mode, provenance=prov, lesson_id=lesson_id, step=step,
                step_color=color, step_cell=cell, step_level=level, verdict=verdict,
                fault_cell=field_of(e, "fault_cell"), fault_level=field_of(e, "fault_level"),
                expected_cell=field_of(e, "expected_cell"), expected_level=field_of(e, "expected_level"),
                attempt_no=attempt_no[step],
                hint_level_before=field_of(e, "hint_level"), retry=field_of(e, "retry"),
                t=e["t"], seconds_since_step_start=round(e["t"] - step_start_t, 3),
            ))
    return out


def build_hints(events: list[dict]) -> list[HintShown]:
    events = _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, prov, lesson_id = start.get("session", ""), provenance(start), field_of(start, "lesson_id")
    return [
        HintShown(sid, prov, lesson_id, field_of(e, "step"), field_of(e, "hint_level"), e["t"])
        for e in events if e["kind"] == "hint"
    ]


def build_robot_cmds(events: list[dict]) -> list[RobotCmd]:
    """Each robot_cmd, with ok/error from the next robot_result for the same command."""
    events = _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, mode, prov = start.get("session", ""), start.get("mode", ""), provenance(start)
    out: list[RobotCmd] = []
    for e in events:
        if e["kind"] == "robot_cmd":
            out.append(RobotCmd(sid, mode, prov, field_of(e, "step"), field_of(e, "cmd"), None, None, e["t"]))
        elif e["kind"] == "robot_result":
            open_cmd = next((r for r in reversed(out) if r.ok is None and r.cmd == field_of(e, "cmd")), None)
            if open_cmd is not None:
                ok = field_of(e, "ok")
                open_cmd.ok = ok if isinstance(ok, bool) else None
                open_cmd.error = field_of(e, "error")
    return out


def build_session(events: list[dict], lessons: dict[str, Lesson], path: str = "", skipped: int = 0) -> Optional[SessionRow]:
    events = _after_start(events)
    if not events:
        return None
    start = events[0]
    steps = lesson_steps(start, lessons)
    n_steps = field_of(start, "steps")
    gate_event = next((e for e in events if e["kind"] == "gate"), None)
    gate = None if gate_event is None else field_of(gate_event, "gate")
    end = next((e for e in events if e["kind"] == "session_end"), None)
    dirty = field_of(start, "dirty")
    return SessionRow(
        session=start.get("session", ""),
        mode=start.get("mode", ""),
        provenance=provenance(start),
        path=path,
        lesson_id=field_of(start, "lesson_id"),
        n_steps=len(steps) if steps is not None else (n_steps if isinstance(n_steps, int) else None),
        grid_source=field_of(start, "grid_source"),
        robot=field_of(start, "robot"),
        commit=field_of(start, "commit"),
        dirty=dirty if isinstance(dirty, bool) else None,
        hints_file_hash=field_of(start, "hints_file_hash"),
        gate_passed=bool(gate.get("passed")) if isinstance(gate, dict) else None,
        t_start=start["t"],
        t_end=None if end is None else end["t"],
        outcome=None if end is None else field_of(end, "outcome"),
        end_reason=None if end is None else field_of(end, "reason"),
        steps_completed=sum(1 for e in events if e["kind"] == "step_advance"),
        skipped_lines=skipped,
    )


# ---------------------------------------------------------------- lessons and robot reports


def load_lessons(dirs: Iterable[Path]) -> tuple[dict[str, Lesson], int]:
    """{lesson_id: Lesson} from every *.json in dirs (earlier dirs win), and a count of bad files.
    Hint side files (*.hints.json) are not lessons and are skipped."""
    lessons: dict[str, Lesson] = {}
    bad = 0
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            if path.name.endswith(".hints.json"):
                continue
            try:
                lesson = Lesson.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError, KeyError, OSError):
                bad += 1
                continue
            lessons.setdefault(lesson.lesson_id, lesson)
    return lessons, bad


CELL_GOAL = re.compile(r"-> cell ([A-C][1-3])$")


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def read_task_reports(tasks_dir: Path) -> list[TaskStep]:
    """Steps from robot/outputs/tasks/*.json. Missing fields read as None; bad files are skipped."""
    out = []
    for path in sorted(Path(tasks_dir).glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(report, dict):
            continue
        for s in report.get("steps") or []:
            if not isinstance(s, dict):
                continue
            desc = s.get("description") if isinstance(s.get("description"), str) else None
            m = CELL_GOAL.search(desc) if desc else None
            ok = s.get("ok")
            out.append(TaskStep(
                file=path.name,
                started=_num(report.get("started")),
                result=report.get("result") if isinstance(report.get("result"), str) else None,
                step=s.get("step") if isinstance(s.get("step"), int) else None,
                description=desc,
                cell=m.group(1) if m else None,
                seconds=_num(s.get("seconds")),
                error_mm=_num(s.get("error_mm")),
                ok=ok if isinstance(ok, bool) else None,
            ))
    return out


# ---------------------------------------------------------------- everything


def load_all(
    sessions_dir: Path = SESSIONS_DIR,
    lesson_dirs: Iterable[Path] = DEFAULT_LESSON_DIRS,
    tasks_dir: Optional[Path] = None,
) -> Tables:
    """Every session of every provenance. Callers filter with Tables.only()."""
    lessons, bad_lessons = load_lessons(lesson_dirs)
    t = Tables(bad_lessons=bad_lessons)
    unknown: set[str] = set()
    for path in sorted(Path(sessions_dir).glob("*.jsonl")):
        events, skipped = read_journal(path)
        t.skipped_lines += skipped
        row = build_session(events, lessons, path.name, skipped)
        if row is None:
            t.never_started += 1
            continue
        if row.lesson_id is not None and lesson_steps(_session_start(events), lessons) is None:
            unknown.add(row.lesson_id)
        t.sessions.append(row)
        t.attempts += build_attempts(events, lessons)
        t.hints += build_hints(events)
        t.robot_cmds += build_robot_cmds(events)
    if tasks_dir is not None:
        t.task_steps = read_task_reports(tasks_dir)
    t.unknown_lessons = sorted(unknown)
    return t
