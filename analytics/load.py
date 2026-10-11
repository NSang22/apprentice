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
        "robot": "robot",                # "none", "fake", "http:<url>", "unknown"; older journals: "FakeRobot"
        "grid_source": "grid_source",    # "sim", "fake_feed", "perception:<url>", "unknown:<url>", "unknown"
        "human_side": "human_side",      # "row1" or "row3": which edge the human stands at
        "commit": "git_commit",          # hash or "unknown"
        "dirty": "git_dirty",            # bool or "unknown"
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
    "note": {"grid_source": "grid_source"},  # the server notes the real source once the first frame arrives
    "step_advance": {"from_step": "from_step", "to_step": "to_step", "first_try": "first_try"},
    "step_back": {"from_step": "from_step", "to_step": "to_step"},  # an earlier block came off

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
    "unknown_prefix": "unknown",
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
    human_side: Optional[str] = None


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
class CameraChecks:
    """One session's stable boards, and how many had a block the camera could not read."""

    session: str
    mode: str
    provenance: str
    lesson_id: Optional[str]
    boards: int       # stable boards the brain graded (after session_start)
    unreadable: int   # WAITING verdicts with a fault_cell: the brain waited on an unreadable block


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
class LessonMismatch:
    """A session whose recorded step count differs from the lesson file its lesson_id maps to
    (the lesson changed under the same id), so its step numbers can't be read against it."""

    session: str
    path: str
    provenance: str
    lesson_id: Optional[str]
    recorded_steps: int
    lesson_steps: int


@dataclass
class Tables:
    sessions: list[SessionRow] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    hints: list[HintShown] = field(default_factory=list)
    robot_cmds: list[RobotCmd] = field(default_factory=list)
    camera: list[CameraChecks] = field(default_factory=list)
    task_steps: list[TaskStep] = field(default_factory=list)
    skipped_lines: int = 0
    never_started: int = 0        # journal files with no session_start (only board lines)
    bad_lessons: int = 0
    unknown_lessons: list[str] = field(default_factory=list)
    lesson_mismatches: list[LessonMismatch] = field(default_factory=list)  # left out of every other table

    def only(self, provenances: Iterable[str]) -> Tables:
        """The same tables restricted to sessions of the given provenances."""
        keep = set(provenances)
        return replace(
            self,
            sessions=[s for s in self.sessions if s.provenance in keep],
            attempts=[a for a in self.attempts if a.provenance in keep],
            hints=[h for h in self.hints if h.provenance in keep],
            robot_cmds=[r for r in self.robot_cmds if r.provenance in keep],
            camera=[c for c in self.camera if c.provenance in keep],
        )


# ---------------------------------------------------------------- journals


def read_journal(path: Path) -> tuple[list[dict], int]:
    """(events, count of skipped lines) for one journal file. See parse_journal_lines.
    utf-8-sig: a journal saved by a Windows editor may start with a byte order mark, which would
    otherwise turn the first line (session_start) into a skipped line."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return parse_journal_lines(f)


def parse_journal_lines(lines: Iterable[str]) -> tuple[list[dict], int]:
    """(events, count of skipped lines). A line is skipped unless it is a JSON object with a
    string kind, a numeric t, and an object for data. A cut-off last line is just skipped."""
    events, skipped = [], 0
    for line in lines:
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


def provenance(start: dict, events: Iterable[dict] = ()) -> str:
    """synthetic, simulation, real, or unverified, for the session that start opens.

    events is the whole journal: grid_source notes from the server (before or after start) count
    too, because a session started before the first frame records "unknown:<url>" and the server
    notes the real source later. No known source, or more than one, is unverified."""
    if field_of(start, "synthetic") is True:
        return "synthetic"
    if field_of(start, "robot") in VALUES["fake_robots"]:
        return "simulation"
    sources = set()
    for e in [start, *events]:
        g = field_of(e, "grid_source") if e["kind"] in ("session_start", "note") else None
        if isinstance(g, str) and not g.startswith(VALUES["unknown_prefix"]):
            sources.add(g)
    if len(sources) != 1:
        return "unverified"
    return "real" if sources.pop().startswith(VALUES["perception_prefix"]) else "simulation"


def lesson_steps(start: dict, lessons: dict[str, Lesson]) -> Optional[list[dict]]:
    """The lesson's steps as dicts: from session_start if it carries the list, else by lesson_id."""
    steps = field_of(start, "steps")
    if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
        return steps
    lesson = lessons.get(field_of(start, "lesson_id"))
    return None if lesson is None else [s.to_dict() for s in lesson.steps]


def lesson_mismatch(row: SessionRow, start: dict, lessons: dict[str, Lesson]) -> Optional[LessonMismatch]:
    """The mismatch if session_start recorded a step count and the lesson file has another."""
    recorded = field_of(start, "steps")
    steps = lesson_steps(start, lessons)
    if not isinstance(recorded, int) or isinstance(recorded, bool) or steps is None or recorded == len(steps):
        return None
    return LessonMismatch(row.session, row.path, row.provenance, row.lesson_id, recorded, len(steps))


def _step_info(steps: Optional[list[dict]], step: int) -> tuple[Any, Any, Any]:
    i = step - STEP_INDEX_BASE
    if steps is None or not 0 <= i < len(steps):
        return None, None, None
    s = steps[i]
    return s.get("color"), s.get("cell"), s.get("level")


def build_attempts(events: list[dict], lessons: dict[str, Lesson]) -> list[Attempt]:
    """One Attempt per graded check in one session's events.

    Progress follows the board, so the brain can step back: taking an earlier block off writes
    step_back, and putting it back writes a PASS and a step_advance for that step again. Only a
    step's first PASS is an attempt; a re-pass is not. Mistakes made on a revisit are attempts.
    A step's clock starts each time it becomes the current step (session_start, step_advance or
    step_back into it), so a revisit is timed from the step_back, not from an older advance."""
    all_events, events = events, _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, mode, prov = start.get("session", ""), start.get("mode", ""), provenance(start, all_events)
    lesson_id = field_of(start, "lesson_id")
    steps = lesson_steps(start, lessons)

    current_since: dict[int, float] = {0: start["t"]}  # step -> when it last became the current step
    attempt_no: dict[int, int] = defaultdict(int)
    passed: set[int] = set()  # steps whose first PASS is already counted
    placed: set[int] = set()  # robot mode: steps with a place issued since they last became current
    out = []
    for e in events[1:]:
        kind = e["kind"]
        if kind in ("step_advance", "step_back"):
            to = field_of(e, "to_step")
            if isinstance(to, int):
                current_since[to] = e["t"]
                placed.discard(to)  # the brain's placed_current resets on every progress change
        elif kind == "robot_cmd" and field_of(e, "cmd") == VALUES["place_cmd"]:
            placed.add(field_of(e, "step"))
        elif kind == "verdict":
            verdict, step = field_of(e, "verdict"), field_of(e, "step")
            if not isinstance(verdict, str) or not isinstance(step, int) or verdict == VALUES["waiting"]:
                continue
            if verdict == VALUES["pass"]:
                if step in passed:
                    continue  # a re-pass after a step_back
                passed.add(step)
            if verdict == VALUES["missing"] and field_of(e, "fault_cell") is None:
                # "Still placing": the board is exactly as before this step. Human mode never counts
                # it (brain/session.py:84-88, is_mistake). Robot mode counts it only once a place for
                # this step was issued, i.e. the place failed (brain/session.py:424-425, placed_current
                # in Session._robot_on_board).
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
                t=e["t"], seconds_since_step_start=round(e["t"] - current_since.get(step, start["t"]), 3),
            ))
    return out


def build_hints(events: list[dict]) -> list[HintShown]:
    all_events, events = events, _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, prov, lesson_id = start.get("session", ""), provenance(start, all_events), field_of(start, "lesson_id")
    return [
        HintShown(sid, prov, lesson_id, field_of(e, "step"), field_of(e, "hint_level"), e["t"])
        for e in events if e["kind"] == "hint"
    ]


def build_robot_cmds(events: list[dict]) -> list[RobotCmd]:
    """Each robot_cmd, with ok/error from the next robot_result for the same command."""
    all_events, events = events, _after_start(events)
    if not events:
        return []
    start = events[0]
    sid, mode, prov = start.get("session", ""), start.get("mode", ""), provenance(start, all_events)
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


def build_camera(events: list[dict]) -> Optional[CameraChecks]:
    """Stable boards and unreadable-block waits in one session, from session_start on."""
    all_events, events = events, _after_start(events)
    if not events:
        return None
    start = events[0]
    unreadable = sum(1 for e in events if e["kind"] == "verdict"
                     and field_of(e, "verdict") == VALUES["waiting"] and field_of(e, "fault_cell") is not None)
    return CameraChecks(start.get("session", ""), start.get("mode", ""), provenance(start, all_events),
                        field_of(start, "lesson_id"), sum(1 for e in events if e["kind"] == "board"), unreadable)


def furthest_step(events: list[dict]) -> int:
    """Steps completed: the furthest step_advance reached. Counting advances would count a
    step again each time it is re-passed after a step_back."""
    reached = [field_of(e, "to_step") for e in events if e["kind"] == "step_advance"]
    return max((s for s in reached if isinstance(s, int)), default=0)


def build_session(events: list[dict], lessons: dict[str, Lesson], path: str = "", skipped: int = 0) -> Optional[SessionRow]:
    all_events, events = events, _after_start(events)
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
        provenance=provenance(start, all_events),
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
        steps_completed=furthest_step(events),
        skipped_lines=skipped,
        human_side=field_of(start, "human_side"),
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
    """Every session of every provenance, except lesson mismatches, which are only listed.
    Callers filter with Tables.only()."""
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
        mismatch = lesson_mismatch(row, _session_start(events), lessons)
        if mismatch is not None:
            t.lesson_mismatches.append(mismatch)  # its steps would map to the wrong cubes: no metric sees it
            continue
        if row.lesson_id is not None and lesson_steps(_session_start(events), lessons) is None:
            unknown.add(row.lesson_id)
        t.sessions.append(row)
        t.attempts += build_attempts(events, lessons)
        t.hints += build_hints(events)
        t.robot_cmds += build_robot_cmds(events)
        t.camera.append(build_camera(events))  # not None: build_session found a session_start
    if tasks_dir is not None:
        t.task_steps = read_task_reports(tasks_dir)
    t.unknown_lessons = sorted(unknown)
    return t
