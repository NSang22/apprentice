# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, Devin) working on Apprentice. Read `README.md` first; it holds the full design. This file tells you how to work in this repo.

## Context you must keep in mind

- This is a 36-hour hackathon project (Knight Hacks IX, ends Sunday Oct 11, 2026). **Reliability beats features.** A small thing that works every time scores higher than a big thing that works half the time.
- The judges score Functionality on whether what we **mention** works. Never add a feature, flag, or UI element that implies a capability that does not work end to end.
- Four people work in parallel on separate modules. Each human owns one module. Agents work inside the module their human owns.

## Module ownership

| Module | Owner role | You may edit |
|---|---|---|
| `perception/` | Jetson | Only if your human owns perception |
| `brain/` | Brain (Nikhil) | Only if your human owns brain |
| `robot/` | Robot | Only if your human owns robot |
| `ui/` | Face | Only if your human owns UI |
| `shared/` | Whole team | **Do not edit without explicit approval from your human, who must confirm the team agreed** |
| `docs/` | Everyone | Append only (see below) |

If a task needs a change outside your module, stop and tell your human what change is needed and why. Do not make it yourself.

## The contracts are frozen

`shared/schemas.py` and `shared/config.py` define `GridState`, `Step`, `Lesson`, `SessionState`, the verdict enum, the robot API, cell names, colors, and ports. Other people's code depends on them byte for byte.

- Import from `shared/`. Never redefine these types locally.
- Never rename a field, change a type, or reorder enum values.
- If a contract seems wrong, propose the change to your human in one short paragraph and wait.

Key conventions (details in README section 5):
- Cells are `A1` to `C3`, as printed on the sheet: the letter is the row (A along the arm's edge, C along the student's edge), the number is the column (1 to 3 left to right for the student, who stands opposite the arm). Tag 0 is at A1, 1 at A3, 2 at C3, 3 at C1.
- Colors are exactly `red`, `orange`, `yellow`, `teal`, `skyblue`, `purple`, or `unknown`.
- Stacks are lists bottom to top. All 9 cells always appear in a board.
- Ports: perception ws `8765`, brain ws `8766`, robot HTTP `8770`.

## Non-negotiable behavior rules

1. **Unknown is never a pass.** Any uncertainty (low confidence, `unknown` color, hands over the grid, arm not at home) yields `WAITING`, never `PASS`.
2. **The grid wins.** Correctness comes from deterministic grid diffs. LLMs and VLMs may write hint text or descriptions; they never decide pass or fail.
3. **Robot retries cap at 2.** After two failed retries, stop the robot, return home, and surface the problem in `SessionState`. Never loop.
4. **The journal is append-only.** Write every board, verdict, hint, robot command, and robot result to `data/sessions/<session_id>.jsonl` through `brain/journal.py`. Never rewrite or delete lines.
5. **No network dependency in the live demo path.** The table demo must run with venue Wi-Fi down. Cloud calls (Whisper API, VLM, Databricks) happen offline, before or after a session, never inside the live check loop.
6. **Fail loudly in development, gracefully in the demo.** Raise on contract violations in tests. In the running demo, log the error to the journal and show a calm message in the UI instead of crashing.

## How to build

- **Python 3.11+** for perception, brain, robot. Use type hints and dataclasses. Keep dependencies minimal: `opencv-python`, `numpy`, `websockets`, `fastapi`, `uvicorn`, `pupil-apriltags` (or OpenCV's 36h11 dictionary), `openai-whisper` or the Whisper API for offline narration. Ask your human before adding anything else.
- **UI:** a single web app. Plain HTML/JS or a small React app is fine. three.js only if the 2D version already works.
- **Build against fakes.** Brain uses `brain/fake_feed.py` and `robot/stub.py`. UI uses a fake `SessionState` stream. Nobody waits on hardware.
- Keep functions small and pure where possible (stabilizer, checker, extractor, gate should be pure functions over boards and steps). Put I/O at the edges.
- Read config values (stable frame count, confidence threshold, ports) from `shared/config.py`. No magic numbers scattered in code.

## Testing (keep it fast: speed matters more than coverage now)

- **Test only core logic:** stabilizer, checker, extractor, session, and pure helpers like cell mapping. Write a **few** tests per feature: the happy path plus the one or two failures that would break the demo. No exhaustive permutations, no large parametrized tables.
- **Do not write tests** for UI code, servers, CLI argument parsing, scripts, or perception visuals. Check those by running them once (perception: `--debug` window; UI: look at it).
- While iterating, run **only the test file you touched** (`pytest tests/test_x.py -q`). Run the full suite **once**, at the end.
- Don't do live end-to-end dry runs, screenshots, or multi-process smoke tests unless your human asks.
- Don't rewrite or expand existing tests unless your change breaks them.
- Scripted board sequences live in `tests/scripts/*.json` and double as fake feed inputs.
- Report in two lines: what you tested and what you didn't.

## Definition of done

A task is done when:
1. It works against the fakes (or real hardware, if the task says so), checked by running it once.
2. Tests for any core logic you changed pass (see Testing; no new tests for UI, servers, or scripts).
3. It writes the right journal events.
4. Your human can run it with one command, and that command appears in README section 8.
5. You added a line to `docs/challenges-log.md` if anything took more than 20 minutes to figure out.

Do not report a task as done if any of these fails. Say what is left instead.

## Docs you must keep updated

- `docs/challenges-log.md`: one line per real struggle: what broke, how we fixed it. The team uses this in the pitch, so be specific and honest.
- `docs/tried-first.md`: when an approach fails and we switch, add a row: idea, what happened, what we did instead.
- README section 8 (how to run): add or fix commands for anything you ship.

Append to these files. Do not rewrite other people's entries.

## Writing style for docs and UI text

- Active voice ("the checker rejects the board", not "the board is rejected").
- No em dashes. Use commas, colons, or separate sentences.
- UI text is short and kind: "Close! Move the blue cube one cell left." Never blame the user.

## Git

- Small commits with clear messages, scoped to your module (`brain: add stabilizer`).
- Pull before you push. Never force push to `main`.
- Never commit large videos, journals, or secrets. `videos/` and `data/sessions/` are gitignored except tiny samples.

## When in doubt

Ask your human. A one-line question costs less than an integration bug at 3 AM.