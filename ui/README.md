# ui/

The monitor the student follows while building. It shows one big instruction, the board as
the student sees it, kind feedback, the hint ladder, and two progress strips. It reads
`SessionState` (README 5.4) from `brain.server` on ws `:8766`, and sends back only two
commands: `{"cmd": "start"}` and `{"cmd": "reset"}`.

## Run

```bash
python ui/serve.py        # open http://localhost:8000, press F11 for fullscreen
```

No network needed: the page, its styles, and three.js (`vendor/`, r160, MIT) are all local.
`serve.py` serves only `ui/` and `videos/` (for show-me clips), with correct MIME types and
byte ranges so clips can seek.

Keys (nobody clicks during the demo): `Space` starts the lesson from the ready screen (only
when the board is clear), `R` starts over (the brain ends the session and opens a new one),
`2` / `3` (or `v`) switch the board between 2D and 3D. Space and R show a short confirmation.
URL options: `?view=2d`, `?ws=ws://<host>:8766`.

The 3D table is the main view: wood tabletop, the grid sheet with AprilTag corners and labels
from the student's side, rounded 48 mm cubes with shadows (new cubes drop in, removed ones fade),
a bobbing ghost with a light column, an arc arrow at hint level 2, amber floor glow on mistakes,
and a low-poly arm on the robot side that reaches while `robot_status` is placing/fixing (or
pointing) and rests otherwise. Ready and complete turn the structure slowly; a lesson finish
bursts confetti. Everything is drawn at runtime with the vendored three.js, no network. If WebGL
or the module fails, the 2D board takes over automatically.

## What it shows

| SessionState | Monitor |
|---|---|
| no data yet | "Waiting for the camera", connection pill "Connecting" |
| `phase` "ready" | ready screen: `lesson_title`, a thumbnail of `final_board`, "Waiting for the camera" while `board_seen` is false, else `hint_text` ("Please clear the board to start."), and "Press Space to start" once the board is clear |
| `expected_step` | big instruction "Place the blue cube on C1" (robot mode: "Blue cube to C1"), ghost cube pulsing in its cell |
| `PASS` | green check, held 1.6 s so it survives the instant step advance |
| `WRONG_*`, `EXTRA`, `MISSING` with `fault_cell` | calm amber card with `hint_text`, amber ring on `fault_cell` |
| `board_seen` false | "Waiting for the camera" (camera text only ever comes from this flag, never from `hint_text`) |
| `hint_level` 2 | ghost pulses harder, arc arrow from `fault_cell` to `expected_cell` (it swings sideways when it would point straight at the camera) |
| `hint_level` 3 | "Watch how it was done": `show_me` clip under the instruction (never over the board), looped between start and end, muted. Skipped when `show_me` is null or the clip fails to load |
| `robot_status` "pointing" | banner "The robot is pointing to the right spot.", 3D arm reaches over `expected_cell` |
| robot mode | robot panel from `robot_status`: Placing, Fixing it, Going home, Checking, Checks out |
| `phase` "complete" | "All done!" with the finished structure |
| `phase` "stopped" | calm "Let's sort it out together" screen with `hint_text` and "Press R to try again" |
| socket closed | last state dimmed, "Reconnecting to Apprentice...", retries every 0.5 to 3 s |

The board is drawn from the student's side, `human_side` in each SessionState (set in
`brain/settings.py`). Cells follow the printed sheet: the letter is the row, the number the column.
For `row3` (the default, student opposite the robot): row A at the top (robot side), row C nearest,
column 1 on the left.

The ghost (2D and 3D) sits at `expected_cell` / `expected_level`, colored from `final_board`.
`phase` and `robot_status` values are defined once, in `shared/schemas.py`; the monitor does
not keep its own copies.

## Checking it

No unit tests (AGENTS.md: UI is checked by looking at it). Run the fake brain commands in
README section 8 next to `python ui/serve.py` and watch the monitor.
