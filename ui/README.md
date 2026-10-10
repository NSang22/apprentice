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
`2` / `3` (or `v`) switch the board between 2D and 3D. Each key shows a short confirmation.
URL options: `?view=3d`, `?ws=ws://<host>:8766`.

## What it shows

| SessionState | Monitor |
|---|---|
| no data yet | "Waiting for the camera", connection pill "Connecting" |
| `phase` "ready" | ready screen: `lesson_title`, a thumbnail of `final_board`, `hint_text` ("Waiting for the camera" / "Please clear the board to start."), and "Press Space to start" once the board is clear |
| `expected_step` | big instruction "Place the blue cube on C1" (robot mode: "Blue cube to C1"), ghost cube pulsing in its cell |
| `PASS` | green check, held 1.6 s so it survives the instant step advance |
| `WRONG_*`, `EXTRA`, `MISSING` with `fault_cell` | calm amber card with `hint_text`, amber ring on `fault_cell` |
| `WAITING` | "Waiting for the camera" |
| `hint_level` 2 | ghost pulses harder, arrow from `fault_cell` to `expected_cell` |
| `hint_level` 3 | "Watch how it was done": `show_me` clip looped between start and end, muted. Skipped when `show_me` is null or the clip fails to load |
| `hint_level` 4 and the arm busy | banner "The robot is pointing to the right spot." |
| robot mode | robot panel from `robot_status`: Placing, Fixing it, Going home, Checking, Checks out |
| `phase` "complete" | "All done!" with the finished structure |
| `phase` "stopped" | calm "Let's sort it out together" screen with `hint_text` and "Press R to try again" |
| socket closed | last state dimmed, "Reconnecting to Apprentice...", retries every 0.5 to 3 s |

The board is drawn from the student's side, `human_side` in each SessionState (set in
`brain/settings.py`). For `row3`: column A on the left, row 3 nearest, robot side at the top.

`robot_status` has no "pointing" value (the shared enum is frozen), so the monitor treats
human mode at `hint_level` 4 with `robot_status` "fixing" (the hover) as pointing.

## Checking it

No unit tests (AGENTS.md: UI is checked by looking at it). Run the fake brain commands in
README section 8 next to `python ui/serve.py` and watch the monitor.
