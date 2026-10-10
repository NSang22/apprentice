# ui/

The monitor the student follows while building. It shows one big instruction, the board as
the student sees it, kind feedback, the hint ladder, and two progress strips. It reads
`SessionState` (README 5.4) from `brain.server` on ws `:8766` and never sends anything back.

## Run

```bash
python ui/serve.py        # open http://localhost:8000, press F11 for fullscreen
```

No network needed: the page, its styles, and three.js (`vendor/`, r160, MIT) are all local.
`serve.py` serves only `ui/` and `videos/` (for show-me clips), with correct MIME types and
byte ranges so clips can seek.

Touch free: nobody clicks during the demo. Keys `2` / `3` (or `v`) switch the board between
2D (default) and 3D. URL options: `?view=3d`, `?ws=ws://<host>:8766`, `?side=row1`.

## What it shows

| SessionState | Monitor |
|---|---|
| no data yet | "Waiting for the camera", connection pill "Connecting" |
| `expected_step` | big instruction "Place the blue cube on C1" (robot mode: "Blue cube to C1"), ghost cube pulsing in its cell |
| `PASS` | green check, held 1.6 s so it survives the instant step advance |
| `WRONG_*`, `EXTRA`, `MISSING` with `fault_cell` | calm amber card with `hint_text`, amber ring on `fault_cell` |
| `WAITING` | "Waiting for the camera" |
| `hint_level` 2 | ghost pulses harder, arrow from the misplaced cube to its cell |
| `hint_level` 3 | "Watch how it was done": `show_me` clip looped between start and end, muted. Skipped when `show_me` is null or the clip fails to load |
| `hint_level` 4 | banner "The robot will point to the right spot." |
| robot mode | large robot status ("Placing", "Fixing it", "Checks out") plus `hint_text` |
| `progress` containing `error` | "The robot needs a hand" |
| `expected_step` null | "All done!" |
| socket closed | last state dimmed, "Reconnecting to Apprentice...", retries every 0.5 to 3 s |

The board is drawn from the student's side, `HUMAN_SIDE` in `config.js`. It must match
`brain/settings.py` by hand: if one changes, change the other. For `row3`: column A on the
left, row 3 nearest, robot side at the top.

## Needs from brain (not built in the UI until these exist)

1. **Lesson overview before a session (task 4).** `SessionState` has `lesson_id` but no
   `title` and no `final_board` (or `steps`), and the brain starts the session the moment it
   launches, so there is no "ready" state to show a Start screen in. Needed: `title`,
   `final_board`, and a `phase` (`ready` / `running` / `complete` / `stopped`), with the brain
   holding in `ready` until the board is clear or the lesson is started.
2. **`expected_cell` for older cubes.** The level 2 arrow points at `expected_step.cell`. When
   the fault is an older cube that got knocked (not the current step's cube), the right
   target is the checker's `expected_cell`, which `SessionState` does not carry. Until it
   does, the arrow only shows when the cube at `fault_cell` is the current step's color.
3. **A robot status field.** The robot panel word ("Placing", "Fixing it") is matched from
   `hint_text`. A `robot_status` enum (or the `phase` above) would make it exact.

## Checking it

No unit tests (AGENTS.md: UI is checked by looking at it). Run the fake brain commands in
README section 8 next to `python ui/serve.py` and watch the monitor.
