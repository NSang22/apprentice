> **Status: proposal, superseded in part by README section 5.** Read
> [`README.md`](README.md) in this folder first for how this differs from the agreed
> `Lesson` format and what is worth keeping. Nothing here is a frozen contract.

# Video Ingestion Pipeline Spec

**Owner:** you (video side). **Consumer:** the robot executor (Taher's side).

**Your job:** turn a phone video of a person rearranging objects into a `task_plan.json` that
the robot can execute. The robot side handles everything after that: the live camera, measuring
where objects are now, arm motion, and checking results. You never touch the robot.

```
 demo video ──► [ YOUR PIPELINE ] ──► task_plan.json ──► [ ROBOT EXECUTOR ] ──► arm moves
               Gemini observe +                          live camera + tags,
               compile + validate                        IK, pick/place, verify
```

This builds on the existing app's two Gemini passes (observe, then compile). The main change is
the **compile output**: it must follow the schema below, use the known object IDs, and describe
goals as **relations between objects** (or sheet positions), not free-form animation poses.

---

## Input

| Item | Details |
|---|---|
| Video | Phone video (mp4/mov), uploaded through the existing app flow |
| Scene | The demo is done **on the printed AprilTag sheet** (`robot/assets/Apprentice_AprilTag_Grid.pdf`, US Letter, printed at 100%) with the known objects |
| Camera | Handheld or propped. Roughly overhead is best. Keep **at least 2 corner tags visible** most of the time |
| Objects | Only objects from the catalog below |

**Why the sheet matters:** it gives a shared reference between the demo and the robot's table.
"Left of", "right of", and grid cells (A1..C3) then mean the same thing in both places. Keep it
upright in the video the way it is printed: tag 0 top-left, tag 1 top-right, tag 3 bottom-left,
tag 2 bottom-right.

Print your own copy of the sheet to record test videos. Its exact geometry is in
`robot/assets/workspace_sheet.json` (tags `DICT_APRILTAG_36h11`, IDs 0-3, 31.75 mm black squares, 3x3 grid of 57.15 mm cells).

### Object catalog (v1)
| id | description |
|---|---|
| `red_cube` | Red foam cube, raised letter on faces |
| `blue_cube` | Blue foam cube, raised letter on faces |
| `yellow_cube` | Yellow foam cube, raised letter on faces |

Cube edge length: **48 mm** (measured 2026-10-10).

The machine-readable version of this table is `catalog.json`, beside this file. Read ids from
there rather than hard-coding them. New objects get added there first, then taught to both
sides.

---

## Output: `task_plan.json`

Write one file per run: `runs/<run_id>/task_plan.json`. Also expose it from the existing backend
(for example `GET /runs/<run_id>/task_plan`) so the executor can fetch it.

### Directions and frame
All directions refer to the **printed page**, as you'd read it:

| Word | Meaning on the sheet | Sheet-frame axis |
|---|---|---|
| `left` | toward tags 0/3 | -x |
| `right` | toward tags 1/2 | +x |
| `top` | toward tags 0/1 (the robot's side) | -y |
| `bottom` | toward tags 3/2 | +y |

`yaw_deg`: the object's rotation seen from above. `0` = edges parallel to the sheet's grid lines;
positive = counterclockwise seen from above. For cubes, only yaw mod 90 matters.

### Schema
```jsonc
{
  "version": "1.0",
  "run_id": "string",
  "source_video": "string",              // filename or path
  "sheet_visible": true,                 // were >=2 tags visible for most of the demo?
  "objects": ["red_cube", "blue_cube", "yellow_cube"],   // catalog ids that appear

  "steps": [                             // in the order the person did them
    {
      "index": 1,
      "action": "PLACE",                 // "PLACE" | "STACK"
      "object": "red_cube",              // catalog id, the object being moved
      "goal": { ... },                   // exactly one goal type, see below
      "yaw_deg": 0,                      // final yaw of the moved object, or null if unclear
      "pose": "flat",                    // v1: always "flat" (resting on a face)
      "evidence": {
        "t_start_s": 3.2,                // when the hand picks the object up
        "t_end_s": 5.9,                  // when the hand lets go
        "confidence": 0.9,               // 0..1, your pipeline's own estimate
        "notes": "string"
      }
    }
  ],

  "final_state": [                       // how the table looks at the end (for verification)
    {"object": "red_cube", "on": "blue_cube"},
    {"object": "yellow_cube", "on": "table", "cell": "B3"}
  ],

  "warnings": []                         // anything unclear: occlusion, low confidence, unknown object
}
```

### Goal types (use exactly one per step)
```jsonc
// Next to another object, on the table
{"type": "beside", "reference": "blue_cube", "side": "right", "gap_mm": 10}

// On top of another object (action must be "STACK")
{"type": "on_top", "reference": "blue_cube"}

// In a named grid cell on the sheet
{"type": "cell", "cell": "B2"}

// Exact sheet position (STRETCH, only if you measure it from the tags; see below)
{"type": "position", "position_mm": [120.0, 160.0]}
```
- `gap_mm` is the gap between the two objects' facing edges. Use `10` if you can't estimate it.
- Prefer `beside` / `on_top` when the person placed something relative to another object.
  Prefer `cell` when they placed it in an open area.

### Example
Demo: the person stacks red on blue, then puts yellow to the right of blue.
```json
{
  "version": "1.0",
  "run_id": "2026-10-10_001",
  "source_video": "demo_stack.mp4",
  "sheet_visible": true,
  "objects": ["red_cube", "blue_cube", "yellow_cube"],
  "steps": [
    {"index": 1, "action": "STACK", "object": "red_cube",
     "goal": {"type": "on_top", "reference": "blue_cube"},
     "yaw_deg": 0, "pose": "flat",
     "evidence": {"t_start_s": 2.1, "t_end_s": 4.8, "confidence": 0.92, "notes": ""}},
    {"index": 2, "action": "PLACE", "object": "yellow_cube",
     "goal": {"type": "beside", "reference": "blue_cube", "side": "right", "gap_mm": 10},
     "yaw_deg": 0, "pose": "flat",
     "evidence": {"t_start_s": 6.0, "t_end_s": 8.3, "confidence": 0.85, "notes": ""}}
  ],
  "final_state": [
    {"object": "blue_cube", "on": "table"},
    {"object": "red_cube", "on": "blue_cube"},
    {"object": "yellow_cube", "on": "table"}
  ],
  "warnings": []
}
```

---

## Validation (your pipeline must enforce these before writing the file)
The robot must never receive a plan it can't safely interpret. Check these, and if any fail,
**fail the run with a clear error** instead of guessing:

1. Valid JSON matching the schema; `version` is `"1.0"`.
2. Every `object` and `reference` is a catalog id. No object is its own reference.
3. `action` is `STACK` exactly when `goal.type` is `on_top`.
4. v1 limits: `pose` is `"flat"`; stacks are at most **2 high** (no stacking on a cube that's already on another).
5. `cell` is one of `A1..C3`; `side` is one of `left/right/top/bottom`; `position_mm` lies within the sheet (0-215.9 x 0-279.4 mm).
6. Steps are in time order, and each `t_end_s` is after its `t_start_s`.
7. `final_state` agrees with what the steps produce (replay the steps symbolically and compare).
8. Any step with `confidence < 0.6` adds a `warnings` entry. The executor will ask a human before running it.
9. **Never invent steps** to make a plan look complete. Missing or unclear means a warning, not a guess.

Save the raw Gemini outputs (observe + compile) next to the plan in `runs/<run_id>/` for debugging.

---

## Stretch goals (after v1 works)
- **Exact positions from the tags.** When the sheet is visible in the final frames, detect the AprilTags
  with OpenCV (`cv2.aruco`, `DICT_APRILTAG_36h11`), compute the image-to-sheet homography from the tag
  corners in `robot/assets/workspace_sheet.json`, and output `position` goals in mm. Working detection code
  already exists in `robot/perception/sheet.py` (`SheetLocalizer`). Copy it, or ask Taher to promote
  it to `shared/` so both sides use one implementation.
- **Yaw from the video.** Estimate each object's final rotation from its edges in the last frame.
- `pose: "on_edge"` for standing objects (only once the robot side says the arm can do it).

---

## Done when
1. 5 test videos you record (single place, beside, cell, stack, and a 2-3 step combo) each produce a
   plan that matches a hand-written expected plan, checked by an automated test.
2. A deliberately bad video (unknown object, or hands blocking the view) produces either a
   validation error or warnings, never a confident wrong plan.
3. The plan is available both as the file and from the backend endpoint, with run status and logs
   visible in the existing dashboard.

## Files to share with you
Everything is in the repo now; nothing is sent separately.
- This spec
- `robot/assets/Apprentice_AprilTag_Grid.pdf`: print at 100% / Actual size and check that the 2.00 in scale bar measures 50.8 mm
- `robot/assets/workspace_sheet.json`: exact tag and grid geometry
- `task_plan.schema.json` (this folder): this spec's output, as a JSON Schema you can validate against
- `catalog.json` (this folder): the object ids a plan may reference
- `robot/samples/overhead_02.png`: what the robot's camera sees (same sheet and cubes)
- `../../Video_to_Robot_Project_Guide.pdf`: project background

Questions about the format go to Taher. If you need a new goal type, propose it first. The robot
side has to implement each type before plans can use it.
