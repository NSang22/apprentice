# Proposal: video to robot plan format

**Status: proposal, not the agreed contract. It overlaps with README section 5 and the team
has to reconcile the two.**

I wrote `SPEC.md` and the two JSON files on 2026-10-10, before the team README landed, as the
hand-off spec for whoever built the video side. It describes a different output than the
README does, so it is parked here rather than in `shared/`.

| | This proposal | README section 5 (agreed) |
|---|---|---|
| Output | `task_plan.json` | `Lesson` with `steps` |
| A step | `{object, action, goal, yaw_deg, pose, evidence}` | `{color, cell, level, t_video, narration}` |
| Where things go | Relations: `beside`, `on_top`, `cell`, exact mm `position` | A cell plus a level |
| Positions | Sheet millimetres, continuous | Grid cells only |
| Objects | `red_cube`, `blue_cube`, `yellow_cube`, 48 mm foam | red, blue, yellow, green, 1.25 in wooden |
| Source of truth | Gemini observe and compile passes, then validation | Deterministic grid diffs, grid wins over narration |

## What is worth keeping

- `task_plan.schema.json` is a working JSON Schema (draft 2020-12), checked against the worked
  example in `SPEC.md`. It catches unknown object ids, bad cells, `action` and `goal`
  mismatches, out-of-range confidences and positions off the sheet. If the team formalises
  `Lesson` as a schema, this is a usable starting point.
- The validation rules in `SPEC.md` section "Validation" survive the format change almost
  unedited, especially "never invent steps" and "fail the run rather than guess".
- `catalog.json` records the measured 48 mm cube edge, which the robot module needs regardless
  of which plan format wins.

## What conflicts

- **Relations versus cells.** This proposal lets the video side say "red beside blue, 10 mm
  gap" and leaves the robot to resolve it against the live scene in mm. The README restricts
  everything to the 9 cells and a level, which is simpler, discretised and easier to grade.
  The README approach is almost certainly the right call for the demo. The relation types
  matter only if the project later wants off-grid placement.
- **Cell naming.** Both this spec and the robot module read `A1` as row A, column 1, with row
  A nearest the arm. The README reads it as column A, row 1, front-left. See the Open
  questions section in `../../../robot/README.md`.
- **Who decides.** This proposal has Gemini compile the plan. The README has deterministic
  grid diffs decide, with LLMs restricted to hint text. The README rule is stricter and better.

## Recommendation

Adopt the README's `Lesson` format. Lift the schema validation and the "fail rather than
guess" rules from here into `shared/`, and drop the relation goal types unless off-grid
placement comes back as a requirement. Nothing in this folder should be imported by running
code while it sits here.
