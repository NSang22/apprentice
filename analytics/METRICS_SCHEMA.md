# metrics.json schema (version 1)

`python -m analytics.report` writes `data/analytics/metrics.json`. It writes a temp file and renames it, so you never read half a file. Regenerate it after sessions; nothing updates it live.

## Rules for the UI

- Show judges numbers from `by_provenance.real` only. Every other bucket is labeled and has `judge_facing: false`.
- If `contains_synthetic_data` is true, show a warning. Synthetic numbers are made up by a generator.
- A rate is always `{"n", "passes", "rate"}`. `rate` is `null` when `n` is 0. Show "no data" for `null`, never 0%.
- Every list is grouped by `lesson_id` first. Step 2 of one lesson is not step 2 of another.
- `step` is a 0-based index into the lesson's steps. Add 1 for display.
- Cells use the brain's names: letter = column, A to C left to right as the human at the row 3 edge sees them; number = row, row 1 = arm side, row 3 = human side.

## Top level

| Key | Meaning |
|---|---|
| `schema_version` | `1` |
| `generated_at` | UTC time, ISO 8601 |
| `sessions` | Session count per provenance found on disk: `real`, `unverified`, `simulation`, `synthetic` |
| `included_provenances` | Buckets present under `by_provenance` |
| `never_started`, `skipped_lines` | Journals with no `session_start`; unreadable lines |
| `bad_lesson_files`, `lessons_not_found` | Lesson problems the loader hit |
| `lesson_mismatch` | `[{session, path, provenance, lesson_id, recorded_steps, lesson_steps}]`: sessions whose recorded step count differs from the lesson file their `lesson_id` maps to (the lesson changed under the same id). Left out of every other number, `sessions` included |
| `contains_synthetic_data`, `contains_simulation_data`, `contains_unverified_data` | Flags for warnings |
| `by_provenance` | One bucket per provenance (below). `real`, `simulation`, `unverified` always appear, empty when they have no sessions. `synthetic` appears only with `--include-synthetic` |
| `robot_trials` | Robot task reports (below), kept apart from everything else |

Provenance: `real` means the grid came from the camera (`perception:`). `simulation` means a fake feed, the in-process sim, or a fake robot. `unverified` means an older journal that does not say. `synthetic` means `analytics/fake_journal.py` wrote it.

## A bucket: `by_provenance.<name>`

| Key | Shape |
|---|---|
| `label`, `judge_facing` | Text to show next to the numbers; true only for `real` |
| `sessions`, `attempts` | Counts |
| `first_try_pass_rate` | `[{lesson_id, mode, n, passes, rate}]`. The eval chart: share of steps passed on the first try |
| `step_difficulty` | `[{lesson_id, step, mode, n, passes, rate, n_passed, n_not_passed, mean_attempts, median_seconds_to_pass}]`. `rate` is first try; `mean_attempts` counts attempts until the pass; seconds run from when the step became the current one |
| `verdict_mix` | `[{lesson_id, mode, n, passes, counts: {VERDICT: int}, shares: {VERDICT: float or null}}]` |
| `cell_error_map` | `{"human" or "robot": {lesson_id: {cell: {n, passes, rate, fails, fault_count}}}}`. All 9 cells per lesson, brain cell names (row 1 = arm side). `n` counts attempts by the verdict's `expected_cell`. For `WRONG_CELL` that can be a later step's slot: a block whose color an open slot still needs is graded "move it there", so the attempt counts against that later slot, not the current step's cell |
| `hint_effectiveness` | `[{lesson_id, failed_verdict, hint_level, n, passes, rate}]`. After a mistake, how often the next try passed with that hint showing. Human only. Only tries up to the step's first pass |
| `robot_retry_stats` | `[{lesson_id, n, pass_try_1, pass_after_1_retry, pass_after_2_retries, stopped, unfinished, stops_by_reason}]`. Each bucket is a rate over `n` robot steps |
| `leaderboard` | `[{rank, lesson_id, session, provenance, total_seconds, mistakes, max_hint_level}]`. Completed human sessions, ranked per lesson |
| `hints_file_effect` | `[{lesson_id, step, with_file, without_file}]`, each side shaped like a `step_difficulty` row without the labels. Empty until the brain logs which hints file it loaded |
| `gate_stats` | `[{lesson_id, sessions, n, passes, rate}]`. `n` counts sessions whose lesson had a gate |

## `robot_trials`

From `robot/outputs/tasks/*.json` when the report runs with `--tasks`.

| Key | Shape |
|---|---|
| `cell_naming` | `"robot"`: letter is the row (A nearest the arm), number is the column. Do not mix with brain cell names |
| `tested_against_real_file` | `false` until we test against a real task report |
| `task_steps` | Steps read |
| `cells` | `[{cell, n, passes, rate, unknown, mean_error_mm, max_error_mm, mean_seconds}]`. `cell` is `null` for goals that are not a cell |
| `overall` | Same fields as one `cells` row, without `cell` |

`unknown` counts steps whose report never recorded `ok`. They are not counted as fails.

## Attempts and step backs

An attempt is one graded check of a step: `WAITING` and "still placing" (`MISSING` with no `fault_cell`) are never attempts, except a robot's "still placing" after its place for that step failed. The brain's progress follows the board: lifting an earlier block off moves the lesson back to that step (`step_back`), and putting it back passes it again. Only a step's first `PASS` is an attempt. A mistake made on a revisit is an attempt, timed from the `step_back`.
