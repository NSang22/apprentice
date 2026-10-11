-- Apprentice dashboard: one named query per dataset.
-- Data queries filter on the :provenance and :lesson_id parameters, so one tile never mixes
-- provenance or lessons. Use 'real' for anything a judge sees.
-- Every rate comes with its n. No query or tile adds up or stacks a rate: rates are always
-- recomputed from passes and n. Tables show rates as text ("40%", "no data" when n is 0).
-- Steps are 0-based in the tables; step_number adds 1 for display.
-- Queries named options_* only list choices for the dropdown filters.

-- name: options_provenance
SELECT provenance
FROM VALUES ('real', 1), ('simulation', 2), ('unverified', 3), ('synthetic', 4) AS t(provenance, sort_key)
ORDER BY sort_key;

-- name: options_lesson
SELECT DISTINCT lesson_id
FROM workspace.apprentice.first_try_by_mode
WHERE lesson_id IS NOT NULL
ORDER BY lesson_id;

-- name: session_count
-- Reads the silver sessions table: no gold table counts sessions.
SELECT :provenance AS provenance, :lesson_id AS lesson_id,
       COUNT(*) AS sessions,
       COUNT_IF(mode = 'human') AS human,
       COUNT_IF(mode = 'robot') AS robot
FROM workspace.apprentice.sessions
WHERE provenance = :provenance AND lesson_id = :lesson_id;

-- name: first_try_human_vs_robot
SELECT lesson_id, mode, n, passes, rate AS first_try_rate
FROM workspace.apprentice.first_try_by_mode
WHERE provenance = :provenance AND lesson_id = :lesson_id
ORDER BY mode;

-- name: hardest_steps
SELECT lesson_id, step + 1 AS step_number, mode,
       CASE WHEN n = 0 THEN 'no data' ELSE CONCAT(CAST(ROUND(rate * 100) AS INT), '%') END AS first_try_rate,
       n, passes, n_passed, n_not_passed, mean_attempts, median_seconds_to_pass
FROM workspace.apprentice.step_difficulty
WHERE provenance = :provenance AND lesson_id = :lesson_id AND n > 0
ORDER BY rate ASC NULLS LAST, n DESC
LIMIT 10;

-- name: mistake_mix
SELECT lesson_id, mode, verdict, n AS attempts, verdict_count, share
FROM workspace.apprentice.verdict_mix
WHERE provenance = :provenance AND lesson_id = :lesson_id AND verdict <> 'PASS'
ORDER BY mode, verdict_count DESC;

-- name: cell_error_map_3x3
-- Brain cell names: columns A to C, rows 1 to 3. One row per (mode, grid row), one text column
-- per grid column: pass rate and n, or "no data".
SELECT mode, grid_row,
       MAX(CASE WHEN grid_col = 'A' THEN label END) AS A,
       MAX(CASE WHEN grid_col = 'B' THEN label END) AS B,
       MAX(CASE WHEN grid_col = 'C' THEN label END) AS C
FROM (
  SELECT mode, grid_row, grid_col,
         CASE WHEN n = 0 THEN 'no data'
              ELSE CONCAT(CAST(ROUND(rate * 100) AS INT), '% (n=', n, ')') END AS label
  FROM workspace.apprentice.cell_error_map
  WHERE provenance = :provenance AND lesson_id = :lesson_id
) AS cells
GROUP BY mode, grid_row
ORDER BY mode, grid_row;

-- name: hint_by_level
-- One rate per hint level: passes and n added up across failed verdicts, then divided.
SELECT lesson_id, hint_level, SUM(n) AS n, SUM(passes) AS passes,
       CASE WHEN SUM(n) = 0 THEN NULL ELSE SUM(passes) / SUM(n) END AS next_try_pass_rate
FROM workspace.apprentice.hint_effectiveness
WHERE provenance = :provenance AND lesson_id = :lesson_id
GROUP BY lesson_id, hint_level
ORDER BY hint_level;

-- name: hint_breakdown
SELECT lesson_id, failed_verdict, hint_level, n, passes,
       CASE WHEN n = 0 THEN 'no data' ELSE CONCAT(CAST(ROUND(rate * 100) AS INT), '%') END AS next_try_pass_rate
FROM workspace.apprentice.hint_effectiveness
WHERE provenance = :provenance AND lesson_id = :lesson_id
ORDER BY failed_verdict, hint_level;

-- name: with_hints_file_vs_without
-- Empty until the brain logs which hints file a session loaded.
SELECT lesson_id, step + 1 AS step_number, hints_file,
       n, passes, rate AS first_try_rate, n_passed, mean_attempts
FROM workspace.apprentice.hints_file_effect
WHERE provenance = :provenance AND lesson_id = :lesson_id
ORDER BY step, hints_file;

-- name: robot_retries
SELECT lesson_id,
       CASE outcome WHEN 'pass_try_1' THEN '1. First try'
                    WHEN 'pass_after_1_retry' THEN '2. After 1 retry'
                    WHEN 'pass_after_2_retries' THEN '3. After 2 retries'
                    WHEN 'stopped' THEN '4. Stopped'
                    ELSE '5. Unfinished' END AS outcome_label,
       n AS robot_steps, steps, rate
FROM workspace.apprentice.robot_retry_stats
WHERE provenance = :provenance AND lesson_id = :lesson_id
ORDER BY outcome_label;
