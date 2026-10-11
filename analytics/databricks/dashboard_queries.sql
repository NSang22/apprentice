-- Apprentice dashboard: one named query per tile. Gold tables only.
-- Every query filters on provenance through the :provenance parameter. Use 'real' for anything a
-- judge sees; 'simulation', 'unverified', and 'synthetic' are for checking the pipeline.
-- Every rate comes with its n. A null rate means n is 0: show "no data", not 0%.
-- Steps are 0-based in the tables; step_number adds 1 for display.

-- name: first_try_human_vs_robot
SELECT lesson_id, mode, n, passes, rate AS first_try_rate
FROM workspace.apprentice.first_try_by_mode
WHERE provenance = :provenance
ORDER BY lesson_id, mode;

-- name: hardest_steps
SELECT lesson_id, step + 1 AS step_number, mode,
       n, passes, rate AS first_try_rate,
       n_passed, n_not_passed, mean_attempts, median_seconds_to_pass
FROM workspace.apprentice.step_difficulty
WHERE provenance = :provenance AND n > 0
ORDER BY rate ASC NULLS LAST, n DESC
LIMIT 10;

-- name: mistake_mix
SELECT lesson_id, mode, verdict, n AS attempts, verdict_count, share
FROM workspace.apprentice.verdict_mix
WHERE provenance = :provenance AND verdict <> 'PASS'
ORDER BY lesson_id, mode, verdict_count DESC;

-- name: cell_error_map_3x3
-- Brain cell names: columns A to C, rows 1 to 3. One output row per grid row.
SELECT lesson_id, mode, grid_row,
       MAX(CASE WHEN grid_col = 'A' THEN rate END) AS a_pass_rate, MAX(CASE WHEN grid_col = 'A' THEN n END) AS a_n,
       MAX(CASE WHEN grid_col = 'B' THEN rate END) AS b_pass_rate, MAX(CASE WHEN grid_col = 'B' THEN n END) AS b_n,
       MAX(CASE WHEN grid_col = 'C' THEN rate END) AS c_pass_rate, MAX(CASE WHEN grid_col = 'C' THEN n END) AS c_n,
       SUM(fault_count) AS faults_in_row
FROM workspace.apprentice.cell_error_map
WHERE provenance = :provenance
GROUP BY lesson_id, mode, grid_row
ORDER BY lesson_id, mode, grid_row;

-- name: hint_effectiveness
SELECT lesson_id, failed_verdict, hint_level, n, passes, rate AS next_try_pass_rate
FROM workspace.apprentice.hint_effectiveness
WHERE provenance = :provenance
ORDER BY lesson_id, failed_verdict, hint_level;

-- name: with_hints_file_vs_without
-- Empty until the brain logs which hints file a session loaded.
SELECT lesson_id, step + 1 AS step_number, hints_file,
       n, passes, rate AS first_try_rate, n_passed, mean_attempts
FROM workspace.apprentice.hints_file_effect
WHERE provenance = :provenance
ORDER BY lesson_id, step, hints_file;

-- name: robot_retries
SELECT lesson_id, outcome, n AS robot_steps, steps, rate
FROM workspace.apprentice.robot_retry_stats
WHERE provenance = :provenance
ORDER BY lesson_id,
         CASE outcome WHEN 'pass_try_1' THEN 1 WHEN 'pass_after_1_retry' THEN 2
                      WHEN 'pass_after_2_retries' THEN 3 WHEN 'stopped' THEN 4 ELSE 5 END;
