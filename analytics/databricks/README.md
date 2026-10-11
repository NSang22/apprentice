# Analytics on Databricks

Everything here runs after sessions, never during one. The notebook runs this repo's own analytics code, so the attempt rules and metrics exist in one place (`analytics/load.py`, `analytics/metrics.py`). Nothing here is rewritten in SQL.

Workspace: catalog `workspace`, schema `apprentice`. Journals go in the volume `/Volumes/workspace/apprentice/journals/`. Hints files come out of `/Volumes/workspace/apprentice/exports/`, which the notebook creates.

## 1. Upload journals (laptop)

```bash
python -m analytics.databricks.sync push --dry-run   # list what would upload
python -m analytics.databricks.sync push             # upload new journals and journals that grew
```

This needs `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in the environment or in `.env` (gitignored). Without them the command prints how to upload by hand in the Databricks UI and exits.

## 2. Get the code into the workspace

Add this repo as a Git folder in the workspace (Workspace > Create > Git folder), or pull the latest into an existing one. The notebook imports `analytics`, `shared`, and `brain.checker` from it. All three need only the Python standard library.

## 3. Build the tables (Databricks)

Open `analytics/databricks/build_tables.py` from the Git folder and run all cells on serverless compute.

| Widget | Default | Meaning |
|---|---|---|
| `catalog` | `workspace` | Target catalog |
| `schema` | `apprentice` | Target schema |
| `volume_path` | `/Volumes/workspace/apprentice/journals` | Where the journals are |
| `repo_root` | blank | Blank means the notebook finds the repo from its own location |
| `include_synthetic` | `false` | Add synthetic sessions to silver and gold |

It overwrites these Delta tables. Schemas are declared in `tables.py`; nothing is inferred.

| Layer | Tables |
|---|---|
| bronze | `journal_raw`: every journal line as written, with file name and line number |
| silver | `sessions`, `attempts`, `hints` |
| gold | `first_try_by_mode`, `step_difficulty`, `verdict_mix`, `cell_error_map`, `hint_effectiveness`, `hints_file_effect`, `leaderboard`, `robot_retry_stats`, `hint_policy` |

Every row has a `provenance` column. Only `real` is judge-facing. Gold numbers are computed per provenance and never pooled.

The notebook also runs the hint write-back for each lesson with real human sessions, with the same real-only guard as the local command. It writes `<lesson_id>.hints.json` to the exports volume and prints its sha256. The last cell prints the same summary as `python -m analytics.report`, so you can compare the two.

## 4. Dashboard

`dashboard_queries.sql` has one named query per tile. Each reads only gold tables and takes a `:provenance` parameter. Set it to `real` for anything a judge sees.

## 5. Download a hints file (laptop)

```bash
python -m analytics.databricks.sync pull-hints --lesson staircase-v1
```

It saves `lessons/<lesson_id>.hints.json` byte for byte and prints the sha256, which matches the `hint_policy` table. The brain does not read this file yet.

## What has been tested

Locally, `tests/test_analytics_tables.py` checks that every row matches its declared schema. Nothing that needs Spark, the workspace, or the network has run yet: the notebook, the SQL queries, and the sync calls.
