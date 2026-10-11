# Databricks notebook source
# MAGIC %md
# MAGIC # Apprentice analytics: build tables
# MAGIC
# MAGIC Reads session journals from a volume, runs this repo's own analytics code (the same loader,
# MAGIC metrics, and hint write-back as the local commands), and writes bronze, silver, and gold
# MAGIC Delta tables. Runs after sessions, never during one. Spark is only used to write tables.
# MAGIC
# MAGIC Bronze keeps every journal line. Silver and gold leave synthetic sessions out unless
# MAGIC `include_synthetic` is true. Every row has a `provenance` column; only `real` is judge-facing.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "apprentice")
dbutils.widgets.text("volume_path", "/Volumes/workspace/apprentice/journals")
dbutils.widgets.text("repo_root", "")  # blank: found from this notebook's own location
dbutils.widgets.dropdown("include_synthetic", "false", ["false", "true"])

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
volume_path = dbutils.widgets.get("volume_path").strip()
include_synthetic = dbutils.widgets.get("include_synthetic") == "true"

# COMMAND ----------

import os
import sys
from pathlib import Path


def looks_like_repo(p: Path) -> bool:
    return all((p / f).is_file() for f in ("analytics/load.py", "shared/config.py", "brain/checker.py"))


def find_repo_root(widget: str) -> Path:
    if widget:
        p = Path(widget)
        if not looks_like_repo(p):
            raise RuntimeError(f"repo_root {p} does not hold analytics/, shared/, and brain/")
        return p
    starts = [Path(os.getcwd())]
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(Path("/Workspace" + nb).parent)
    except Exception:  # noqa: BLE001 - context lookup differs across runtimes; cwd may be enough
        pass
    for start in starts:
        for p in [start, *start.parents]:
            if looks_like_repo(p):
                return p
    raise RuntimeError(f"could not find the repo root from {starts}; set the repo_root widget")


repo_root = find_repo_root(dbutils.widgets.get("repo_root").strip())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
for name in [m for m in sys.modules if m.split(".")[0] in ("analytics", "shared", "brain")]:
    del sys.modules[name]  # pick up fresh code after a git pull, without restarting Python
print(f"repo_root {repo_root}")

# COMMAND ----------

from analytics.databricks import tables as T
from analytics.hint_writeback import build_hints, write_hints
from analytics.load import DEFAULT_LESSON_DIRS, PROVENANCES, load_all
from analytics.report import build_report, summary

journals = Path(volume_path)
everything = load_all(journals, DEFAULT_LESSON_DIRS)
included = [p for p in PROVENANCES if include_synthetic or p != "synthetic"]
t = everything.only(included)
print(f"{len(everything.sessions)} sessions on the volume, {len(t.sessions)} included ({', '.join(included)})")

# COMMAND ----------

rows = {
    "journal_raw": [
        r for p in sorted(journals.glob("*.jsonl"))
        for r in T.journal_raw_rows(p.name, p.read_text(encoding="utf-8", errors="replace"))
    ]
}
rows.update(T.silver(t))
rows.update(T.gold(everything, include_synthetic))

# COMMAND ----------

# Hint write-back: the same function and real-only guard as `python -m analytics.hint_writeback`.
import shutil
import tempfile

spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`exports`")
exports = Path(f"/Volumes/{catalog}/{schema}/exports")
rows["hint_policy"] = []
real_lessons = sorted({s.lesson_id for s in everything.sessions
                       if s.provenance == "real" and s.mode == "human" and s.lesson_id})
for lesson_id in real_lessons:
    hints = build_hints(lesson_id, everything.sessions, everything.attempts)
    if hints is None:
        print(f"{lesson_id}: no step with enough real data, no hints file")
        continue
    with tempfile.TemporaryDirectory() as tmp:  # write locally, then one plain copy onto the volume
        local, digest = write_hints(hints, Path(tmp))
        shutil.copyfile(local, exports / local.name)
    print(f"{lesson_id}: wrote {exports / local.name} sha256 {digest}")
    rows["hint_policy"] += T.hint_policy_rows(hints, digest)
if not real_lessons:
    print("no real human sessions, so no hints files")

# COMMAND ----------

for name in [n for layer in T.LAYERS.values() for n in layer]:
    df = spark.createDataFrame(T.as_tuples(name, rows[name]), schema=T.SCHEMAS[name])
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"`{catalog}`.`{schema}`.`{name}`"))
    print(f"{catalog}.{schema}.{name}: {len(rows[name])} rows")

# COMMAND ----------

# The same summary `python -m analytics.report` prints, to compare by eye.
print(summary(build_report(everything, include_synthetic)))
