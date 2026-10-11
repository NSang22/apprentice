import json

import pytest

from analytics.databricks.dashboard import (
    BANNER, DEFAULT_LESSON, QUERIES_FILE, _bar, build_dashboard, dataset_id, parse_queries, query_parameters)

OPTIONS = {"options_provenance", "options_lesson"}


def test_payload_datasets_filters_and_tiles():
    queries = parse_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    d = json.loads(json.dumps(build_dashboard(queries)))  # valid JSON all the way through
    ds = {x["displayName"]: x for x in d["datasets"]}
    assert DEFAULT_LESSON == "sample-flat-v1"

    # Every data query is pinned to one provenance and one lesson, defaulting to synthetic and the demo lesson.
    assert set(ds) == set(queries) and OPTIONS < set(ds)
    for name, x in ds.items():
        sql = "".join(x["queryLines"])
        assert not sql.rstrip().endswith(";") and "-- name:" not in sql
        if name in OPTIONS:
            assert "parameters" not in x
            continue
        assert [p["keyword"] for p in x["parameters"]] == ["provenance", "lesson_id"]
        assert [p["defaultSelection"]["values"]["values"] for p in x["parameters"]] == [
            [{"value": "synthetic"}], [{"value": "sample-flat-v1"}]]
        assert "provenance = :provenance AND lesson_id = :lesson_id" in sql
    # The provenance dropdown lists all four values, so "real" can be picked before real data exists.
    assert all(f"'{p}'" in "".join(ds["options_provenance"]["queryLines"])
               for p in ("real", "simulation", "unverified", "synthetic"))

    (page,) = d["pages"]
    widgets = {item["widget"]["name"]: item["widget"] for item in page["layout"]}
    assert widgets["banner"]["multilineTextboxSpec"]["lines"] == [BANNER]
    assert BANNER.startswith("Only 'real' comes from live sessions at the table.")

    # Filters: dropdown choices from the options dataset, sent to every dataset that uses the parameter.
    for fname, param, options, default in (("filter_provenance", "provenance", "options_provenance", "synthetic"),
                                           ("filter_lesson", "lesson_id", "options_lesson", "sample-flat-v1")):
        f = widgets[fname]["spec"]
        assert f["widgetType"] == "filter-single-select"
        fields = f["encodings"]["fields"]
        assert fields[0]["fieldName"] == param
        assert widgets[fname]["queries"][0]["query"]["datasetName"] == dataset_id(options)
        bound = {q["query"]["datasetName"] for q in widgets[fname]["queries"][1:]}
        assert bound == {dataset_id(n) for n in queries if n not in OPTIONS}
        assert all(e["parameterName"] == param for e in fields[1:])
        assert f["selection"]["defaultSelection"]["values"]["values"] == [{"value": default}]

    def bar(name):
        """(category field, title), (value field, title), color field, value format, horizontal."""
        spec = widgets[name]["spec"]
        assert spec["widgetType"] == "bar"
        e = spec["encodings"]
        horizontal = e["y"]["scale"]["type"] == "categorical"
        cat, val = (e["y"], e["x"]) if horizontal else (e["x"], e["y"])
        assert (cat["scale"]["type"], val["scale"]["type"]) == ("categorical", "quantitative")
        return ((cat["fieldName"], cat["displayName"]), (val["fieldName"], val["displayName"]),
                e.get("color", {}).get("fieldName"), val.get("format", {}).get("type"), horizontal)
    assert bar("first_try") == (("mode", "Mode"), ("first_try_rate", "First-try pass rate"), None, "number-percent", False)
    assert bar("mistake_mix") == (("verdict", "Mistake"), ("verdict_count", "Mistakes"), "mode", None, False)
    assert bar("hint_by_level") == (("level_label", "Hint level"), ("next_try_pass_rate", "Pass rate"), None,
                                    "number-percent", False)
    # Horizontal, so the full outcome labels fit.
    assert bar("robot_retries") == (("outcome_label", "Outcome"), ("steps", "Robot steps"), None, None, True)
    # Each hint level's label carries its count: "Level 1 (n=10)".
    assert "CONCAT('Level ', hint_level, ' (n=', SUM(n), ')') AS level_label" in "".join(ds["hint_by_level"]["queryLines"])

    # No tile stacks or adds up a rate: rate bars have no color split, tables carry rates as text,
    # and the per-level hint rate is recomputed from passes and n, never a sum of rates.
    for w in widgets.values():
        spec = w.get("spec", {})
        if spec.get("widgetType") == "bar" and bar(w["name"])[3] == "number-percent":
            assert "color" not in spec["encodings"], w["name"]
        for col in spec.get("encodings", {}).get("columns", []):
            assert "%" not in col.get("numberFormat", ""), (w["name"], col["fieldName"])
    hint_sql = "".join(ds["hint_by_level"]["queryLines"])
    assert "SUM(passes) / SUM(n)" in hint_sql and "SUM(rate)" not in hint_sql and "AVG(rate)" not in hint_sql
    with pytest.raises(ValueError):
        _bar("x", "hint_by_level", "t", ("hint_level", "Hint level"), ("next_try_pass_rate", "r"),
             color=("failed_verdict", "Mistake"), percent=True)

    # Layout on the 6-wide grid: no overlaps, every table full width (no sideways scrolling),
    # and the session count big enough to show its row.
    pos = {item["widget"]["name"]: item["position"] for item in page["layout"]}
    cells = {}
    for name, p in pos.items():
        assert 0 <= p["x"] and p["x"] + p["width"] <= 6, name
        for x in range(p["x"], p["x"] + p["width"]):
            for y in range(p["y"], p["y"] + p["height"]):
                assert (x, y) not in cells, (name, cells.get((x, y)))
                cells[(x, y)] = name
    for name in ("hardest_steps", "hint_breakdown", "cell_error_map"):
        assert (pos[name]["x"], pos[name]["width"]) == (0, 6), name
    assert pos["session_count"]["width"] >= 4 and pos["session_count"]["height"] >= 3

    def columns(name):
        return [c["fieldName"] for c in widgets[name]["spec"]["encodings"]["columns"]]
    assert columns("hardest_steps") == ["step_number", "mode", "first_try_rate", "n", "mean_attempts", "n_not_passed"]
    assert columns("hint_breakdown") == ["failed_verdict", "hint_level", "n", "passes", "next_try_pass_rate"]
    assert columns("cell_error_map") == ["mode", "grid_row", "A", "B", "C"]
    assert columns("session_count") == ["provenance", "sessions", "human", "robot"]
    # Table rates and the cell map come from SQL as text, with "no data" when n is 0.
    for q in ("hardest_steps", "hint_breakdown", "cell_error_map_3x3"):
        sql = "".join(ds[q]["queryLines"])
        assert "WHEN n = 0 THEN 'no data'" in sql and "CAST(ROUND(rate * 100) AS INT), '%" in sql
    # Robot outcomes: numbered plain labels, so they sort in order.
    robot_sql = "".join(ds["robot_retries"]["queryLines"])
    assert all(label in robot_sql for label in (
        "'1. First try'", "'2. After 1 retry'", "'3. After 2 retries'", "'4. Stopped'", "'5. Unfinished'"))

    # with_hints_file_vs_without is loaded but has no tile yet (the filters still drive it).
    hints_ds = dataset_id("with_hints_file_vs_without")
    tiles = [w for n, w in widgets.items() if not n.startswith("filter_")]
    assert not any(q["query"]["datasetName"] == hints_ds for w in tiles for q in w.get("queries", []))
    assert query_parameters("WHERE provenance = :provenance") == ["provenance"]
