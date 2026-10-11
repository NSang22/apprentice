import json

from analytics.databricks.dashboard import BANNER, QUERIES_FILE, build_dashboard, dataset_id, parse_queries


def test_payload_has_seven_datasets_and_the_tiles():
    queries = parse_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    d = json.loads(json.dumps(build_dashboard(queries)))  # valid JSON all the way through

    assert len(d["datasets"]) == 7
    for ds in d["datasets"]:
        sql = "".join(ds["queryLines"])
        assert ":provenance" in sql and not sql.rstrip().endswith(";") and "-- name:" not in sql
        (p,) = ds["parameters"]
        assert p["keyword"] == "provenance"
        assert p["defaultSelection"]["values"]["values"] == [{"value": "synthetic"}]

    (page,) = d["pages"]
    widgets = {item["widget"]["name"]: item["widget"] for item in page["layout"]}
    assert widgets["banner"]["multilineTextboxSpec"]["lines"] == [f"**{BANNER}**"]

    def bar(name):
        spec = widgets[name]["spec"]
        assert spec["widgetType"] == "bar"
        e = spec["encodings"]
        return e["x"]["fieldName"], e["y"]["fieldName"], e.get("color", {}).get("fieldName"), e["y"].get("format", {}).get("type")
    assert bar("first_try") == ("mode", "first_try_rate", "mode", "number-percent")
    assert bar("mistake_mix") == ("verdict", "verdict_count", "mode", None)
    assert bar("hint_effectiveness") == ("hint_level", "next_try_pass_rate", "failed_verdict", "number-percent")
    assert bar("robot_retries") == ("outcome", "steps", None, None)

    def table(name):
        cols = widgets[name]["spec"]["encodings"]["columns"]
        return [c["fieldName"] for c in cols], {c["fieldName"] for c in cols if c.get("numberFormat", "").endswith("%")}
    assert table("hardest_steps") == (
        ["step_number", "mode", "first_try_rate", "n", "mean_attempts", "n_not_passed"], {"first_try_rate"})
    assert table("cell_error_map") == (
        ["mode", "grid_row", "a_pass_rate", "a_n", "b_pass_rate", "b_n", "c_pass_rate", "c_n"],
        {"a_pass_rate", "b_pass_rate", "c_pass_rate"})

    # The filter drives the provenance parameter of all seven datasets.
    f = widgets["filter_provenance"]
    assert f["spec"]["widgetType"] == "filter-single-select"
    assert {q["query"]["datasetName"] for q in f["queries"]} == {ds["name"] for ds in d["datasets"]}
    assert all(e["parameterName"] == "provenance" for e in f["spec"]["encodings"]["fields"])

    # with_hints_file_vs_without is loaded but has no tile yet.
    hints_ds = dataset_id("with_hints_file_vs_without")
    tiles = [w for n, w in widgets.items() if n != "filter_provenance"]
    assert not any(q["query"]["datasetName"] == hints_ds for w in tiles for q in w.get("queries", []))
