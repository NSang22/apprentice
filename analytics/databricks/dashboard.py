"""Create or update the "Apprentice analytics" AI/BI (Lakeview) dashboard through the REST API.

    python -m analytics.databricks.dashboard --dry-run     # print the payload, no network
    python -m analytics.databricks.dashboard               # create it, or update the one with this name

The datasets are the named queries in dashboard_queries.sql. A query that uses :provenance or
:lesson_id gets that parameter; two single-select filters drive them on every dataset, with
choices from the options_* queries. Defaults: provenance "synthetic", the demo lesson.
The dashboard stays a draft: this never publishes it. Run it by hand, after sessions; nothing
in the live session loop imports this. Standard library only. Uses DATABRICKS_HOST and
DATABRICKS_TOKEN from the environment or .env, and never prints the token.

No tile adds up or stacks a rate: a bar with a rate on its y axis has no color split, and every
data query is filtered to one provenance and one lesson, so each x value is one row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from analytics.databricks.sync import MANUAL, credentials
from brain.settings import DEMO_LESSON
from shared.config import REPO_ROOT

NAME = "Apprentice analytics"
QUERIES_FILE = Path(__file__).with_name("dashboard_queries.sql")
DEFAULT_PROVENANCE = "synthetic"
DEFAULT_LESSON = json.loads(DEMO_LESSON.read_text(encoding="utf-8"))["lesson_id"]
PARAMETERS = {"provenance": DEFAULT_PROVENANCE, "lesson_id": DEFAULT_LESSON}
BANNER = ("Only 'real' comes from live sessions at the table. 'synthetic' is a simulated student "
          "and 'simulation' uses fake inputs. Neither is a result.")
TIMEOUT_S = 60

PERCENT = {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 0}}


# ---------------------------------------------------------------- payload (pure)


def parse_queries(sql: str) -> dict[str, str]:
    """{name: query text} from "-- name: <name>" blocks. Comment lines and the final ; are dropped."""
    out: dict[str, str] = {}
    for block in re.split(r"^-- name: ", sql, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("--")]
        out[name.strip()] = "\n".join(lines).rstrip().rstrip(";")
    return out


def dataset_id(query_name: str) -> str:
    return hashlib.sha256(query_name.encode()).hexdigest()[:8]


def query_parameters(sql: str) -> list[str]:
    """The dashboard parameters a query uses, in PARAMETERS order."""
    return [p for p in PARAMETERS if re.search(rf":{p}\b", sql)]


def _dataset(query_name: str, sql: str) -> dict[str, Any]:
    lines = sql.split("\n")
    out: dict[str, Any] = {
        "name": dataset_id(query_name),
        "displayName": query_name,
        "queryLines": [line + "\n" for line in lines[:-1]] + lines[-1:],
    }
    params = [{
        "displayName": p,
        "keyword": p,
        "dataType": "STRING",
        "defaultSelection": {"values": {"dataType": "STRING", "values": [{"value": PARAMETERS[p]}]}},
    } for p in query_parameters(sql)]
    if params:  # the server drops an empty list, so leave it out and a read-back matches exactly
        out["parameters"] = params
    return out


def _fields(names: list[str]) -> list[dict[str, str]]:
    return [{"name": n, "expression": f"`{n}`"} for n in names]


def _axis(field: str, title: str, kind: str, percent: bool = False) -> dict[str, Any]:
    return {"fieldName": field, "scale": {"type": kind}, "displayName": title, **({"format": PERCENT} if percent else {})}


def _bar(name: str, query: str, title: str, category: tuple[str, str], value: tuple[str, str],
         color: Optional[tuple[str, str]] = None, percent: bool = False, horizontal: bool = False) -> dict[str, Any]:
    """category, value, color: (field, axis title). The category axis is x, or y when horizontal
    (Lakeview draws horizontal bars when the categorical field is on y). A rate value never
    gets a color split: that would stack rates."""
    if percent and color is not None and color[0] != category[0]:
        raise ValueError(f"{name}: a rate bar cannot be split by color; that would stack rates")
    cat_axis = _axis(category[0], category[1], "categorical")
    value_axis = _axis(value[0], value[1], "quantitative", percent)
    enc: dict[str, Any] = {"x": value_axis, "y": cat_axis} if horizontal else {"x": cat_axis, "y": value_axis}
    if color:
        enc["color"] = _axis(color[0], color[1], "categorical")
    names = list(dict.fromkeys([category[0], value[0]] + ([color[0]] if color else [])))
    return {
        "name": name,
        "queries": [{"name": "main_query", "query": {"datasetName": dataset_id(query), "fields": _fields(names),
                                                     "disaggregated": True}}],
        "spec": {"version": 3, "widgetType": "bar", "encodings": enc, "frame": {"showTitle": True, "title": title}},
    }


def _column(field: str, order: int, kind: str) -> dict[str, Any]:
    """kind: "string", "integer", or "float". Rates reach tables as text from SQL ("40%"),
    because the table number format adds a % sign without scaling the value."""
    numeric = kind in ("integer", "float")
    col = {
        "fieldName": field, "displayName": field, "title": field, "order": order, "visible": True,
        "type": kind, "displayAs": "number" if numeric else "string",
        "alignContent": "right" if numeric else "left",
        "booleanValues": ["false", "true"], "imageUrlTemplate": "{{ @ }}", "imageTitleTemplate": "{{ @ }}",
        "imageWidth": "", "imageHeight": "", "linkUrlTemplate": "{{ @ }}", "linkTextTemplate": "{{ @ }}",
        "linkTitleTemplate": "{{ @ }}", "linkOpenInNewTab": True, "allowSearch": False, "allowHTML": False,
        "highlightLinks": False, "useMonospaceFont": False, "preserveWhitespace": False,
    }
    if numeric:
        col["numberFormat"] = "0" if kind == "integer" else "0.00"
    return col


def _table(name: str, query: str, title: str, columns: list[tuple[str, str]]) -> dict[str, Any]:
    """columns: (field, "string" | "integer" | "float")."""
    return {
        "name": name,
        "queries": [{"name": "main_query", "query": {"datasetName": dataset_id(query),
                                                     "fields": _fields([c[0] for c in columns]), "disaggregated": True}}],
        "spec": {
            "version": 1, "widgetType": "table",
            "encodings": {"columns": [_column(f, 100000 + i, kind) for i, (f, kind) in enumerate(columns)]},
            "invisibleColumns": [], "allowHTMLByDefault": False, "itemsPerPage": 25, "paginationSize": "default",
            "condensed": True, "withRowNumber": False, "frame": {"showTitle": True, "title": title},
        },
    }


def _filter(name: str, title: str, param: str, options_query: str, queries: dict[str, str]) -> dict[str, Any]:
    """A single-select dropdown: choices from the `param` column of options_query, the choice
    sent to the `param` parameter of every dataset that uses it."""
    opt = f"{name}_options"
    targets = [q for q, sql in queries.items() if param in query_parameters(sql)]
    pq = {q: f"parameter_{dataset_id(q)}_{param}" for q in targets}
    return {
        "name": name,
        "queries": [{"name": opt, "query": {"datasetName": dataset_id(options_query), "fields": [
            {"name": param, "expression": f"`{param}`"},
            {"name": f"{param}_associativity", "expression": "COUNT_IF(`associative_filter_predicate_group`)"},
        ], "disaggregated": False}}] + [
            {"name": pq[q], "query": {"datasetName": dataset_id(q), "parameters": [{"name": param, "keyword": param}],
                                      "disaggregated": False}} for q in targets],
        "spec": {
            "version": 2, "widgetType": "filter-single-select",
            "encodings": {"fields": [{"fieldName": param, "displayName": title, "queryName": opt}]
                          + [{"parameterName": param, "queryName": pq[q]} for q in targets]},
            "selection": {"defaultSelection": {"values": {"dataType": "STRING", "values": [{"value": PARAMETERS[param]}]}}},
            "frame": {"showTitle": True, "title": title},
        },
    }


def build_dashboard(queries: dict[str, str]) -> dict[str, Any]:
    """The serialized_dashboard object: every query as a dataset, one page of tiles."""
    # Grid is 6 wide. Filters stack on the left; every table is full width so none scrolls sideways.
    tiles = [
        ({"name": "banner", "multilineTextboxSpec": {"lines": [BANNER]}}, (0, 0, 6, 1)),
        (_filter("filter_provenance", "Provenance", "provenance", "options_provenance", queries), (0, 1, 2, 1)),
        (_filter("filter_lesson", "Lesson", "lesson_id", "options_lesson", queries), (0, 2, 2, 1)),
        (_table("session_count", "session_count", "Sessions behind these numbers", [
            ("provenance", "string"), ("sessions", "integer"), ("human", "integer"), ("robot", "integer")]),
         (2, 1, 4, 3)),
        (_bar("first_try", "first_try_human_vs_robot", "First-try pass rate, human vs robot",
              ("mode", "Mode"), ("first_try_rate", "First-try pass rate"), percent=True), (0, 4, 3, 6)),
        (_bar("robot_retries", "robot_retries", "Robot steps by outcome",
              ("outcome_label", "Outcome"), ("steps", "Robot steps"), horizontal=True), (3, 4, 3, 6)),
        (_bar("mistake_mix", "mistake_mix", "Mistake mix",
              ("verdict", "Mistake"), ("verdict_count", "Mistakes"), color=("mode", "Mode")), (0, 10, 3, 6)),
        (_bar("hint_by_level", "hint_by_level", "Next try passes, by hint level",
              ("level_label", "Hint level"), ("next_try_pass_rate", "Pass rate"), percent=True), (3, 10, 3, 6)),
        (_table("hardest_steps", "hardest_steps", "Hardest steps", [
            ("step_number", "integer"), ("mode", "string"), ("first_try_rate", "string"),
            ("n", "integer"), ("mean_attempts", "float"), ("n_not_passed", "integer")]),
         (0, 16, 6, 6)),
        (_table("hint_breakdown", "hint_breakdown", "Next try passes, by mistake and hint level", [
            ("failed_verdict", "string"), ("hint_level", "integer"), ("n", "integer"), ("passes", "integer"),
            ("next_try_pass_rate", "string")]),
         (0, 22, 6, 6)),
        (_table("cell_error_map", "cell_error_map_3x3", "Pass rate by cell (rows 1 to 3, columns A to C)", [
            ("mode", "string"), ("grid_row", "string"), ("A", "string"), ("B", "string"), ("C", "string")]),
         (0, 28, 6, 6)),
    ]
    return {
        "datasets": [_dataset(name, sql) for name, sql in queries.items()],
        "pages": [{
            "name": "overview",
            "displayName": NAME,
            "pageType": "PAGE_TYPE_CANVAS",
            "layout": [{"widget": w, "position": {"x": x, "y": y, "width": wd, "height": h}} for w, (x, y, wd, h) in tiles],
        }],
    }


# ---------------------------------------------------------------- API


class ApiError(RuntimeError):
    pass


class Api:
    def __init__(self, host: str, token: str):
        self._host, self._token = host, token

    def call(self, method: str, path: str, body: Optional[dict] = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self._token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._host + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise ApiError(f"{method} {path}: HTTP {e.code} {e.read().decode('utf-8', errors='replace')[:2000]}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"{method} {path}: {e.reason}") from None

    def warehouse_id(self) -> str:
        warehouses = self.call("GET", "/api/2.0/sql/warehouses").get("warehouses", [])
        if not warehouses:
            raise ApiError("no SQL warehouse in this workspace")
        running = [w for w in warehouses if w.get("state") == "RUNNING"]
        return (running or warehouses)[0]["id"]

    def find_dashboard(self, name: str) -> Optional[dict[str, Any]]:
        token = None
        while True:
            q = {"page_size": 100, **({"page_token": token} if token else {})}
            page = self.call("GET", "/api/2.0/lakeview/dashboards?" + urllib.parse.urlencode(q))
            for d in page.get("dashboards", []):
                if d.get("display_name") == name and d.get("lifecycle_state", "ACTIVE") == "ACTIVE":
                    return d
            token = page.get("next_page_token")
            if not token:
                return None


def push(api: Api, serialized: dict[str, Any]) -> tuple[str, str]:
    """(dashboard_id, "created" | "updated")."""
    body = {"display_name": NAME, "warehouse_id": api.warehouse_id(), "serialized_dashboard": json.dumps(serialized)}
    existing = api.find_dashboard(NAME)
    if existing is None:
        return api.call("POST", "/api/2.0/lakeview/dashboards", body)["dashboard_id"], "created"
    did = existing["dashboard_id"]
    api.call("PATCH", f"/api/2.0/lakeview/dashboards/{did}", body)
    return did, "updated"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the payload and make no network call")
    ap.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    args = ap.parse_args(argv)

    serialized = build_dashboard(parse_queries(QUERIES_FILE.read_text(encoding="utf-8")))
    if args.dry_run:
        print(json.dumps(serialized, indent=2))
        return
    creds = credentials(args.env_file)
    if creds is None:
        print(MANUAL.split("\n")[0])
        return
    host = creds[0]
    try:
        did, action = push(Api(*creds), serialized)
    except ApiError as e:
        print(f"dashboard failed: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"{action} draft dashboard {NAME!r} (not published)")
    print(f"{host}/sql/dashboardsv3/{did}")


if __name__ == "__main__":
    main()
