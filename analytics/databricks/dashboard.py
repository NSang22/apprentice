"""Create or update the "Apprentice analytics" AI/BI (Lakeview) dashboard through the REST API.

    python -m analytics.databricks.dashboard --dry-run     # print the payload, no network
    python -m analytics.databricks.dashboard               # create it, or update the one with this name

The datasets are the named queries in dashboard_queries.sql, each with a :provenance parameter
(default "synthetic"). The dashboard stays a draft: this never publishes it. Run it by hand,
after sessions; nothing in the live session loop imports this. Standard library only. Uses
DATABRICKS_HOST and DATABRICKS_TOKEN from the environment or .env, and never prints the token.
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
from shared.config import REPO_ROOT

NAME = "Apprentice analytics"
QUERIES_FILE = Path(__file__).with_name("dashboard_queries.sql")
DEFAULT_PROVENANCE = "synthetic"
BANNER = "Showing synthetic data from a simulated student. These are not real results."
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


def _dataset(query_name: str, sql: str) -> dict[str, Any]:
    lines = sql.split("\n")
    return {
        "name": dataset_id(query_name),
        "displayName": query_name,
        "queryLines": [line + "\n" for line in lines[:-1]] + lines[-1:],
        "parameters": [{
            "displayName": "provenance",
            "keyword": "provenance",
            "dataType": "STRING",
            "defaultSelection": {"values": {"dataType": "STRING", "values": [{"value": DEFAULT_PROVENANCE}]}},
        }],
    }


def _fields(names: list[str]) -> list[dict[str, str]]:
    return [{"name": n, "expression": f"`{n}`"} for n in names]


def _bar(name: str, query: str, title: str, x: str, y: str, color: Optional[str] = None,
         y_percent: bool = False) -> dict[str, Any]:
    enc: dict[str, Any] = {
        "x": {"fieldName": x, "scale": {"type": "categorical"}, "displayName": x},
        "y": {"fieldName": y, "scale": {"type": "quantitative"}, "displayName": y, **({"format": PERCENT} if y_percent else {})},
    }
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    names = [x, y] + ([color] if color and color not in (x, y) else [])
    return {
        "name": name,
        "queries": [{"name": "main_query", "query": {"datasetName": dataset_id(query), "fields": _fields(names),
                                                     "disaggregated": True}}],
        "spec": {"version": 3, "widgetType": "bar", "encodings": enc, "frame": {"showTitle": True, "title": title}},
    }


def _column(field: str, order: int, kind: str, percent: bool = False) -> dict[str, Any]:
    number_format = {"integer": "0", "float": "0.00%" if percent else "0.00"}.get(kind)
    col = {
        "fieldName": field, "displayName": field, "title": field, "order": order, "visible": True,
        "type": kind, "displayAs": "number" if kind in ("integer", "float") else "string",
        "alignContent": "right" if kind in ("integer", "float") else "left",
        "booleanValues": ["false", "true"], "imageUrlTemplate": "{{ @ }}", "imageTitleTemplate": "{{ @ }}",
        "imageWidth": "", "imageHeight": "", "linkUrlTemplate": "{{ @ }}", "linkTextTemplate": "{{ @ }}",
        "linkTitleTemplate": "{{ @ }}", "linkOpenInNewTab": True, "allowSearch": False, "allowHTML": False,
        "highlightLinks": False, "useMonospaceFont": False, "preserveWhitespace": False,
    }
    if number_format:
        col["numberFormat"] = number_format
    return col


def _table(name: str, query: str, title: str, columns: list[tuple[str, str, bool]]) -> dict[str, Any]:
    """columns: (field, "string" | "integer" | "float", is a rate)."""
    return {
        "name": name,
        "queries": [{"name": "main_query", "query": {"datasetName": dataset_id(query),
                                                     "fields": _fields([c[0] for c in columns]), "disaggregated": True}}],
        "spec": {
            "version": 1, "widgetType": "table",
            "encodings": {"columns": [_column(f, 100000 + i, kind, pct) for i, (f, kind, pct) in enumerate(columns)]},
            "invisibleColumns": [], "allowHTMLByDefault": False, "itemsPerPage": 25, "paginationSize": "default",
            "condensed": True, "withRowNumber": False, "frame": {"showTitle": True, "title": title},
        },
    }


def _provenance_filter(query_names: list[str]) -> dict[str, Any]:
    qname = {q: f"parameter_{dataset_id(q)}_provenance" for q in query_names}
    return {
        "name": "filter_provenance",
        "queries": [{"name": qname[q], "query": {"datasetName": dataset_id(q),
                                                 "parameters": [{"name": "provenance", "keyword": "provenance"}],
                                                 "disaggregated": False}} for q in query_names],
        "spec": {
            "version": 2, "widgetType": "filter-single-select",
            "encodings": {"fields": [{"parameterName": "provenance", "queryName": qname[q]} for q in query_names]},
            "frame": {"showTitle": True, "title": "provenance"},
        },
    }


def build_dashboard(queries: dict[str, str]) -> dict[str, Any]:
    """The serialized_dashboard object: every query as a dataset, one page of tiles."""
    tiles = [
        ({"name": "banner", "multilineTextboxSpec": {"lines": [f"**{BANNER}**"]}}, (0, 0, 6, 1)),
        (_provenance_filter(list(queries)), (0, 1, 2, 1)),
        (_bar("first_try", "first_try_human_vs_robot", "First-try pass rate, human vs robot",
              "mode", "first_try_rate", color="mode", y_percent=True), (0, 2, 3, 6)),
        (_bar("robot_retries", "robot_retries", "Robot steps by outcome", "outcome", "steps"), (3, 2, 3, 6)),
        (_bar("mistake_mix", "mistake_mix", "Mistake mix", "verdict", "verdict_count", color="mode"), (0, 8, 3, 6)),
        (_bar("hint_effectiveness", "hint_effectiveness", "Next try passes after a hint",
              "hint_level", "next_try_pass_rate", color="failed_verdict", y_percent=True), (3, 8, 3, 6)),
        (_table("hardest_steps", "hardest_steps", "Hardest steps", [
            ("step_number", "integer", False), ("mode", "string", False), ("first_try_rate", "float", True),
            ("n", "integer", False), ("mean_attempts", "float", False), ("n_not_passed", "integer", False)]),
         (0, 14, 6, 6)),
        (_table("cell_error_map", "cell_error_map_3x3", "Pass rate by cell (rows 1 to 3, columns A to C)", [
            ("mode", "string", False), ("grid_row", "string", False),
            ("a_pass_rate", "float", True), ("a_n", "integer", False),
            ("b_pass_rate", "float", True), ("b_n", "integer", False),
            ("c_pass_rate", "float", True), ("c_n", "integer", False)]),
         (0, 20, 6, 5)),
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
