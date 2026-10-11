"""Copy journals up to the Databricks volume and hints files back down. Never part of a live session.

    python -m analytics.databricks.sync push                       # new or grown journals -> volume
    python -m analytics.databricks.sync push --dry-run             # list what would upload
    python -m analytics.databricks.sync pull-hints --lesson staircase-v1

Uses the Databricks Files API (PUT and GET on /api/2.0/fs/files/Volumes/...) with
DATABRICKS_HOST and DATABRICKS_TOKEN from the environment or the repo's .env. The token is
only ever put in the Authorization header; nothing here prints it. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from shared.config import LESSONS_DIR, REPO_ROOT, SESSIONS_DIR

JOURNALS = "/Volumes/workspace/apprentice/journals"
EXPORTS = "/Volumes/workspace/apprentice/exports"
TIMEOUT_S = 60

MANUAL = """DATABRICKS_HOST and DATABRICKS_TOKEN are not set (environment or .env). To do this by hand:
  push:       In Databricks, open Catalog > workspace > apprentice > journals and click
              "Upload to this volume". Pick the .jsonl files from data/sessions/.
  pull-hints: Open Catalog > workspace > apprentice > exports, download <lesson_id>.hints.json,
              and save it as lessons/<lesson_id>.hints.json (do not edit it; the sha256 must match)."""


class SyncError(RuntimeError):
    pass


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def credentials(env_file: Path) -> Optional[tuple[str, str]]:
    """(host, token) from the environment, falling back to env_file; None if either is missing."""
    file_vars = read_env_file(env_file)
    host = os.environ.get("DATABRICKS_HOST") or file_vars.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN") or file_vars.get("DATABRICKS_TOKEN")
    if not host or not token:
        return None
    host = host.rstrip("/")
    if not host.startswith("https://"):
        host = "https://" + host.removeprefix("http://")
    return host, token


class Files:
    """The few Files API calls we need. Errors come back as SyncError with no secrets in them."""

    def __init__(self, host: str, token: str):
        self._host = host
        self._token = token

    def _call(self, method: str, api_path: str, data: Optional[bytes] = None, query: str = "") -> bytes:
        url = self._host + urllib.parse.quote(api_path) + (f"?{query}" if query else "")
        headers = {"Authorization": f"Bearer {self._token}"}
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise SyncError(f"{method} {api_path}: HTTP {e.code} {body}") from None
        except urllib.error.URLError as e:
            raise SyncError(f"{method} {api_path}: {e.reason}") from None

    def sizes(self, directory: str) -> dict[str, int]:
        """{file name: size} for the files directly in a volume directory."""
        out, token = {}, None
        while True:
            query = urllib.parse.urlencode({"page_token": token}) if token else ""
            page = json.loads(self._call("GET", f"/api/2.0/fs/directories{directory}", query=query) or b"{}")
            for item in page.get("contents", []):
                if not item.get("is_directory"):
                    out[item.get("name") or item["path"].rsplit("/", 1)[-1]] = int(item.get("file_size") or 0)
            token = page.get("next_page_token")
            if not token:
                return out

    def put(self, path: str, data: bytes, overwrite: bool) -> None:
        self._call("PUT", f"/api/2.0/fs/files{path}", data=data, query=f"overwrite={'true' if overwrite else 'false'}")

    def get(self, path: str) -> bytes:
        return self._call("GET", f"/api/2.0/fs/files{path}")


def push(api: Files, sessions_dir: Path, volume: str, dry_run: bool) -> int:
    """Upload journals the volume lacks, and journals that grew since the last push (journals only
    ever append, so a bigger local file is newer). Never shrinks a file on the volume."""
    remote = api.sizes(volume)
    uploaded = 0
    for path in sorted(sessions_dir.glob("*.jsonl")):
        size = path.stat().st_size
        have = remote.get(path.name)
        if have is not None and have >= size:
            if have > size:
                print(f"skip {path.name}: the volume copy is bigger ({have} > {size} bytes)")
            continue
        action = "new" if have is None else f"grew {have} -> {size} bytes"
        if not dry_run:
            api.put(f"{volume}/{path.name}", path.read_bytes(), overwrite=have is not None)
        print(f"{'would upload' if dry_run else 'uploaded'} {path.name} ({action})")
        uploaded += 1
    print(f"{uploaded} journal(s) {'to upload' if dry_run else 'uploaded'}, {len(remote)} already on {volume}")
    return uploaded


def pull_hints(api: Files, lesson_id: str, exports: str, lessons_dir: Path) -> Path:
    """Download exports/<lesson_id>.hints.json byte for byte, so its sha256 matches hint_policy."""
    name = f"{lesson_id}.hints.json"
    data = api.get(f"{exports}/{name}")
    hints = json.loads(data)
    if hints.get("lesson_id") != lesson_id or not isinstance(hints.get("steps"), list):
        raise SyncError(f"{name} does not look like a hints file for {lesson_id}")
    out = lessons_dir / name
    tmp = out.with_name(name + ".tmp")
    lessons_dir.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(hints['steps'])} steps)")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload new or grown journals")
    p.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    p.add_argument("--volume", default=JOURNALS)
    p.add_argument("--dry-run", action="store_true")
    h = sub.add_parser("pull-hints", help="download one lesson's hints file")
    h.add_argument("--lesson", required=True)
    h.add_argument("--exports", default=EXPORTS)
    h.add_argument("--lessons-dir", type=Path, default=LESSONS_DIR)
    args = ap.parse_args(argv)

    creds = credentials(args.env_file)
    if creds is None:
        print(MANUAL)
        return
    api = Files(*creds)
    try:
        if args.cmd == "push":
            push(api, args.sessions_dir, args.volume, args.dry_run)
        else:
            pull_hints(api, args.lesson, args.exports, args.lessons_dir)
    except SyncError as e:
        print(f"sync failed: {e}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
