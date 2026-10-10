"""Serve the monitor locally: python ui/serve.py, then open http://localhost:8000 (F11 for fullscreen).

Serves ui/ at /ui/ and the expert clips in videos/ at /videos/ (SessionState.show_me.video is a
repo-relative path like videos/expert_staircase.mp4). Nothing else in the repo is reachable.
MIME types are fixed here because Windows can map .js to text/plain, which breaks module loading.
Range requests are supported because the browser needs them to seek the show-me clip to its start.
No network needed: three.js is vendored in ui/vendor/.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = ("ui/", "videos/")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/ui"):
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return
        rel = posixpath.normpath(unquote(path)).lstrip("/")
        if path.endswith("/") and not rel.endswith("/"):
            rel += "/"
        if ".." in rel.split("/") or not rel.startswith(ALLOWED):
            self.send_error(404)
            return
        range_header = self.headers.get("Range")
        if range_header and self._send_range(range_header):
            return
        super().do_GET()

    def _send_range(self, header: str) -> bool:
        """Serve one byte range (206). Returns False to fall back to a full 200 response."""
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        start_s, end_s = m.groups()
        if start_s:
            start, end = int(start_s), int(end_s) if end_s else size - 1
        elif end_s:
            start, end = max(0, size - int(end_s)), size - 1
        else:
            return False
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break  # the browser cancels range reads all the time while seeking
                remaining -= len(chunk)
        return True

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # a reload always picks up the latest ui files
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        pass  # quiet: the demo terminal stays readable


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(REPO_ROOT)))
    print(f"[ui] monitor on http://localhost:{args.port}  (brain SessionState expected on ws :8766)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
