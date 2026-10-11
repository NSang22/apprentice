"""HTTP routes for the Buddy phone app, mounted into the Jetson dashboard (jetson-kushi/dashboard.py).

    GET  /buddy                  the phone app (buddy/static/index.html)
    GET  /api/buddy/state        everything the screen needs; the phone polls it
    POST /api/buddy/profile      {name, avatar}
    POST /api/buddy/settings     {difficulty, build_s, source}
    POST /api/buddy/start        {game}
    POST /api/buddy/cell         {cell, color|null}   practice mode: the phone is the board
    POST /api/buddy/choose       {kind: "hint"|"fix"}
    POST /api/buddy/done         finished building early: stop the timer
    POST /api/buddy/rescan       scan the table again
    POST /api/buddy/next         next round (or the summary)
    POST /api/buddy/quit         back to the game picker

The dashboard runs table_task() for the camera board (perception ws :8765).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from buddy.session import BuddySession

STATIC = Path(__file__).resolve().parent / "static"

router = APIRouter()
game = BuddySession()


def table_task():
    """Coroutine for the dashboard's event loop: keeps the camera board up to date."""
    return game.table.run()


class ProfileBody(BaseModel):
    name: str = ""
    avatar: str = "teal"


class SettingsBody(BaseModel):
    difficulty: Optional[str] = None
    build_s: Optional[int] = None
    source: Optional[str] = None


class StartBody(BaseModel):
    game: str = "recall"


class CellBody(BaseModel):
    cell: str
    color: Optional[str] = None


class ChooseBody(BaseModel):
    kind: str


@router.get("/buddy")
def buddy_page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/api/buddy/state")
def state():
    return game.state()


@router.post("/api/buddy/profile")
def profile(body: ProfileBody):
    game.set_profile(body.name, body.avatar)
    return game.state()


@router.post("/api/buddy/settings")
def settings(body: SettingsBody):
    game.set_settings(body.difficulty, body.build_s, body.source)
    return game.state()


@router.post("/api/buddy/start")
def start(body: StartBody):
    try:
        game.start(body.game)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return game.state()


@router.post("/api/buddy/cell")
def cell(body: CellBody):
    game.set_cell(body.cell, body.color)
    return game.state()


@router.post("/api/buddy/choose")
def choose(body: ChooseBody):
    game.choose(body.kind)
    return game.state()


@router.post("/api/buddy/done")
def done():
    game.done()
    return game.state()


@router.post("/api/buddy/rescan")
def rescan():
    game.rescan()
    return game.state()


@router.post("/api/buddy/next")
def next_round():
    game.next()
    return game.state()


@router.post("/api/buddy/quit")
def quit_game():
    game.quit()
    return game.state()
