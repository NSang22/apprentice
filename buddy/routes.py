"""HTTP routes for the Buddy phone app, mounted into the Jetson dashboard (jetson-kushi/dashboard.py).

    GET  /buddy                  the phone app (buddy/static/index.html)
    GET  /api/buddy/state        everything the screen needs; the phone polls it
    POST /api/buddy/profile      {name, avatar}
    POST /api/buddy/start        {game}
    POST /api/buddy/cell         {cell, color|null}   practice mode: the phone is the board
    POST /api/buddy/check        grade the board
    POST /api/buddy/next         next round (or the summary)
    POST /api/buddy/quit         back to the path screen
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


class ProfileBody(BaseModel):
    name: str = ""
    avatar: str = "teal"


class StartBody(BaseModel):
    game: str = "recall"


class CellBody(BaseModel):
    cell: str
    color: Optional[str] = None


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


@router.post("/api/buddy/check")
def check():
    game.check()
    return game.state()


@router.post("/api/buddy/next")
def next_round():
    game.next()
    return game.state()


@router.post("/api/buddy/quit")
def quit_game():
    game.quit()
    return game.state()
