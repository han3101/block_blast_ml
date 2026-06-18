from __future__ import annotations

import pathlib
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.session import BLOCK_CATALOG, grid_session, play_session

app = FastAPI(title="Block Blast")

_static = pathlib.Path(__file__).parent / "static"


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/manual_grid", response_class=HTMLResponse)
async def manual_grid_page() -> HTMLResponse:
    return HTMLResponse((_static / "manual_grid.html").read_text())


@app.get("/manual_play", response_class=HTMLResponse)
async def manual_play_page() -> HTMLResponse:
    return HTMLResponse((_static / "manual_play.html").read_text())


# ── Manual Grid API ───────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state() -> dict:
    return {
        "grid": grid_session.grid.to_matrix(),
        "size": grid_session.grid.size,
        "history": grid_session.history,
        "blocks": sorted(BLOCK_CATALOG.keys()),
    }


class PlaceRequest(BaseModel):
    block: str
    row: int
    col: int


@app.post("/api/place")
async def place_block(req: PlaceRequest) -> dict:
    block_key = req.block.upper()
    if block_key not in BLOCK_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown block: {req.block!r}")
    block = BLOCK_CATALOG[block_key]
    try:
        grid_session.grid.place(block, req.row, req.col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    grid_session.history.append({"block": block.name, "row": req.row, "col": req.col})
    return {"grid": grid_session.grid.to_matrix(), "history": grid_session.history}


@app.post("/api/clear")
async def clear_lines() -> dict:
    lines = grid_session.grid.clear_full_lines()
    if grid_session.history:
        grid_session.history[-1]["lines_cleared"] = lines
    return {"grid": grid_session.grid.to_matrix(), "lines_cleared": lines, "history": grid_session.history}


@app.post("/api/reset")
async def reset() -> dict:
    grid_session.reset()
    return {"grid": grid_session.grid.to_matrix(), "history": grid_session.history}


@app.get("/api/preview")
async def preview(
    block: Annotated[str, Query()],
    row: Annotated[int, Query()],
    col: Annotated[int, Query()],
) -> dict:
    block_key = block.upper()
    if block_key not in BLOCK_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown block: {block!r}")
    b = BLOCK_CATALOG[block_key]
    cells = [[row + dr, col + dc] for dr, dc in b.cells]
    valid = grid_session.grid.can_place(b, row, col)
    return {"cells": cells, "valid": valid}


# ── Manual Play API ───────────────────────────────────────────────────────────

@app.get("/api/play/state")
async def play_get_state() -> dict:
    return play_session.state_dict()


class PlayPlaceRequest(BaseModel):
    slot: int
    row: int
    col: int


@app.post("/api/play/place")
async def play_place(req: PlayPlaceRequest) -> dict:
    block = play_session.game.hand[req.slot] if req.slot in range(3) else None
    if block is None:
        raise HTTPException(status_code=400, detail=f"slot {req.slot} is empty or invalid")
    placed_name = block.name
    try:
        result = play_session.game.place(req.slot, req.row, req.col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    play_session.history.append({
        "slot": req.slot,
        "block": placed_name,
        "row": req.row,
        "col": req.col,
        "lines_cleared": result.lines_cleared,
        "score": result.score,
    })
    return play_session.state_dict({
        "lines_cleared": result.lines_cleared,
        "cells_placed": result.cells_placed,
    })


@app.post("/api/play/reset")
async def play_reset() -> dict:
    play_session.reset()
    return play_session.state_dict()


@app.get("/api/play/preview")
async def play_preview(
    slot: Annotated[int, Query()],
    row: Annotated[int, Query()],
    col: Annotated[int, Query()],
) -> dict:
    if slot not in range(3):
        raise HTTPException(status_code=400, detail=f"slot must be 0–2; got {slot!r}")
    block = play_session.game.hand[slot]
    if block is None:
        raise HTTPException(status_code=400, detail=f"slot {slot} is empty")
    cells = [[row + dr, col + dc] for dr, dc in block.cells]
    valid = play_session.game.grid.can_place(block, row, col)
    return {"cells": cells, "valid": valid}
