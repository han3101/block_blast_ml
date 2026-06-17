from __future__ import annotations

import pathlib
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine.block import ALL_BLOCKS, Block
from engine.grid import Grid

app = FastAPI(title="Block Blast Manual Grid")

_BLOCK_CATALOG: dict[str, Block] = {b.name.upper(): b for b in ALL_BLOCKS}
_static = pathlib.Path(__file__).parent / "static"

_grid = Grid()
_history: list[dict] = []


@app.get("/manual_grid", response_class=HTMLResponse)
async def manual_grid_page() -> HTMLResponse:
    return HTMLResponse((_static / "index.html").read_text())


@app.get("/api/state")
async def get_state() -> dict:
    return {
        "grid": _grid.to_matrix(),
        "size": _grid.size,
        "history": _history,
        "blocks": sorted(_BLOCK_CATALOG.keys()),
    }


class PlaceRequest(BaseModel):
    block: str
    row: int
    col: int


@app.post("/api/place")
async def place_block(req: PlaceRequest) -> dict:
    block_key = req.block.upper()
    if block_key not in _BLOCK_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown block: {req.block!r}")
    block = _BLOCK_CATALOG[block_key]
    try:
        _grid.place(block, req.row, req.col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _history.append({"block": block.name, "row": req.row, "col": req.col})
    return {"grid": _grid.to_matrix(), "history": _history}


@app.post("/api/clear")
async def clear_lines() -> dict:
    lines = _grid.clear_full_lines()
    if _history:
        _history[-1]["lines_cleared"] = lines
    return {"grid": _grid.to_matrix(), "lines_cleared": lines, "history": _history}


@app.post("/api/reset")
async def reset() -> dict:
    global _grid, _history
    _grid = Grid()
    _history = []
    return {"grid": _grid.to_matrix(), "history": _history}


@app.get("/api/preview")
async def preview(
    block: Annotated[str, Query()],
    row: Annotated[int, Query()],
    col: Annotated[int, Query()],
) -> dict:
    block_key = block.upper()
    if block_key not in _BLOCK_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown block: {block!r}")
    b = _BLOCK_CATALOG[block_key]
    cells = [[row + dr, col + dc] for dr, dc in b.cells]
    valid = _grid.can_place(b, row, col)
    return {"cells": cells, "valid": valid}
