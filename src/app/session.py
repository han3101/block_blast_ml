from __future__ import annotations

from engine.block import ALL_BLOCKS, Block
from engine.game import GameState
from engine.grid import Grid

BLOCK_CATALOG: dict[str, Block] = {b.name.upper(): b for b in ALL_BLOCKS}


class GridSession:
    def __init__(self) -> None:
        self.grid = Grid()
        self.history: list[dict] = []

    def reset(self) -> None:
        self.grid = Grid()
        self.history = []


class PlaySession:
    def __init__(self) -> None:
        self.game = GameState()
        self.history: list[dict] = []

    def reset(self) -> None:
        self.game.reset()
        self.history = []

    def state_dict(self, extra: dict | None = None) -> dict:
        snap = self.game.snapshot()
        snap["hand_shapes"] = [
            [list(cell) for cell in b.cells] if b is not None else None
            for b in self.game.hand
        ]
        snap["history"] = self.history
        if extra:
            snap.update(extra)
        return snap


# TODO (production): per-tab isolation
#   Currently one GridSession and one PlaySession are shared across all browser
#   tabs / clients hitting the same server. Fix: key sessions by a UUID cookie
#   (dict[str, GridSession | PlaySession]), set it on first page load, look it
#   up via a FastAPI dependency on every request. Add a TTL + background cleanup
#   task so idle sessions don't accumulate.
grid_session = GridSession()
play_session = PlaySession()
