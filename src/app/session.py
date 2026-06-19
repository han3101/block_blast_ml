from __future__ import annotations

from typing import Any, Callable

from engine.block import ALL_BLOCKS, Block
from engine.game import GameState
from engine.generator import Mode
from engine.grid import Grid
from rl.agents.greedy import choose_action as _greedy_choose
from rl.encoding import decode_action

BLOCK_CATALOG: dict[str, Block] = {b.name.upper(): b for b in ALL_BLOCKS}

_policy_cache: dict[str, Callable] = {}


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


class AgentPlaySession:
    def __init__(self) -> None:
        self._mode: Mode = "at_least_one"
        self._agent_name: str = "greedy"
        self._checkpoint: str | None = None
        self.game: GameState = GameState(mode=self._mode)
        self._choose: Callable = _greedy_choose
        self.history: list[dict] = []
        self.total_lines: int = 0

    def configure(self, agent_name: str, checkpoint: str | None, mode: str) -> None:
        valid_modes = ("at_least_one", "random", "solvable")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        if agent_name == "greedy":
            choose_fn: Callable = _greedy_choose
        elif agent_name == "policy":
            if not checkpoint:
                raise ValueError("policy agent requires a checkpoint path")
            if checkpoint not in _policy_cache:
                from rl.agents.policy_agent import make_policy_agent
                _policy_cache[checkpoint] = make_policy_agent(checkpoint, device="cpu")
            choose_fn = _policy_cache[checkpoint]
        else:
            raise ValueError(f"unknown agent: {agent_name!r}")
        self._mode = mode  # type: ignore[assignment]
        self._agent_name = agent_name
        self._checkpoint = checkpoint
        self._choose = choose_fn
        self.game = GameState(mode=self._mode)
        self.history = []
        self.total_lines = 0

    def step(self) -> dict:
        if self.game.game_over:
            return self.state_dict()
        action = self._choose(self.game)
        slot, row, col = decode_action(action)
        block = self.game.hand[slot]
        block_name = block.name if block else "?"
        placed_cells = [[row + dr, col + dc] for dr, dc in block.cells]
        result = self.game.place(slot, row, col)
        score_delta = result.score - result.prev_score
        self.total_lines += result.lines_cleared
        self.history.append({
            "step": len(self.history) + 1,
            "block": block_name,
            "slot": slot,
            "row": row,
            "col": col,
            "lines_cleared": result.lines_cleared,
            "score": result.score,
            "score_delta": score_delta,
            "combo": result.combo,
        })
        return self.state_dict({"last_placed_cells": placed_cells})

    def state_dict(self, extra: dict | None = None) -> dict:
        snap = self.game.snapshot()
        snap["hand_shapes"] = [
            [list(cell) for cell in b.cells] if b is not None else None
            for b in self.game.hand
        ]
        snap["history"] = self.history
        snap["total_lines"] = self.total_lines
        snap["current_combo"] = getattr(self.game._scorer, "combo", 0)
        snap["agent"] = self._agent_name
        snap["mode"] = self._mode
        snap["checkpoint"] = self._checkpoint
        if extra:
            snap.update(extra)
        return snap

    def reset(self) -> dict:
        self.game = GameState(mode=self._mode)
        self.history = []
        self.total_lines = 0
        return self.state_dict()


# TODO (production): per-tab isolation
#   Currently one GridSession and one PlaySession are shared across all browser
#   tabs / clients hitting the same server. Fix: key sessions by a UUID cookie
#   (dict[str, GridSession | PlaySession]), set it on first page load, look it
#   up via a FastAPI dependency on every request. Add a TTL + background cleanup
#   task so idle sessions don't accumulate.
grid_session = GridSession()
play_session = PlaySession()
agent_play_session = AgentPlaySession()
