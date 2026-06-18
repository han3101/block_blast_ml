"""Gymnasium environment wrapping GameState for RL training."""
from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from engine.block import ALL_BLOCKS, Block
from engine.game import GameState
from engine.generator import Mode
from rl.encoding import NUM_ACTIONS, GRID_SIZE, NUM_SLOTS, decode_action, encode_obs, action_mask

_OBS_SHAPE = (NUM_SLOTS + 1, GRID_SIZE, GRID_SIZE)  # (4, 8, 8)


class BlockBlastEnv(gymnasium.Env):
    """Gymnasium wrapper around GameState.

    Observation: (4, 8, 8) float32 — channel 0 is board occupancy,
      channels 1-3 are hand slot planes (slot i → channel i+1).
    Action: Discrete(192) — decoded as (slot, row, col) via rl.encoding.
    Reward: score delta per step, with optional additive shaping.
    Info: always contains 'action_mask', a (192,) bool array.

    Illegal actions (not in the mask) propagate as ValueError from
    GameState.place — callers are expected to respect the mask.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: int | None = None,
        mode: Mode = "at_least_one",
        pool: tuple[Block, ...] = ALL_BLOCKS,
        line_clear_bonus: float = 0.0,
        game_over_penalty: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._mode: Mode = mode
        self._pool = pool
        self._line_clear_bonus = line_clear_bonus
        self._game_over_penalty = game_over_penalty
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=_OBS_SHAPE, dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Pre-reset state; training loops always call reset() before step().
        self._state = GameState(seed=seed, pool=pool, mode=mode)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = GameState(seed=seed, pool=self._pool, mode=self._mode)
        return self._obs(), {"action_mask": self._mask()}

    def step(self, action: int):
        if self._state.game_over:
            raise RuntimeError("env is terminated — call reset() before stepping")

        slot, row, col = decode_action(int(action))
        result = self._state.place(slot, row, col)  # raises ValueError on illegal action

        reward = float(result.score - result.prev_score)
        if self._line_clear_bonus and result.lines_cleared:
            reward += self._line_clear_bonus * result.lines_cleared
        if self._game_over_penalty and result.game_over:
            reward -= self._game_over_penalty

        return self._obs(), reward, result.game_over, False, {"action_mask": self._mask()}

    def render(self):
        if self.render_mode != "ansi":
            return None
        snap = self._state.snapshot()
        rows = [" ".join("X" if c else "." for c in row) for row in snap["grid"]]
        hand = ", ".join(str(b) if b else "None" for b in snap["hand"])
        rows.append(f"Hand: [{hand}]  Score: {snap['score']}")
        return "\n".join(rows)

    # --- helpers ---

    def _obs(self) -> np.ndarray:
        return np.array(encode_obs(self._state), dtype=np.float32)

    def _mask(self) -> np.ndarray:
        return np.array(action_mask(self._state), dtype=bool)
