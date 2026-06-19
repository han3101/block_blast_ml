from __future__ import annotations

import random
from typing import Literal

from engine.block import ALL_BLOCKS, Block
from engine.grid import Grid


Mode = Literal["at_least_one", "random", "solvable"]

HAND_SIZE = 3


class HandGenerator:
    """Deals hands of blocks using a seeded RNG.

    Modes:
    - ``random``: purely random — each of the 3 pieces is drawn independently
      from the pool with no regard for the board state. Can deal a hand where
      nothing fits, ending the game immediately.
    - ``at_least_one`` (default): same random draw, but guarantees that at
      least one piece in the hand fits somewhere on the current board. The
      other two are still random and may not fit. Mid-hand dead-ends are still
      possible (place piece 1, then pieces 2 & 3 no longer fit). Returns None
      when nothing in the pool fits anywhere — the caller should treat this as
      game-over.
    - ``solvable``: greedily guarantees all 3 pieces are placeable in
      sequence — piece 1 fits on the current board, piece 2 fits after piece 1
      is placed, piece 3 fits after pieces 1 and 2 are placed. Returns None if
      full solvability can't be achieved at any point in the sequence. Makes
      the game easier; intended for testing or training warm-starts.
    """

    def __init__(
        self,
        seed: int | None = None,
        pool: tuple[Block, ...] = ALL_BLOCKS,
        mode: Mode = "at_least_one",
    ) -> None:
        """Create a generator.

        Args:
            seed: RNG seed for reproducibility. None for non-deterministic play.
            pool: the set of block shapes to draw from.
            mode: hand-generation strategy (see class docstring).
        """
        self._rng = random.Random(seed)
        self._pool = pool
        self._mode = mode

    def clone(self) -> "HandGenerator":
        """Return an independent copy with the RNG positioned identically.

        The cloned generator will deal the *same* future hands as this one
        (pool and mode shared — both immutable; RNG bit-state copied), so a
        search rollout off the clone sees the real next-hand distribution.
        """
        new = HandGenerator.__new__(HandGenerator)
        new._rng = random.Random()
        new._rng.setstate(self._rng.getstate())
        new._pool = self._pool
        new._mode = self._mode
        return new

    def reseed(self, seed: int | None) -> None:
        """Restart the RNG on a fresh stream (search chance-node sampling).

        ``clone()`` copies the bit-state so a clone deals the *same* future
        hands — useful for ``restore()``, but a single realization, not the
        distribution. ``reseed`` lets a search draw an *independent* next hand
        for the same board, so repeated reseed+deal calls Monte-Carlo the real
        marginal next-hand distribution. Pool and mode are unchanged.
        """
        self._rng.seed(seed)

    def deal(self, grid: Grid) -> list[Block] | None:
        """Deal a new hand of blocks for the given board state.

        Returns None if no hand can be constructed (only possible in
        ``at_least_one`` mode when nothing in the pool fits the board).
        The caller is responsible for treating None as game-over.
        """
        if self._mode == "random":
            return self._random()
        elif self._mode == "at_least_one":
            return self._at_least_one(grid)
        elif self._mode == "solvable":
            return self._solvable(grid)
        raise ValueError(f"unknown mode: {self._mode!r}")

    def _random(self) -> list[Block]:
        return self._rng.choices(self._pool, k=HAND_SIZE)

    def _at_least_one(self, grid: Grid) -> list[Block] | None:
        fits = [b for b in self._pool if grid.can_place_anywhere(b)]
        if not fits:
            # Nothing in the pool fits the current board return None to signal game-over
            return None
        playable = self._rng.choice(fits)
        others = self._rng.choices(self._pool, k=HAND_SIZE - 1)
        hand = [playable, *others]
        self._rng.shuffle(hand)
        return hand

    def _solvable(self, grid: Grid) -> list[Block] | None:
        hand: list[Block] = []
        current_grid = Grid(size=grid.size, cells=grid.to_matrix())
        for _ in range(HAND_SIZE):
            fits = [b for b in self._pool if current_grid.can_place_anywhere(b)]
            if not fits:
                # Can't guarantee all 3 pieces are placeable in sequence —
                # return None rather than padding with unplayable random pieces.
                return None
            pick = self._rng.choice(fits)
            hand.append(pick)
            row, col = next(current_grid.placements(pick))
            current_grid.place(pick, row, col)
        self._rng.shuffle(hand)
        return hand
