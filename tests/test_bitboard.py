"""Diff-test the bitboard ``Grid`` against an independent list-of-lists oracle.

The plan (plans/phase-3.md) requires the bitboard fast-path to be "diff-tested
against the pure engine". ``RefGrid`` below re-implements the *original*
list-of-lists algorithms (the engine before the bitboard rewrite) as an
oracle, and the fuzz tests assert the real ``Grid`` matches it cell-for-cell
across randomized boards, blocks, and origins — including out-of-bounds and
overlapping placements, and full row/column clears.
"""
from __future__ import annotations

import random

import pytest

from engine.block import ALL_BLOCKS, Block
from engine.grid import Grid


class RefGrid:
    """The original list-of-lists implementation, kept as a diff oracle."""

    def __init__(self, size: int, cells: list[list[int]]) -> None:
        self.size = size
        self._cells = [row.copy() for row in cells]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def can_place(self, block: Block, row: int, col: int) -> bool:
        for dr, dc in block.cells:
            tr, tc = row + dr, col + dc
            if not self.in_bounds(tr, tc):
                return False
            if self._cells[tr][tc] == 1:
                return False
        return True

    def place(self, block: Block, row: int, col: int) -> None:
        for dr, dc in block.cells:
            self._cells[row + dr][col + dc] = 1

    def clear_full_lines(self) -> int:
        full_rows = {r for r in range(self.size) if all(self._cells[r])}
        full_cols = {
            c for c in range(self.size)
            if all(self._cells[r][c] for r in range(self.size))
        }
        for r in full_rows:
            for c in range(self.size):
                self._cells[r][c] = 0
        for c in full_cols:
            for r in range(self.size):
                self._cells[r][c] = 0
        return len(full_rows) + len(full_cols)

    def placements(self, block: Block) -> list[tuple[int, int]]:
        out = []
        for r in range(self.size):
            for c in range(self.size):
                if self.can_place(block, r, c):
                    out.append((r, c))
        return out

    def to_matrix(self) -> list[list[int]]:
        return [row.copy() for row in self._cells]


def _random_cells(size: int, rng: random.Random, density: float) -> list[list[int]]:
    return [[1 if rng.random() < density else 0 for _ in range(size)] for _ in range(size)]


@pytest.mark.parametrize("size", [4, 8])
def test_fuzz_matches_reference(size: int) -> None:
    rng = random.Random(1234 + size)
    for _ in range(400):
        cells = _random_cells(size, rng, density=rng.choice([0.0, 0.2, 0.5, 0.85]))
        grid = Grid(size=size, cells=cells)
        ref = RefGrid(size, cells)

        block = rng.choice([b for b in ALL_BLOCKS if b.height <= size and b.width <= size])

        # placements() — identical set *and* order (generator.py relies on the first).
        assert list(grid.placements(block)) == ref.placements(block)
        assert grid.can_place_anywhere(block) == bool(ref.placements(block))

        # can_place over the full origin range, including out-of-bounds origins.
        for r in range(-1, size + 1):
            for c in range(-1, size + 1):
                assert grid.can_place(block, r, c) == ref.can_place(block, r, c), (r, c)

        # is_empty / occupancy match everywhere in bounds.
        for r in range(size):
            for c in range(size):
                assert grid.is_empty(r, c) == (ref._cells[r][c] == 0)
        assert grid.count_occupied() == sum(v for row in cells for v in row)
        assert grid.to_matrix() == ref.to_matrix()


def test_fuzz_place_and_clear_match_reference() -> None:
    size = 8
    rng = random.Random(99)
    for _ in range(300):
        cells = _random_cells(size, rng, density=rng.choice([0.0, 0.3, 0.6]))
        grid = Grid(size=size, cells=cells)
        ref = RefGrid(size, cells)
        block = rng.choice([b for b in ALL_BLOCKS if b.height <= size and b.width <= size])

        legal = ref.placements(block)
        if not legal:
            continue
        r, c = rng.choice(legal)
        grid.place(block, r, c)
        ref.place(block, r, c)
        assert grid.to_matrix() == ref.to_matrix()

        cleared_grid = grid.clear_full_lines()
        cleared_ref = ref.clear_full_lines()
        assert cleared_grid == cleared_ref
        assert grid.to_matrix() == ref.to_matrix()


def test_fuzz_illegal_place_raises_consistently() -> None:
    size = 8
    rng = random.Random(7)
    for _ in range(200):
        cells = _random_cells(size, rng, density=0.6)
        grid = Grid(size=size, cells=cells)
        ref = RefGrid(size, cells)
        block = rng.choice(ALL_BLOCKS)
        r = rng.randint(-1, size)
        c = rng.randint(-1, size)
        if ref.can_place(block, r, c):
            grid.place(block, r, c)  # should not raise
        else:
            with pytest.raises(ValueError, match="cannot place"):
                grid.place(block, r, c)
