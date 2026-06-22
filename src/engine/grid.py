from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from engine.block import Block, Cell


@lru_cache(maxsize=None)
def _block_geom(cells: tuple[Cell, ...], size: int) -> tuple[int, int, int]:
    """Return ``(base_mask, height, width)`` for a block's normalized cells.

    ``base_mask`` is the block's occupancy bitmask anchored at origin (0, 0)
    under the row-major bit layout ``bit = row * size + col``. Placing the block
    at ``(row, col)`` is then just ``base_mask << (row * size + col)`` — valid
    with no row-wraparound precisely when the bounds check (``col + width <=
    size``) passes, which every caller does first. Cached because the in-play
    blocks are module-level singletons reused across millions of search probes.
    """
    mask = 0
    max_row = max_col = 0
    for row_offset, col_offset in cells:
        mask |= 1 << (row_offset * size + col_offset)
        if row_offset > max_row:
            max_row = row_offset
        if col_offset > max_col:
            max_col = col_offset
    return mask, max_row + 1, max_col + 1


@lru_cache(maxsize=None)
def _anchor_masks(size: int) -> tuple[int, int]:
    """Return ``(row0_mask, col0_mask)`` for a size.

    ``row0_mask`` is the top row (bits ``0..size-1``); ``col0_mask`` is the
    left column (bit ``r*size`` for each row). They double as the two anchor
    grids used by the branchless full-line detection in ``clear_full_lines``.
    """
    row0_mask = (1 << size) - 1
    col0_mask = sum(1 << (row * size) for row in range(size))
    return row0_mask, col0_mask


class Grid:
    """An 8×8 (or custom square) 0/1 grid for Block Blast-style play.

    Backed by a single integer **bitboard** (``self._bits``): cell ``(row, col)``
    is bit ``row * size + col``. This collapses the ``can_place``/``in_bounds``
    inner loop — the search hot path — into a handful of integer ops (shift,
    ``&``), replacing the per-cell Python scan. The public API is unchanged; the
    bitboard is purely an internal fast-path (diff-tested against the original
    list-of-lists engine in ``tests/test_bitboard.py``).
    """

    def __init__(self, size: int = 8, cells: list[list[int]] | None = None) -> None:
        """Create a grid. Pass ``cells`` to initialise from an existing matrix."""
        if size <= 0:
            raise ValueError("grid size must be positive")

        self.size = size
        if cells is None:
            self._bits = 0
        else:
            self._bits = self._bits_from_cells(cells, size)

    @staticmethod
    def _validate_cells(cells: list[list[int]], size: int) -> list[list[int]]:
        if len(cells) != size or any(len(row) != size for row in cells):
            raise ValueError(f"cells must be a {size}x{size} matrix")

        copied = [[int(value) for value in row] for row in cells]
        if any(value not in (0, 1) for row in copied for value in row):
            raise ValueError("grid cells must be 0 or 1")

        return copied

    @staticmethod
    def _bits_from_cells(cells: list[list[int]], size: int) -> int:
        validated = Grid._validate_cells(cells, size)
        bits = 0
        for row_index, row in enumerate(validated):
            row_base = row_index * size
            for col_index, value in enumerate(row):
                if value:
                    bits |= 1 << (row_base + col_index)
        return bits

    def in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid."""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        """Return True if the cell at (row, col) is unoccupied."""
        if not self.in_bounds(row, col):
            raise IndexError(f"cell ({row}, {col}) is out of bounds")
        return not (self._bits >> (row * self.size + col)) & 1

    def can_place(self, block: Block, row: int, col: int) -> bool:
        """Return True if the block can be placed with its origin at (row, col)."""
        size = self.size
        mask, height, width = _block_geom(block.cells, size)
        if row < 0 or col < 0 or row + height > size or col + width > size:
            return False
        return self._bits & (mask << (row * size + col)) == 0

    def place(self, block: Block, row: int, col: int) -> None:
        """Fill the block's cells onto the grid. Raises ValueError if placement is illegal."""
        size = self.size
        mask, height, width = _block_geom(block.cells, size)
        if row < 0 or col < 0 or row + height > size or col + width > size:
            raise ValueError(f"cannot place block {block.name!r} at ({row}, {col})")
        shifted = mask << (row * size + col)
        if self._bits & shifted:
            raise ValueError(f"cannot place block {block.name!r} at ({row}, {col})")
        self._bits |= shifted

    def clear_full_lines(self) -> int:
        """Clear any fully-filled rows and columns and return the total count cleared.

        Branchless detection: AND-reduce ``size`` consecutive bits — horizontally
        (shifts of 1) for full rows, vertically (shifts of ``size``) for full
        columns. After the reduction, the anchor bit of each line (its col-0 /
        row-0 cell) is set iff the whole line was full. The doubling+remainder
        loop ANDs *exactly* ``size`` cells, so it is correct for any grid size.
        Each anchor is then smeared back into a full-line mask by multiplying by
        the orthogonal anchor grid (the windows are disjoint and adjacent, so the
        product is a carry-free OR), and the union is cleared in one ``&=``.
        """
        bits = self._bits
        # Cheap early-out: no line of length ``size`` can be full with fewer than
        # ``size`` occupied cells (the common, mostly-empty search board).
        size = self.size
        if bits.bit_count() < size:
            return 0
        row0_mask, col0_mask = _anchor_masks(size)

        rows = bits  # AND-reduce horizontally → bit r*size set iff row r full
        cols = bits  # AND-reduce vertically   → bit c set iff column c full
        step = 1
        while step * 2 <= size:
            rows &= rows >> step
            cols &= cols >> (step * size)
            step *= 2
        if step < size:
            rest = size - step
            rows &= rows >> rest
            cols &= cols >> (rest * size)

        full_rows = rows & col0_mask  # anchors of full rows (bit at each r*size)
        full_cols = cols & row0_mask  # anchors of full columns (bit at each c)
        # Smear anchors back to whole lines; disjoint windows → product == OR.
        clear = full_rows * row0_mask | full_cols * col0_mask
        self._bits = bits & ~clear
        return full_rows.bit_count() + full_cols.bit_count()

    ### Utility methods

    def placements(self, block: Block) -> Iterator[tuple[int, int]]:
        """Yield every (row, col) origin where the block legally fits on the current grid."""
        size = self.size
        mask, height, width = _block_geom(block.cells, size)
        bits = self._bits
        for row in range(size - height + 1):
            row_base = row * size
            for col in range(size - width + 1):
                if bits & (mask << (row_base + col)) == 0:
                    yield row, col

    def can_place_anywhere(self, block: Block) -> bool:
        """Return True if the block has at least one legal placement on the current grid."""
        return any(True for _ in self.placements(block))

    def count_occupied(self) -> int:
        """Return the number of occupied cells (fast popcount over the bitboard)."""
        return self._bits.bit_count()

    def to_matrix(self) -> list[list[int]]:
        """Return a fresh 0/1 matrix reconstructed from the bitboard."""
        size = self.size
        bits = self._bits
        return [
            [(bits >> (row * size + col)) & 1 for col in range(size)]
            for row in range(size)
        ]

    def clone(self) -> "Grid":
        """Return an independent copy. The bitboard is an immutable int, so this
        just carries the value across — no deep cell copy needed (search hot path)."""
        new = Grid.__new__(Grid)
        new.size = self.size
        new._bits = self._bits
        return new

    def __str__(self) -> str:
        size = self.size
        bits = self._bits
        return "\n".join(
            " ".join(str((bits >> (row * size + col)) & 1) for col in range(size))
            for row in range(size)
        )
