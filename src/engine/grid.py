from __future__ import annotations

from typing import Iterator

from engine.block import Block


class Grid:
    """An 8×8 (or custom square) 0/1 grid for Block Blast-style play."""

    def __init__(self, size: int = 8, cells: list[list[int]] | None = None) -> None:
        """Create a grid. Pass ``cells`` to initialise from an existing matrix."""
        if size <= 0:
            raise ValueError("grid size must be positive")

        self.size = size
        if cells is None:
            self._cells = [[0 for _ in range(size)] for _ in range(size)]
        else:
            self._cells = self._validate_cells(cells, size)

    @staticmethod
    def _validate_cells(cells: list[list[int]], size: int) -> list[list[int]]:
        if len(cells) != size or any(len(row) != size for row in cells):
            raise ValueError(f"cells must be a {size}x{size} matrix")

        copied = [[int(value) for value in row] for row in cells]
        if any(value not in (0, 1) for row in copied for value in row):
            raise ValueError("grid cells must be 0 or 1")

        return copied

    def in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid."""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        """Return True if the cell at (row, col) is unoccupied."""
        if not self.in_bounds(row, col):
            raise IndexError(f"cell ({row}, {col}) is out of bounds")
        return self._cells[row][col] == 0

    def can_place(self, block: Block, row: int, col: int) -> bool:
        """Return True if the block can be placed with its origin at (row, col)."""
        for row_offset, col_offset in block.cells:
            target_row = row + row_offset
            target_col = col + col_offset
            if not self.in_bounds(target_row, target_col):
                return False
            if self._cells[target_row][target_col] == 1:
                return False
        return True

    def place(self, block: Block, row: int, col: int) -> None:
        """Fill the block's cells onto the grid. Raises ValueError if placement is illegal."""
        if not self.can_place(block, row, col):
            raise ValueError(f"cannot place block {block.name!r} at ({row}, {col})")

        for row_offset, col_offset in block.cells:
            self._cells[row + row_offset][col + col_offset] = 1

    def clear_full_lines(self) -> int:
        """Clear any fully-filled rows and columns and return the total count cleared."""
        # TODO: optimise by only scanning rows/cols touched by the last placement
        full_rows = {row for row in range(self.size) if all(self._cells[row])}
        full_cols = {
            col for col in range(self.size) if all(self._cells[row][col] for row in range(self.size))
        }

        for row in full_rows:
            for col in range(self.size):
                self._cells[row][col] = 0

        for col in full_cols:
            for row in range(self.size):
                self._cells[row][col] = 0

        return len(full_rows) + len(full_cols)

    ### Utility methods

    def placements(self, block: Block) -> Iterator[tuple[int, int]]:
        """Yield every (row, col) origin where the block legally fits on the current grid."""
        for row in range(self.size):
            for col in range(self.size):
                if self.can_place(block, row, col):
                    yield row, col

    def can_place_anywhere(self, block: Block) -> bool:
        """Return True if the block has at least one legal placement on the current grid."""
        return any(True for _ in self.placements(block))

    def to_matrix(self) -> list[list[int]]:
        """Return a deep copy of the internal cell matrix."""
        return [row.copy() for row in self._cells]

    def clone(self) -> "Grid":
        """Return an independent deep copy, skipping constructor re-validation.

        Hot path for search rollouts — assumes the source grid is already valid,
        so it copies cells directly instead of going through ``_validate_cells``.
        """
        new = Grid.__new__(Grid)
        new.size = self.size
        new._cells = [row.copy() for row in self._cells]
        return new

    def __str__(self) -> str:
        return "\n".join(" ".join(str(value) for value in row) for row in self._cells)
