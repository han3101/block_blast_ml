from __future__ import annotations

from engine.block import Block


class Grid:
    """Fixed-size 0/1 8x8 grid with Block Blast-style placement and line clearing."""

    def __init__(self, size: int = 8, cells: list[list[int]] | None = None) -> None:
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
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            raise IndexError(f"cell ({row}, {col}) is out of bounds")
        return self._cells[row][col] == 0

    def can_place(self, block: Block, row: int, col: int) -> bool:
        for row_offset, col_offset in block.cells:
            target_row = row + row_offset
            target_col = col + col_offset
            if not self.in_bounds(target_row, target_col):
                return False
            if self._cells[target_row][target_col] == 1:
                return False
        return True

    def place(self, block: Block, row: int, col: int) -> int:
        if not self.can_place(block, row, col):
            raise ValueError(f"cannot place block {block.name!r} at ({row}, {col})")

        for row_offset, col_offset in block.cells:
            self._cells[row + row_offset][col + col_offset] = 1

        return self.clear_full_lines()

    def clear_full_lines(self) -> int:
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

    def to_matrix(self) -> list[list[int]]:
        return [row.copy() for row in self._cells]

    def __str__(self) -> str:
        return "\n".join(" ".join(str(value) for value in row) for row in self._cells)
