import pytest

from engine.block import Block, LEFT_T
from engine.grid import Grid


SINGLE = Block("single", [(0, 0)])
LINE_2_H = Block("line_2_h", [(0, 0), (0, 1)])
LINE_2_V = Block("line_2_v", [(0, 0), (1, 0)])


def test_empty_grid_initializes_to_zeroes() -> None:
    grid = Grid()

    assert grid.to_matrix() == [[0 for _ in range(8)] for _ in range(8)]


def test_place_updates_expected_cells() -> None:
    grid = Grid()

    cleared = grid.place(LEFT_T, 2, 3)

    assert cleared == 0
    assert grid.to_matrix()[2][3] == 1
    assert grid.to_matrix()[3][3] == 1
    assert grid.to_matrix()[3][4] == 1
    assert grid.to_matrix()[4][3] == 1


def test_collision_blocks_placement() -> None:
    grid = Grid()
    grid.place(SINGLE, 0, 1)

    assert not grid.can_place(LINE_2_H, 0, 0)
    with pytest.raises(ValueError, match="cannot place"):
        grid.place(LINE_2_H, 0, 0)


def test_out_of_bounds_blocks_placement() -> None:
    grid = Grid()

    assert not grid.can_place(LINE_2_H, 0, 7)
    assert not grid.can_place(LINE_2_V, 7, 0)


def test_full_row_clears_after_place() -> None:
    cells = [[0 for _ in range(8)] for _ in range(8)]
    cells[4] = [1, 1, 1, 1, 1, 1, 1, 0]
    grid = Grid(cells=cells)

    cleared = grid.place(SINGLE, 4, 7)

    assert cleared == 1
    assert grid.to_matrix()[4] == [0 for _ in range(8)]


def test_full_column_clears_after_place() -> None:
    cells = [[0 for _ in range(8)] for _ in range(8)]
    for row in range(7):
        cells[row][2] = 1
    grid = Grid(cells=cells)

    cleared = grid.place(SINGLE, 7, 2)

    assert cleared == 1
    assert [row[2] for row in grid.to_matrix()] == [0 for _ in range(8)]


def test_full_row_and_column_clear_together() -> None:
    cells = [[0 for _ in range(8)] for _ in range(8)]
    cells[3] = [1 for _ in range(8)]
    cells[3][5] = 0
    for row in range(8):
        cells[row][5] = 1
    cells[3][5] = 0
    grid = Grid(cells=cells)

    cleared = grid.place(SINGLE, 3, 5)

    assert cleared == 2
    matrix = grid.to_matrix()
    assert matrix[3] == [0 for _ in range(8)]
    assert [row[5] for row in matrix] == [0 for _ in range(8)]


def test_to_matrix_returns_a_copy() -> None:
    grid = Grid()
    matrix = grid.to_matrix()

    matrix[0][0] = 1

    assert grid.to_matrix()[0][0] == 0


def test_invalid_initial_cells_are_rejected() -> None:
    with pytest.raises(ValueError, match="8x8"):
        Grid(cells=[[0]])

    cells = [[0 for _ in range(8)] for _ in range(8)]
    cells[0][0] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        Grid(cells=cells)
