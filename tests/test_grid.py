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

    grid.place(LEFT_T, 2, 3)

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

    grid.place(SINGLE, 4, 7)
    cleared = grid.clear_full_lines()

    assert cleared == 1
    assert grid.to_matrix()[4] == [0 for _ in range(8)]


def test_full_column_clears_after_place() -> None:
    cells = [[0 for _ in range(8)] for _ in range(8)]
    for row in range(7):
        cells[row][2] = 1
    grid = Grid(cells=cells)

    grid.place(SINGLE, 7, 2)
    cleared = grid.clear_full_lines()

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

    grid.place(SINGLE, 3, 5)
    cleared = grid.clear_full_lines()

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


def test_invalid_cells_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="8x8"):
        Grid(cells=[[0] * 8 for _ in range(7)])


def test_invalid_cells_wrong_column_count() -> None:
    cells = [[0] * 8 for _ in range(8)]
    cells[3] = [0] * 7
    with pytest.raises(ValueError, match="8x8"):
        Grid(cells=cells)


def test_grid_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Grid(size=0)
    with pytest.raises(ValueError, match="positive"):
        Grid(size=-1)


def test_custom_size_grid_initializes_correctly() -> None:
    grid = Grid(size=4)
    assert grid.to_matrix() == [[0] * 4 for _ in range(4)]


def test_in_bounds_corners_and_out_of_range() -> None:
    grid = Grid()
    assert grid.in_bounds(0, 0)
    assert grid.in_bounds(7, 7)
    assert not grid.in_bounds(-1, 0)
    assert not grid.in_bounds(0, -1)
    assert not grid.in_bounds(8, 0)
    assert not grid.in_bounds(0, 8)


def test_is_empty_reflects_cell_state() -> None:
    grid = Grid()
    grid.place(SINGLE, 3, 3)
    assert not grid.is_empty(3, 3)
    assert grid.is_empty(3, 4)


def test_is_empty_raises_for_out_of_bounds() -> None:
    grid = Grid()
    with pytest.raises(IndexError, match="out of bounds"):
        grid.is_empty(8, 0)
    with pytest.raises(IndexError, match="out of bounds"):
        grid.is_empty(-1, 0)


def test_can_place_with_negative_origin_returns_false() -> None:
    grid = Grid()
    assert not grid.can_place(SINGLE, -1, 0)
    assert not grid.can_place(SINGLE, 0, -1)


def test_can_place_single_cell_at_bottom_right_corner() -> None:
    grid = Grid()
    assert grid.can_place(SINGLE, 7, 7)
    assert not grid.can_place(LINE_2_H, 7, 7)
    assert not grid.can_place(LINE_2_V, 7, 7)


def test_multiple_full_rows_clear_simultaneously() -> None:
    cells = [[0] * 8 for _ in range(8)]
    cells[2] = [1, 1, 1, 1, 1, 1, 1, 0]
    cells[3] = [1, 1, 1, 1, 1, 1, 1, 0]
    grid = Grid(cells=cells)

    grid.place(LINE_2_V, 2, 7)
    cleared = grid.clear_full_lines()

    assert cleared == 2
    matrix = grid.to_matrix()
    assert matrix[2] == [0] * 8
    assert matrix[3] == [0] * 8


def test_multiple_full_columns_clear_simultaneously() -> None:
    cells = [[0] * 8 for _ in range(8)]
    for row in range(7):
        cells[row][1] = 1
        cells[row][2] = 1
    grid = Grid(cells=cells)

    grid.place(LINE_2_H, 7, 1)
    cleared = grid.clear_full_lines()

    assert cleared == 2
    matrix = grid.to_matrix()
    assert [row[1] for row in matrix] == [0] * 8
    assert [row[2] for row in matrix] == [0] * 8


def test_clear_full_lines_on_empty_grid_returns_zero() -> None:
    assert Grid().clear_full_lines() == 0


def test_clear_full_lines_called_directly() -> None:
    cells = [[0] * 8 for _ in range(8)]
    cells[0] = [1] * 8
    grid = Grid(cells=cells)

    assert grid.clear_full_lines() == 1
    assert grid.to_matrix()[0] == [0] * 8
    assert grid.clear_full_lines() == 0


def test_str_represents_grid_as_space_separated_rows() -> None:
    grid = Grid()
    grid.place(SINGLE, 0, 0)
    lines = str(grid).splitlines()
    assert lines[0] == "1 0 0 0 0 0 0 0"
    assert lines[1] == "0 0 0 0 0 0 0 0"
    assert len(lines) == 8
