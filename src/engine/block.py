from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Block:
    """A block shape represented by row/column offsets from an origin cell."""

    name: str
    cells: tuple[Cell, ...]

    def __init__(self, name: str, cells: Iterable[Cell]) -> None:
        normalized = self._normalize(tuple(cells))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "cells", normalized)

    @staticmethod
    def _normalize(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
        if not cells:
            raise ValueError("a block must contain at least one cell")

        min_row = min(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        normalized = tuple(sorted({(row - min_row, col - min_col) for row, col in cells}))
        return normalized

    @property
    def height(self) -> int:
        return max(row for row, _ in self.cells) + 1

    @property
    def width(self) -> int:
        return max(col for _, col in self.cells) + 1


# T shapes

# #.
# ##
# #.
LEFT_T = Block("left_t", [(0, 0), (1, 0), (1, 1), (2, 0)])

# .#
# ##
# .#
RIGHT_T = Block("right_t", [(0, 1), (1, 0), (1, 1), (2, 1)])

# .#.
# ###
UP_T = Block("up_t", [(0, 1), (1, 0), (1, 1), (1, 2)])

# ###
# .#.
DOWN_T = Block("down_t", [(0, 0), (0, 1), (0, 2), (1, 1)])


# L shapes

# ###
# #..
H_LEFT_L_T = Block("h_left_l_t", [(0, 0), (0, 1), (0, 2), (1, 0)])

# ###
# ..#
H_RIGHT_L_T = Block("h_right_l_t", [(0, 0), (0, 1), (0, 2), (1, 2)])

# #..
# ###
H_LEFT_L_B = Block("h_left_l_b", [(0, 0), (1, 0), (1, 1), (1, 2)])

# ..#
# ###
H_RIGHT_L_B = Block("h_right_l_b", [(0, 2), (1, 0), (1, 1), (1, 2)])

# #.
# #.
# ##
V_LEFT_L_B = Block("v_left_l_b", [(0, 0), (1, 0), (2, 0), (2, 1)])

# .#
# .#
# ##
V_RIGHT_L_B = Block("v_right_l_b", [(0, 1), (1, 1), (2, 0), (2, 1)])

# ##
# .#
# .#
V_RIGHT_L_T = Block("v_right_l_t", [(0, 0), (0, 1), (1, 1), (2, 1)])

# ##
# #.
# #.
V_LEFT_L_T = Block("v_left_l_t", [(0, 0), (0, 1), (1, 0), (2, 0)])


# Z shapes

# .#
# ##
# #.
Z_VERT = Block("z_vert", [(0, 1), (1, 0), (1, 1), (2, 0)])

# ##.
# .##
Z_HORZ_R = Block("z_horz_r", [(0, 0), (0, 1), (1, 1), (1, 2)])

# .##
# ##.
Z_HORZ_L = Block("z_horz_l", [(0, 1), (0, 2), (1, 0), (1, 1)])

# #.
# ##
# .#
S_VERT = Block("s_vert", [(0, 0), (1, 0), (1, 1), (2, 1)])


# Corner shapes

# ##
# .#
CORNER_RIGHT_TOP_2 = Block("corner_right_top_2", [(0, 0), (0, 1), (1, 1)])

# ##
# #.
CORNER_LEFT_TOP_2 = Block("corner_left_top_2", [(0, 0), (0, 1), (1, 0)])

# #.
# ##
CORNER_LEFT_BOTTOM_2 = Block("corner_left_bottom_2", [(0, 0), (1, 0), (1, 1)])

# .#
# ##
CORNER_RIGHT_BOTTOM_2 = Block("corner_right_bottom_2", [(0, 1), (1, 0), (1, 1)])

# ###
# #..
# #..
CORNER_LEFT_TOP_3 = Block("corner_left_top_3", [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)])

# ###
# ..#
# ..#
CORNER_RIGHT_TOP_3 = Block("corner_right_top_3", [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])

# ..#
# ..#
# ###
CORNER_RIGHT_BOTTOM_3 = Block("corner_right_bottom_3", [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])

# #..
# #..
# ###
CORNER_LEFT_BOTTOM_3 = Block("corner_left_bottom_3", [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])


# Rectangle shapes

# ###
# ###
TWO_BY_THREE = Block("two_by_three", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])

# ##
# ##
# ##
THREE_BY_TWO = Block("three_by_two", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

# ##
# ##
TWO_BY_TWO = Block("two_by_two", [(0, 0), (0, 1), (1, 0), (1, 1)])

# ###
# ###
# ###
THREE_BY_THREE = Block(
    "three_by_three",
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
)


# Line shapes

# #
# #
LINE_2_DOWN = Block("line_2_down", [(0, 0), (1, 0)])

# #
# #
# #
LINE_3_DOWN = Block("line_3_down", [(0, 0), (1, 0), (2, 0)])

# #
# #
# #
# #
LINE_4_DOWN = Block("line_4_down", [(0, 0), (1, 0), (2, 0), (3, 0)])

# #
# #
# #
# #
# #
LINE_5_DOWN = Block("line_5_down", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])

# ##
LINE_2_ACROSS = Block("line_2_across", [(0, 0), (0, 1)])

# ###
LINE_3_ACROSS = Block("line_3_across", [(0, 0), (0, 1), (0, 2)])

# ####
LINE_4_ACROSS = Block("line_4_across", [(0, 0), (0, 1), (0, 2), (0, 3)])

# #####
LINE_5_ACROSS = Block("line_5_across", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])


T_BLOCKS: tuple[Block, ...] = (LEFT_T, RIGHT_T, UP_T, DOWN_T)
L_BLOCKS: tuple[Block, ...] = (
    H_LEFT_L_T,
    H_RIGHT_L_T,
    H_LEFT_L_B,
    H_RIGHT_L_B,
    V_LEFT_L_B,
    V_RIGHT_L_B,
    V_RIGHT_L_T,
    V_LEFT_L_T,
)
Z_BLOCKS: tuple[Block, ...] = (Z_VERT, Z_HORZ_R, Z_HORZ_L, S_VERT)
CORNER_BLOCKS: tuple[Block, ...] = (
    CORNER_RIGHT_TOP_2,
    CORNER_LEFT_TOP_2,
    CORNER_LEFT_BOTTOM_2,
    CORNER_RIGHT_BOTTOM_2,
    CORNER_LEFT_TOP_3,
    CORNER_RIGHT_TOP_3,
    CORNER_RIGHT_BOTTOM_3,
    CORNER_LEFT_BOTTOM_3,
)
RECTANGLE_BLOCKS: tuple[Block, ...] = (TWO_BY_THREE, THREE_BY_TWO, TWO_BY_TWO, THREE_BY_THREE)
LINE_BLOCKS: tuple[Block, ...] = (
    LINE_2_DOWN,
    LINE_3_DOWN,
    LINE_4_DOWN,
    LINE_5_DOWN,
    LINE_2_ACROSS,
    LINE_3_ACROSS,
    LINE_4_ACROSS,
    LINE_5_ACROSS,
)

ALL_BLOCKS: tuple[Block, ...] = (
    *T_BLOCKS,
    *L_BLOCKS,
    *Z_BLOCKS,
    *CORNER_BLOCKS,
    *RECTANGLE_BLOCKS,
    *LINE_BLOCKS,
)

STARTER_BLOCKS: tuple[Block, ...] = T_BLOCKS[:2]
