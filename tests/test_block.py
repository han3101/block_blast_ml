import pytest

from engine import block as blocks
from engine.block import Block


def cells_from_diagram(diagram: str) -> tuple[tuple[int, int], ...]:
    rows = tuple(line.strip() for line in diagram.strip().splitlines())
    return tuple(
        (row_index, col_index)
        for row_index, row in enumerate(rows)
        for col_index, cell in enumerate(row)
        if cell == "#"
    )


EXPECTED_BLOCK_DIAGRAMS = (
    (blocks.LEFT_T, """
        #.
        ##
        #.
        """),
    (blocks.RIGHT_T, """
        .#
        ##
        .#
        """),
    (blocks.UP_T, """
        .#.
        ###
        """),
    (blocks.DOWN_T, """
        ###
        .#.
        """),
    (blocks.H_LEFT_L_T, """
        ###
        #..
        """),
    (blocks.H_RIGHT_L_T, """
        ###
        ..#
        """),
    (blocks.H_LEFT_L_B, """
        #..
        ###
        """),
    (blocks.H_RIGHT_L_B, """
        ..#
        ###
        """),
    (blocks.V_LEFT_L_B, """
        #.
        #.
        ##
        """),
    (blocks.V_RIGHT_L_B, """
        .#
        .#
        ##
        """),
    (blocks.V_RIGHT_L_T, """
        ##
        .#
        .#
        """),
    (blocks.V_LEFT_L_T, """
        ##
        #.
        #.
        """),
    (blocks.Z_VERT, """
        .#
        ##
        #.
        """),
    (blocks.Z_HORZ_R, """
        ##.
        .##
        """),
    (blocks.Z_HORZ_L, """
        .##
        ##.
        """),
    (blocks.S_VERT, """
        #.
        ##
        .#
        """),
    (blocks.CORNER_RIGHT_TOP_2, """
        ##
        .#
        """),
    (blocks.CORNER_LEFT_TOP_2, """
        ##
        #.
        """),
    (blocks.CORNER_LEFT_BOTTOM_2, """
        #.
        ##
        """),
    (blocks.CORNER_RIGHT_BOTTOM_2, """
        .#
        ##
        """),
    (blocks.CORNER_LEFT_TOP_3, """
        ###
        #..
        #..
        """),
    (blocks.CORNER_RIGHT_TOP_3, """
        ###
        ..#
        ..#
        """),
    (blocks.CORNER_RIGHT_BOTTOM_3, """
        ..#
        ..#
        ###
        """),
    (blocks.CORNER_LEFT_BOTTOM_3, """
        #..
        #..
        ###
        """),
    (blocks.TWO_BY_THREE, """
        ###
        ###
        """),
    (blocks.THREE_BY_TWO, """
        ##
        ##
        ##
        """),
    (blocks.TWO_BY_TWO, """
        ##
        ##
        """),
    (blocks.THREE_BY_THREE, """
        ###
        ###
        ###
        """),
    (blocks.LINE_2_DOWN, """
        #
        #
        """),
    (blocks.LINE_3_DOWN, """
        #
        #
        #
        """),
    (blocks.LINE_4_DOWN, """
        #
        #
        #
        #
        """),
    (blocks.LINE_5_DOWN, """
        #
        #
        #
        #
        #
        """),
    (blocks.LINE_2_ACROSS, "##"),
    (blocks.LINE_3_ACROSS, "###"),
    (blocks.LINE_4_ACROSS, "####"),
    (blocks.LINE_5_ACROSS, "#####"),
)


def diagram_size(diagram: str) -> tuple[int, int]:
    rows = tuple(line.strip() for line in diagram.strip().splitlines())
    return len(rows), len(rows[0])


def test_block_normalizes_cells() -> None:
    block = Block("offset_left_t", [(3, 2), (4, 2), (4, 3), (5, 2)])

    assert block.cells == blocks.LEFT_T.cells
    assert block.height == 3
    assert block.width == 2


def test_starter_blocks_include_left_and_right_t() -> None:
    assert blocks.STARTER_BLOCKS == (blocks.LEFT_T, blocks.RIGHT_T)
    assert blocks.STARTER_BLOCKS == blocks.T_BLOCKS[:2]
    assert blocks.RIGHT_T.cells == ((0, 1), (1, 0), (1, 1), (2, 1))


@pytest.mark.parametrize(("block", "diagram"), EXPECTED_BLOCK_DIAGRAMS)
def test_named_blocks_match_diagrams(block: Block, diagram: str) -> None:
    expected_height, expected_width = diagram_size(diagram)

    assert block.cells == cells_from_diagram(diagram)
    assert block.height == expected_height
    assert block.width == expected_width


def test_blocks_are_grouped_by_shape_category() -> None:
    assert blocks.T_BLOCKS == (blocks.LEFT_T, blocks.RIGHT_T, blocks.UP_T, blocks.DOWN_T)
    assert blocks.L_BLOCKS == (
        blocks.H_LEFT_L_T,
        blocks.H_RIGHT_L_T,
        blocks.H_LEFT_L_B,
        blocks.H_RIGHT_L_B,
        blocks.V_LEFT_L_B,
        blocks.V_RIGHT_L_B,
        blocks.V_RIGHT_L_T,
        blocks.V_LEFT_L_T,
    )
    assert blocks.Z_BLOCKS == (blocks.Z_VERT, blocks.Z_HORZ_R, blocks.Z_HORZ_L, blocks.S_VERT)
    assert blocks.CORNER_BLOCKS == (
        blocks.CORNER_RIGHT_TOP_2,
        blocks.CORNER_LEFT_TOP_2,
        blocks.CORNER_LEFT_BOTTOM_2,
        blocks.CORNER_RIGHT_BOTTOM_2,
        blocks.CORNER_LEFT_TOP_3,
        blocks.CORNER_RIGHT_TOP_3,
        blocks.CORNER_RIGHT_BOTTOM_3,
        blocks.CORNER_LEFT_BOTTOM_3,
    )
    assert blocks.RECTANGLE_BLOCKS == (
        blocks.TWO_BY_THREE,
        blocks.THREE_BY_TWO,
        blocks.TWO_BY_TWO,
        blocks.THREE_BY_THREE,
    )
    assert blocks.LINE_BLOCKS == (
        blocks.LINE_2_DOWN,
        blocks.LINE_3_DOWN,
        blocks.LINE_4_DOWN,
        blocks.LINE_5_DOWN,
        blocks.LINE_2_ACROSS,
        blocks.LINE_3_ACROSS,
        blocks.LINE_4_ACROSS,
        blocks.LINE_5_ACROSS,
    )


def test_all_blocks_collects_each_category_once() -> None:
    categorized_blocks = (
        *blocks.T_BLOCKS,
        *blocks.L_BLOCKS,
        *blocks.Z_BLOCKS,
        *blocks.CORNER_BLOCKS,
        *blocks.RECTANGLE_BLOCKS,
        *blocks.LINE_BLOCKS,
    )
    diagram_blocks = tuple(block for block, _ in EXPECTED_BLOCK_DIAGRAMS)
    defined_blocks = tuple(value for value in vars(blocks).values() if isinstance(value, Block))

    assert blocks.ALL_BLOCKS == categorized_blocks
    assert blocks.ALL_BLOCKS == diagram_blocks
    assert blocks.ALL_BLOCKS == defined_blocks
    assert len({block.name for block in blocks.ALL_BLOCKS}) == len(blocks.ALL_BLOCKS)


def test_block_rejects_empty_cells() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        Block("empty", [])
