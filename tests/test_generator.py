import pytest

from engine.block import ALL_BLOCKS, Block, LINE_5_ACROSS, LINE_5_DOWN
from engine.generator import HAND_SIZE, HandGenerator
from engine.grid import Grid


SINGLE = Block("single", [(0, 0)])
TINY_POOL = (SINGLE,)

HUGE_BLOCK = Block("huge", [(r, c) for r in range(8) for c in range(8)])


def full_grid() -> Grid:
    return Grid(cells=[[1] * 8 for _ in range(8)])


def _all_fit(hand: list[Block], grid: Grid) -> bool:
    return all(grid.can_place_anywhere(b) for b in hand)


# --- random ---

def test_random_always_returns_hand_size() -> None:
    gen = HandGenerator(seed=0, mode="random")
    hand = gen.deal(Grid())
    assert len(hand) == HAND_SIZE


def test_random_is_seeded_deterministically() -> None:
    g1 = HandGenerator(seed=42, mode="random")
    g2 = HandGenerator(seed=42, mode="random")
    assert g1.deal(Grid()) == g2.deal(Grid())


def test_random_different_seeds_differ() -> None:
    g1 = HandGenerator(seed=1, mode="random")
    g2 = HandGenerator(seed=2, mode="random")
    assert g1.deal(Grid()) != g2.deal(Grid())


def test_random_can_deal_unplaceable_hand() -> None:
    gen = HandGenerator(seed=0, pool=(HUGE_BLOCK,), mode="random")
    grid = Grid(cells=[[1] * 8 for _ in range(8)])
    hand = gen.deal(grid)
    assert all(b is HUGE_BLOCK for b in hand)


# --- at_least_one ---

def test_at_least_one_returns_hand_size() -> None:
    gen = HandGenerator(seed=0, mode="at_least_one")
    assert len(gen.deal(Grid())) == HAND_SIZE


def test_at_least_one_has_placeable_piece_on_empty_grid() -> None:
    gen = HandGenerator(seed=0, mode="at_least_one")
    hand = gen.deal(Grid())
    assert any(Grid().can_place_anywhere(b) for b in hand)


def test_at_least_one_returns_none_when_nothing_fits() -> None:
    gen = HandGenerator(seed=0, pool=(HUGE_BLOCK,), mode="at_least_one")
    assert gen.deal(full_grid()) is None


def test_at_least_one_is_seeded_deterministically() -> None:
    g1 = HandGenerator(seed=7, mode="at_least_one")
    g2 = HandGenerator(seed=7, mode="at_least_one")
    assert g1.deal(Grid()) == g2.deal(Grid())


def test_at_least_one_draws_from_pool_only() -> None:
    gen = HandGenerator(seed=0, pool=TINY_POOL, mode="at_least_one")
    hand = gen.deal(Grid())
    assert all(b is SINGLE for b in hand)


# --- solvable ---

def test_solvable_returns_hand_size_on_empty_grid() -> None:
    gen = HandGenerator(seed=0, mode="solvable")
    hand = gen.deal(Grid())
    assert hand is not None
    assert len(hand) == HAND_SIZE


def test_solvable_returns_none_when_full_solvability_impossible() -> None:
    gen = HandGenerator(seed=0, pool=(HUGE_BLOCK,), mode="solvable")
    assert gen.deal(full_grid()) is None


def test_solvable_all_pieces_fit_in_sequence() -> None:
    gen = HandGenerator(seed=0, mode="solvable")
    grid = Grid()
    hand = gen.deal(grid)
    for block in hand:
        assert grid.can_place_anywhere(block)
        row, col = next(grid.placements(block))
        grid.place(block, row, col)


def test_solvable_is_seeded_deterministically() -> None:
    g1 = HandGenerator(seed=99, mode="solvable")
    g2 = HandGenerator(seed=99, mode="solvable")
    assert g1.deal(Grid()) == g2.deal(Grid())


# --- general ---

def test_unknown_mode_raises() -> None:
    gen = HandGenerator(seed=0, mode="random")  # type: ignore[arg-type]
    gen._mode = "bad"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown mode"):
        gen.deal(Grid())


def test_sequential_deals_advance_rng() -> None:
    gen = HandGenerator(seed=0, mode="random")
    h1 = gen.deal(Grid())
    h2 = gen.deal(Grid())
    assert h1 != h2
