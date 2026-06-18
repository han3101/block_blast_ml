import pytest

from engine.block import Block, LINE_2_ACROSS, LINE_2_DOWN, LINE_3_DOWN, TWO_BY_TWO
from engine.game import GameState, StepResult
from engine.grid import Grid
from engine.scoring import SimpleScorer


# LINE_2_DOWN (2 cells, vertical) is the smallest real block and fits almost anywhere.
SMALL = LINE_2_DOWN
SMALL_POOL = (LINE_2_DOWN,)

# A block that never fits on an 8×8 grid.
HUGE = Block("huge", [(r, c) for r in range(9) for c in range(9)])


# --- initial state ---

def test_initial_hand_has_three_slots() -> None:
    gs = GameState(seed=0)
    assert len(gs.hand) == 3


def test_initial_hand_not_all_none() -> None:
    gs = GameState(seed=0)
    assert any(b is not None for b in gs.hand)


def test_initial_game_not_over() -> None:
    gs = GameState(seed=0)
    assert not gs.game_over


def test_initial_score_zero() -> None:
    gs = GameState(seed=0)
    assert gs.score == 0


def test_initial_grid_is_empty() -> None:
    gs = GameState(seed=0)
    assert gs.grid.to_matrix() == [[0] * 8 for _ in range(8)]


# --- place() basic ---

def test_place_nulls_slot() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    assert gs.hand[0] is None


def test_place_fills_grid_cells() -> None:
    # LINE_2_DOWN at (3, 4) occupies (3,4) and (4,4)
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 3, 4)
    matrix = gs.grid.to_matrix()
    assert matrix[3][4] == 1
    assert matrix[4][4] == 1


def test_place_returns_step_result() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    result = gs.place(0, 0, 0)
    assert isinstance(result, StepResult)


def test_place_cells_placed_count() -> None:
    gs = GameState(seed=0, pool=(TWO_BY_TWO,))
    result = gs.place(0, 0, 0)
    assert result.cells_placed == 4


def test_place_no_lines_cleared_on_partial_fill() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    result = gs.place(0, 0, 0)
    assert result.lines_cleared == 0


def test_place_score_accumulates() -> None:
    # LINE_2_DOWN has 2 cells; 2 placements = 4 points at cell_points=1
    gs = GameState(seed=0, pool=SMALL_POOL, scorer=SimpleScorer(cell_points=1, line_points=10))
    gs.place(0, 0, 0)   # fills (0,0),(1,0)
    gs.place(1, 2, 0)   # fills (2,0),(3,0) — non-overlapping
    assert gs.score == 4


# --- line clearing ---

def test_place_clears_full_row() -> None:
    # Pre-fill cols 0–5 in row 0, then place LINE_2_ACROSS at (0,6) to complete the row.
    cells = [[0] * 8 for _ in range(8)]
    for c in range(6):
        cells[0][c] = 1
    gs = GameState(seed=0, pool=(LINE_2_ACROSS,))
    gs._grid = Grid(cells=cells)
    result = gs.place(0, 0, 6)
    assert result.lines_cleared == 1
    assert all(gs.grid.to_matrix()[0][c] == 0 for c in range(8))


def test_place_line_clear_scores_correctly() -> None:
    cells = [[0] * 8 for _ in range(8)]
    for c in range(6):
        cells[0][c] = 1
    scorer = SimpleScorer(cell_points=1, line_points=10)
    gs = GameState(seed=0, pool=(LINE_2_ACROSS,), scorer=scorer)
    gs._grid = Grid(cells=cells)
    result = gs.place(0, 0, 6)
    # 2 cells placed + 1 line cleared → 2*1 + 1*10 = 12
    assert result.score == 12


# --- hand refresh ---

def test_all_slots_none_triggers_new_hand() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)   # (0,0),(1,0)
    gs.place(1, 2, 0)   # (2,0),(3,0)
    gs.place(2, 4, 0)   # (4,0),(5,0)
    assert any(b is not None for b in gs.hand)


def test_hand_refreshed_flag_set() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    gs.place(1, 2, 0)
    result = gs.place(2, 4, 0)
    assert result.hand_refreshed is True


def test_hand_refreshed_flag_not_set_mid_hand() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    result = gs.place(0, 0, 0)
    assert result.hand_refreshed is False


# --- legal_actions ---

def test_legal_actions_non_empty_on_fresh_game() -> None:
    gs = GameState(seed=0)
    assert len(gs.legal_actions()) > 0


def test_legal_actions_only_includes_non_none_slots() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    actions = gs.legal_actions()
    assert all(slot != 0 for slot, _, _ in actions)


def test_legal_actions_correct_format() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    for slot, row, col in gs.legal_actions():
        assert 0 <= slot <= 2
        assert 0 <= row < 8
        assert 0 <= col < 8


# --- game over ---

def test_game_over_when_nothing_fits() -> None:
    gs = GameState(seed=0, pool=(HUGE,), mode="random")
    assert gs.game_over


def test_place_raises_when_game_over() -> None:
    gs = GameState(seed=0, pool=(HUGE,), mode="random")
    with pytest.raises(ValueError, match="game is over"):
        gs.place(0, 0, 0)


def test_mid_hand_game_over() -> None:
    # Slots 1 & 2 are replaced with HUGE (9×9, never fits).
    # After placing slot 0, no legal moves remain.
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs._hand[1] = HUGE
    gs._hand[2] = HUGE
    result = gs.place(0, 0, 0)
    assert result.game_over is True


def test_game_over_after_first_real_placement() -> None:
    # Board is nearly full. clear_full_lines() scans ALL rows/cols after
    # each placement, so we must ensure no row or column is ever full —
    # each is given at least one empty cell that survives the placement.
    # (0,0)-(1,0) is the only vertical pair; every other gap is an isolated
    # 1×1 hole, so after slot 0 is placed no LINE_2_DOWN can fit anywhere.
    cells = [[1] * 8 for _ in range(8)]
    cells[0][0] = 0   # slot-0 target (top)
    cells[1][0] = 0   # slot-0 target (bottom)
    cells[0][2] = 0   # row-0 extra → row 0 stays incomplete after place
    cells[1][3] = 0   # row-1 extra → row 1 stays incomplete after place
    cells[5][0] = 0   # col-0 extra → col 0 stays incomplete after place
    cells[2][7] = 0   # isolated gaps keep remaining rows/cols non-full
    cells[3][1] = 0
    cells[4][4] = 0
    cells[6][5] = 0
    cells[7][6] = 0
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs._grid = Grid(cells=cells)
    gs._hand = [LINE_2_DOWN, LINE_2_DOWN, LINE_2_DOWN]
    result = gs.place(0, 0, 0)
    assert result.game_over is True
    assert gs.game_over is True


def test_game_over_after_second_real_placement() -> None:
    # Same no-full-row/col constraint as above, but with two target pairs.
    # After slot 0 fills (0,0)-(1,0), slot 1 (LINE_2_DOWN) can still reach
    # (0,4)-(1,4) → not game over.  After slot 1 fills that pair, only
    # LINE_3_DOWN remains; it needs 3 consecutive vertical cells and none
    # exist, so game over is triggered on the second placement.
    cells = [[1] * 8 for _ in range(8)]
    cells[0][0] = 0   # slot-0 target (top)
    cells[1][0] = 0   # slot-0 target (bottom)
    cells[0][4] = 0   # slot-1 target (top)
    cells[1][4] = 0   # slot-1 target (bottom)
    cells[0][2] = 0   # row-0 extra
    cells[1][3] = 0   # row-1 extra
    cells[5][0] = 0   # col-0 extra
    cells[4][4] = 0   # col-4 extra
    cells[2][7] = 0
    cells[3][1] = 0
    cells[6][5] = 0
    cells[7][6] = 0
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs._grid = Grid(cells=cells)
    gs._hand = [LINE_2_DOWN, LINE_2_DOWN, LINE_3_DOWN]
    result = gs.place(0, 0, 0)
    assert result.game_over is False   # slot 1 still fits at (0,4)-(1,4)
    result = gs.place(1, 0, 4)
    assert result.game_over is True    # LINE_3_DOWN has no 3-cell column run
    assert gs.game_over is True


# --- reset ---

def test_reset_clears_grid() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    gs.reset()
    assert gs.grid.to_matrix() == [[0] * 8 for _ in range(8)]


def test_reset_clears_score() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    gs.reset()
    assert gs.score == 0


def test_reset_deals_new_hand() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.reset()
    assert any(b is not None for b in gs.hand)


def test_reset_clears_game_over() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs._hand = [HUGE, HUGE, HUGE]
    gs.game_over = True
    assert gs.game_over
    gs.reset()
    assert not gs.game_over


# --- snapshot ---

def test_snapshot_has_expected_keys() -> None:
    gs = GameState(seed=0)
    snap = gs.snapshot()
    assert set(snap.keys()) == {"grid", "hand", "score", "game_over"}


def test_snapshot_grid_matches_to_matrix() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 3, 3)
    snap = gs.snapshot()
    assert snap["grid"] == gs.grid.to_matrix()


def test_snapshot_hand_names_are_catalog_names() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    snap = gs.snapshot()
    assert all(name == LINE_2_DOWN.name for name in snap["hand"])


def test_snapshot_empty_slot_is_none() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    snap = gs.snapshot()
    assert snap["hand"][0] is None


def test_snapshot_score() -> None:
    # LINE_2_DOWN has 2 cells; 2 cells × 5 points = 10
    gs = GameState(seed=0, pool=SMALL_POOL, scorer=SimpleScorer(cell_points=5))
    gs.place(0, 0, 0)
    snap = gs.snapshot()
    assert snap["score"] == 10


# --- error handling ---

def test_place_invalid_slot_raises() -> None:
    gs = GameState(seed=0)
    with pytest.raises(ValueError, match="slot must be"):
        gs.place(3, 0, 0)


def test_place_empty_slot_raises() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    with pytest.raises(ValueError, match="already empty"):
        gs.place(0, 2, 0)


def test_place_out_of_bounds_raises() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    with pytest.raises(ValueError):
        gs.place(0, 10, 10)


def test_place_occupied_cell_raises() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)   # fills (0,0),(1,0)
    gs.place(1, 0, 2)   # fills (0,2),(1,2)
    with pytest.raises(ValueError):
        gs.place(2, 0, 0)   # (0,0) already occupied


# --- custom scorer injection ---

def test_custom_scorer_injected() -> None:
    scorer = SimpleScorer(cell_points=99)
    gs = GameState(seed=0, pool=SMALL_POOL, scorer=scorer)
    gs.place(0, 0, 0)
    # LINE_2_DOWN has 2 cells → 2 × 99 = 198
    assert gs.score == 198


# --- last_result ---

def test_last_result_none_before_first_move() -> None:
    gs = GameState(seed=0)
    assert gs.last_result is None


def test_last_result_matches_return_value() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    result = gs.place(0, 0, 0)
    assert gs.last_result is result


def test_last_result_cleared_on_reset() -> None:
    gs = GameState(seed=0, pool=SMALL_POOL)
    gs.place(0, 0, 0)
    gs.reset()
    assert gs.last_result is None


# --- determinism ---

def test_seeded_game_is_deterministic() -> None:
    gs1 = GameState(seed=42)
    gs2 = GameState(seed=42)
    assert gs1.hand == gs2.hand
