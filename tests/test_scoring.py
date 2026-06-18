import pytest

from engine.scoring import BASE_LINE_SCORES, ComboScorer, SimpleScorer


# --- BASE_LINE_SCORES table ---

def test_base_line_scores_zero() -> None:
    assert BASE_LINE_SCORES[0] == 0

def test_base_line_scores_one() -> None:
    assert BASE_LINE_SCORES[1] == 10

def test_base_line_scores_two() -> None:
    assert BASE_LINE_SCORES[2] == 20

def test_base_line_scores_three() -> None:
    assert BASE_LINE_SCORES[3] == 60

def test_base_line_scores_four() -> None:
    assert BASE_LINE_SCORES[4] == 120

def test_base_line_scores_five() -> None:
    assert BASE_LINE_SCORES[5] == 200

def test_base_line_scores_eight() -> None:
    assert BASE_LINE_SCORES[8] == 560


# --- ComboScorer: single placement ---

def test_combo_no_clear_placement_only() -> None:
    cs = ComboScorer()
    score = cs.score_placement(block_placed=4, lines_cleared=0)
    assert score == 4

def test_combo_single_clear_no_prior_streak() -> None:
    # N=0 at start; multiplier = 1; BASE[1]=10; cells=3
    cs = ComboScorer()
    score = cs.score_placement(block_placed=3, lines_cleared=1)
    assert score == 3 + 10 * 1  # 13

def test_combo_does_not_change_mid_round() -> None:
    # N only changes at end_round(); mid-round combo stays at 0
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    assert cs.combo == 0

def test_combo_multi_line_simultaneous() -> None:
    # 4 lines at once, no prior streak
    cs = ComboScorer()
    score = cs.score_placement(block_placed=4, lines_cleared=4)
    assert score == 4 + 120 * 1  # 124


# --- ComboScorer: end_round() streak updates ---

def test_end_round_clearing_round_increments_N() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)  # cleared
    cs.end_round()
    assert cs.combo == 1

def test_end_round_clearless_round_resets_N() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=0)
    cs.end_round()
    assert cs.combo == 0

def test_end_round_clears_the_flag() -> None:
    # After end_round, a new non-clearing round should reset N to 0
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 1
    cs.score_placement(block_placed=2, lines_cleared=0)
    cs.end_round()  # no clear this round → N → 0
    assert cs.combo == 0


# --- ComboScorer: multi-round streak ---

def test_streak_grows_across_clearing_rounds() -> None:
    cs = ComboScorer()
    # Round 1: N=0, multiplier=1
    cs.score_placement(block_placed=2, lines_cleared=1)  # +2 + 10*1 = 12
    cs.end_round()  # N → 1
    score_after_r1 = cs.score
    assert score_after_r1 == 12

    # Round 2: N=1, multiplier=2
    cs.score_placement(block_placed=2, lines_cleared=1)  # +2 + 10*2 = 22
    cs.end_round()  # N → 2
    assert cs.score == score_after_r1 + 22

def test_streak_uncapped_past_reference_limit() -> None:
    # Reference capped at 8; ours should go well beyond
    cs = ComboScorer()
    for _ in range(12):
        cs.score_placement(block_placed=1, lines_cleared=1)
        cs.end_round()
    assert cs.combo == 12

def test_streak_resets_after_clearless_round() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 1
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 2
    # Clearless round
    cs.score_placement(block_placed=3, lines_cleared=0)
    cs.end_round()  # N → 0
    assert cs.combo == 0

def test_streak_restarts_after_reset_round() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 1
    cs.score_placement(block_placed=2, lines_cleared=0)
    cs.end_round()  # N → 0
    # New clearing round: back to multiplier 1
    cs.score_placement(block_placed=2, lines_cleared=1)
    assert cs.combo == 0   # still 0 until end_round
    cs.end_round()
    assert cs.combo == 1


# --- ComboScorer: set-level reset (setup-then-clear round stays alive) ---

def test_setup_then_clear_round_preserves_streak() -> None:
    # A round where placements 1–2 clear nothing but placement 3 clears — N should increment.
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 1

    # Round 2: first two placements are structural (no clear)
    cs.score_placement(block_placed=3, lines_cleared=0)
    cs.score_placement(block_placed=4, lines_cleared=0)
    # Third placement clears
    cs.score_placement(block_placed=2, lines_cleared=1)  # multiplier = N+1 = 2
    cs.end_round()  # cleared_this_round=True → N → 2
    assert cs.combo == 2

def test_all_clearless_round_breaks_streak() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()  # N → 1
    # Entire next round: no clears
    cs.score_placement(block_placed=3, lines_cleared=0)
    cs.score_placement(block_placed=2, lines_cleared=0)
    cs.score_placement(block_placed=4, lines_cleared=0)
    cs.end_round()  # N → 0
    assert cs.combo == 0


# --- ComboScorer: all clears in a round share the same multiplier ---

def test_multiple_clears_in_round_use_same_N() -> None:
    # N=1 entering round; each clearing placement uses multiplier 2
    cs = ComboScorer()
    cs.score_placement(block_placed=1, lines_cleared=1)
    cs.end_round()  # N → 1

    cs.score_placement(block_placed=1, lines_cleared=1)   # 1 + 10*2 = 21
    cs.score_placement(block_placed=1, lines_cleared=2)   # 1 + 20*2 = 41
    cs.end_round()  # N → 2
    # Round 1 total: 1+10 = 11; Round 2: 21 + 41 = 62 → total 11+62 = 73
    assert cs.score == 73


# --- ComboScorer: reset() ---

def test_combo_scorer_reset_clears_score() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()
    cs.reset()
    assert cs.score == 0

def test_combo_scorer_reset_clears_streak() -> None:
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)
    cs.end_round()
    cs.reset()
    assert cs.combo == 0

def test_combo_scorer_reset_clears_in_round_flag() -> None:
    # After reset, a clearless end_round should not increment N
    cs = ComboScorer()
    cs.score_placement(block_placed=2, lines_cleared=1)  # would set cleared_this_round
    cs.reset()
    cs.end_round()  # cleared_this_round was reset → N stays 0
    assert cs.combo == 0


# --- SimpleScorer still works and has end_round() no-op ---

def test_simple_scorer_end_round_noop() -> None:
    ss = SimpleScorer(cell_points=1, line_points=10)
    ss.score_placement(block_placed=2, lines_cleared=1)
    ss.end_round()
    assert ss.score == 12

def test_simple_scorer_unaffected() -> None:
    ss = SimpleScorer(cell_points=1, line_points=10)
    ss.score_placement(block_placed=2, lines_cleared=1)
    assert ss.score == 12
