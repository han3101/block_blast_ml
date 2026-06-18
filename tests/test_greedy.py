from __future__ import annotations

import random
import statistics

import pytest

from engine.game import GameState
from rl.agents.greedy import choose_action, _count_holes
from rl.encoding import decode_action, action_mask


# --- choose_action returns a legal action ---

def test_choose_action_returns_legal_action() -> None:
    state = GameState(seed=0)
    action = choose_action(state)
    slot, row, col = decode_action(action)
    assert (slot, row, col) in state.legal_actions()


def test_choose_action_encoded_index_in_mask() -> None:
    state = GameState(seed=0)
    action = choose_action(state)
    mask = action_mask(state)
    assert mask[action]


def test_choose_action_raises_when_no_legal_actions() -> None:
    state = GameState(seed=0)
    state.game_over = True
    state._hand = [None, None, None]
    with pytest.raises(ValueError, match="no legal actions"):
        choose_action(state)


# --- _count_holes ---

def test_count_holes_empty_grid() -> None:
    matrix = [[0] * 8 for _ in range(8)]
    assert _count_holes(matrix) == 0


def test_count_holes_full_grid() -> None:
    matrix = [[1] * 8 for _ in range(8)]
    assert _count_holes(matrix) == 0


def test_count_holes_one_hole() -> None:
    # filled cell at row 0 col 0, empty below → 1 hole
    matrix = [[0] * 8 for _ in range(8)]
    matrix[0][0] = 1
    assert _count_holes(matrix) == 7  # rows 1-7 in col 0 are holes


def test_count_holes_no_overhang() -> None:
    # empty cell above filled cell is NOT a hole
    matrix = [[0] * 8 for _ in range(8)]
    matrix[7][0] = 1  # filled at bottom, nothing above
    assert _count_holes(matrix) == 0


# --- greedy beats random on mean score ---

def _play_random(seed: int) -> int:
    rng = random.Random(seed + 1000)
    state = GameState(seed=seed)
    while not state.game_over:
        legal = state.legal_actions()
        slot, row, col = rng.choice(legal)
        state.place(slot, row, col)
    return state.score


def _play_greedy(seed: int) -> int:
    state = GameState(seed=seed)
    while not state.game_over:
        action = choose_action(state)
        slot, row, col = decode_action(action)
        state.place(slot, row, col)
    return state.score


def test_greedy_beats_random_mean_score() -> None:
    """Greedy mean score must exceed random-legal mean over 30 seeds."""
    seeds = list(range(30))
    greedy_scores = [_play_greedy(s) for s in seeds]
    random_scores = [_play_random(s) for s in seeds]
    assert statistics.mean(greedy_scores) > statistics.mean(random_scores), (
        f"greedy mean {statistics.mean(greedy_scores):.1f} did not beat "
        f"random mean {statistics.mean(random_scores):.1f}"
    )


# --- full greedy rollout runs without error ---

def test_greedy_rollout_completes() -> None:
    state = GameState(seed=42)
    steps = 0
    while not state.game_over:
        action = choose_action(state)
        slot, row, col = decode_action(action)
        state.place(slot, row, col)
        steps += 1
        assert steps < 10_000, "rollout unexpectedly long"
