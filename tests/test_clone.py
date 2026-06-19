"""Tests for the search clone()/restore() primitive (Phase 3a prerequisite).

These diff-test the clones against the live (pure) engine: a clone advanced
with the same actions must reproduce the original's trajectory exactly —
including hand refreshes, which exercise the captured generator RNG state.
"""
import random

import pytest

from engine.block import ALL_BLOCKS, LINE_2_DOWN, TWO_BY_TWO
from engine.game import GameState
from engine.grid import Grid
from engine.scoring import ComboScorer, SimpleScorer


def _play_random(state: GameState, rng: random.Random, n: int) -> list[tuple]:
    """Play up to n legal moves chosen by `rng`; return the (slot,row,col) trace."""
    trace: list[tuple] = []
    for _ in range(n):
        if state.game_over:
            break
        action = rng.choice(state.legal_actions())
        state.place(*action)
        trace.append(action)
    return trace


def _full_state(state: GameState) -> tuple:
    """A comparable tuple of everything that defines play continuation."""
    return (
        tuple(tuple(row) for row in state.grid.to_matrix()),
        state.hand,
        state.score,
        state.combo,
        state.cleared_this_round,
        state.game_over,
    )


# --- Grid.clone ---

def test_grid_clone_equal_and_independent() -> None:
    g = Grid(cells=[[0] * 8 for _ in range(8)])
    c = g.clone()
    assert c.to_matrix() == g.to_matrix()
    # mutating the clone must not touch the original
    c.place(LINE_2_DOWN, 1, 1)
    assert g.to_matrix() != c.to_matrix()
    assert g.is_empty(1, 1)


def test_grid_clone_skips_validation_but_matches_constructor() -> None:
    g = Grid(cells=[[0] * 8 for _ in range(8)])
    g.place(TWO_BY_TWO, 0, 0)
    assert g.clone().to_matrix() == g.to_matrix()


# --- scorer.clone ---

def test_combo_scorer_clone_captures_streak() -> None:
    s = ComboScorer()
    s.score_placement(4, 2)   # clears lines → cleared_this_round True
    s.end_round()             # N -> 1
    c = s.clone()
    assert (c.score, c.combo, c.cleared_this_round) == (s.score, s.combo, s.cleared_this_round)
    # independence
    c.score_placement(4, 1)
    assert c.score != s.score


def test_simple_scorer_clone_captures_score() -> None:
    s = SimpleScorer(cell_points=5, line_points=10)
    s.score_placement(3, 1)
    c = s.clone()
    assert c.score == s.score
    c.score_placement(2, 0)
    assert c.score != s.score


# --- GameState.clone: trajectory equivalence ---

def test_clone_reproduces_trajectory_in_lockstep() -> None:
    original = GameState(seed=7, pool=ALL_BLOCKS)
    clone = original.clone()
    assert _full_state(clone) == _full_state(original)

    # Drive both with the same action choices; the actions are derived from each
    # state's own legal_actions() so this only matches if the clone is a true copy.
    rng_o, rng_c = random.Random(123), random.Random(123)
    for _ in range(60):
        if original.game_over:
            break
        a_o = rng_o.choice(original.legal_actions())
        a_c = rng_c.choice(clone.legal_actions())
        assert a_o == a_c
        original.place(*a_o)
        clone.place(*a_c)
        assert _full_state(clone) == _full_state(original)


def test_clone_future_hands_match_after_refresh() -> None:
    """The captured RNG must make the clone deal the same future hands."""
    original = GameState(seed=11, pool=ALL_BLOCKS)
    # advance a few moves so the generator RNG has been consumed at least once
    _play_random(original, random.Random(1), 5)

    clone = original.clone()
    trace = _play_random(original, random.Random(2), 40)
    # replay the exact same actions on the clone
    for action in trace:
        clone.place(*action)
    assert _full_state(clone) == _full_state(original)


def test_mutating_clone_does_not_touch_original() -> None:
    original = GameState(seed=3, pool=ALL_BLOCKS)
    before = _full_state(original)
    clone = original.clone()
    _play_random(clone, random.Random(99), 30)
    assert _full_state(original) == before          # original frozen
    assert _full_state(clone) != before             # clone advanced


def test_clone_of_clone_is_independent() -> None:
    s0 = GameState(seed=5)
    s1 = s0.clone()
    s2 = s1.clone()
    _play_random(s1, random.Random(1), 10)
    # s2 was cloned from s1 *before* s1 was mutated → still equals s0
    assert _full_state(s2) == _full_state(s0)


# --- restore() ---

def test_restore_undoes_a_rollout() -> None:
    state = GameState(seed=9, pool=ALL_BLOCKS)
    _play_random(state, random.Random(1), 8)
    saved = state.clone()
    checkpoint = _full_state(state)

    _play_random(state, random.Random(2), 20)
    assert _full_state(state) != checkpoint         # rollout changed state

    state.restore(saved)
    assert _full_state(state) == checkpoint          # back to the checkpoint


def test_restore_source_reusable() -> None:
    state = GameState(seed=4)
    saved = state.clone()
    checkpoint = _full_state(state)
    # restore twice from the same snapshot, mutating in between
    for _ in range(3):
        _play_random(state, random.Random(7), 5)
        state.restore(saved)
        assert _full_state(state) == checkpoint


def test_restored_state_continues_identically() -> None:
    """After restore(), future deals/plays must match a fresh clone's."""
    state = GameState(seed=8, pool=ALL_BLOCKS)
    _play_random(state, random.Random(1), 6)
    saved = state.clone()

    # reference continuation from a separate clone of the same checkpoint
    reference = saved.clone()
    trace = _play_random(reference, random.Random(5), 25)

    state.restore(saved)
    for action in trace:
        if state.game_over:
            break
        state.place(*action)
    assert _full_state(state) == _full_state(reference)


# --- resample_hand: chance-node sampling primitive (Phase 3a search) ---

def test_resample_hand_keeps_board_and_score() -> None:
    """Resampling redeals the hand only — board, score, combo are untouched."""
    state = GameState(seed=2, pool=ALL_BLOCKS)
    _play_random(state, random.Random(1), 7)
    sample = state.resample_hand(seed=123)
    assert sample.grid.to_matrix() == state.grid.to_matrix()
    assert sample.score == state.score
    assert sample.combo == state.combo


def test_resample_hand_does_not_touch_original() -> None:
    state = GameState(seed=6, pool=ALL_BLOCKS)
    before = _full_state(state)
    state.resample_hand(seed=42)
    assert _full_state(state) == before


def test_resample_hand_is_seed_deterministic() -> None:
    state = GameState(seed=6, pool=ALL_BLOCKS)
    a = state.resample_hand(seed=42)
    b = state.resample_hand(seed=42)
    assert a.hand == b.hand


def test_resample_hand_spans_the_distribution() -> None:
    """Different seeds must produce a spread of hands (real MC sampling)."""
    state = GameState(seed=6, pool=ALL_BLOCKS)
    hands = {state.resample_hand(seed=s).hand for s in range(50)}
    assert len(hands) > 1


def test_resampled_hand_is_independently_playable() -> None:
    """A resampled state can be played forward without disturbing the source."""
    state = GameState(seed=10, pool=ALL_BLOCKS)
    _play_random(state, random.Random(1), 4)
    before = _full_state(state)
    sample = state.resample_hand(seed=7)
    if not sample.game_over:
        _play_random(sample, random.Random(3), 10)
    assert _full_state(state) == before
