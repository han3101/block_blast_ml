"""Tests for the Phase 3a inference-time lookahead search agent.

These cover correctness (the agent always commits a legal move and never raises
mid-game), the headline property (exact within-hand lookahead beats myopic
greedy), and that the expectimax path (horizon >= 2, chance nodes over resampled
hands) runs end to end. Configs are kept small (low beam, few games) so the
search stays fast enough for the unit suite.
"""
import random

import pytest

from engine.game import GameState
from rl.agents.greedy import choose_action as greedy_choose
from rl.agents.search import (
    SearchConfig,
    choose_action,
    greedy_health_leaf,
    make_search_agent,
    score_only_leaf,
)
from rl.encoding import decode_action, encode_action


def _legal_encoded(state: GameState) -> set[int]:
    return {encode_action(*a) for a in state.legal_actions()}


def _play_to_end(agent, seed: int, mode: str = "at_least_one", cap: int = 500) -> GameState:
    state = GameState(seed=seed, mode=mode)
    steps = 0
    while not state.game_over and steps < cap:
        action = agent(state)
        assert action in _legal_encoded(state)  # every committed action must be legal
        state.place(*decode_action(action))
        steps += 1
    return state


# --- leaf evaluators (fast unit checks) ------------------------------------ #

def test_score_only_leaf_is_zero() -> None:
    state = GameState(seed=0)
    assert score_only_leaf(state, random.Random(0)) == 0.0


def test_greedy_health_leaf_is_nonpositive() -> None:
    # 0 on an empty board; strictly negative once cells are occupied.
    empty = GameState(seed=0)
    assert greedy_health_leaf(empty, random.Random(0)) == 0.0
    played = empty.clone()
    played.place(*played.legal_actions()[0])
    assert greedy_health_leaf(played, random.Random(0)) < 0.0


# --- agent correctness ------------------------------------------------------ #

def test_agent_commits_only_legal_moves_and_finishes() -> None:
    agent = make_search_agent("greedy_health", cfg=SearchConfig(horizon_hands=1, beam=8))
    state = _play_to_end(agent, seed=1)
    assert state.game_over  # ran to a real terminal within the cap


def test_choose_action_raises_on_terminal() -> None:
    # A state with no legal actions must be rejected (caller should gate on game_over).
    state = GameState(seed=0)
    # Drive to game over with greedy, then assert the search refuses to act.
    while not state.game_over:
        state.place(*decode_action(greedy_choose(state)))
    cfg = SearchConfig(horizon_hands=1, beam=4)
    with pytest.raises(ValueError):
        choose_action(state, cfg, score_only_leaf, random.Random(0))


# --- the headline property: exact lookahead beats myopic greedy ------------- #

@pytest.mark.parametrize("seed", [0, 3])
def test_search_beats_greedy(seed: int) -> None:
    search_agent = make_search_agent("greedy_health", cfg=SearchConfig(horizon_hands=1, beam=12))
    search_state = _play_to_end(search_agent, seed=seed)

    greedy_state = GameState(seed=seed)
    while not greedy_state.game_over:
        greedy_state.place(*decode_action(greedy_choose(greedy_state)))

    assert search_state.score > greedy_state.score


# --- expectimax path runs (horizon >= 2, chance nodes) ---------------------- #

def test_expectimax_horizon2_returns_legal_action() -> None:
    agent = make_search_agent(
        "score_only", cfg=SearchConfig(horizon_hands=2, beam=4, samples=2), seed=0
    )
    state = GameState(seed=2)
    action = agent(state)
    assert action in _legal_encoded(state)


def test_value_net_leaf_requires_checkpoint() -> None:
    with pytest.raises(ValueError):
        make_search_agent("value_net", checkpoint=None)
