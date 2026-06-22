"""Inference-time lookahead search agent (Phase 3a).

Wraps a Phase 2 policy/value net in planning at play time — no retraining. The
mechanism (see plans/phase-3.md): a reactive policy does one forward pass per
move and so (1) walks into dead-ends and (2) misses multi-move combo setups.
Search uses the *exact* cloneable engine to try moves and read the resulting
board before committing.

Two regimes, kept deliberately separate because the value net is the weak link
(explained_variance ~0.16):

  * **Within the current hand — exact.** The 3 pieces are known and no refill
    happens until all are placed, so the subtree is deterministic. We beam-search
    over placements *and orderings* and score each line by the *exact engine
    score* it banks. No approximation.
  * **Across the next hand — expectimax.** The next hand is stochastic, so at a
    hand boundary we Monte-Carlo the real next-hand distribution
    (``GameState.resample_hand``) and average. The leaf evaluator only ever
    estimates survivability *beyond* what we searched.

The leaf evaluator is **swappable** (the Phase 3a bake-off / value-head
diagnostic): ``score_only`` (myopic — exact score only), ``greedy_health``
(greedy's board-health heuristic), or ``value_net`` (the learned critic). The
objective at a horizon leaf is always ``exact_score(state) + leaf_future(state)``
— exact points for what we know, an estimate for what we don't.

Fairness note: ``clone()`` copies the RNG bit-state, so a cloned rollout that
exhausts its hand deals the *real* next hand the live game would. The search must
not exploit that (the agent can't see the future hand in real play), so every
next-hand read goes through ``resample_hand`` — both at expectimax chance nodes
and when the value-net leaf needs a hand to encode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from engine.game import GameState
from rl.encoding import _count_holes, encode_action, encode_aux, encode_obs

# A leaf evaluator estimates the value of the game *beyond* the searched horizon.
# It returns a future bonus in (roughly) score units; the searched exact score is
# added separately by the caller. ``rng`` lets it resample hands without peeking.
LeafFn = Callable[[GameState, random.Random], float]

_RNG_MAX = 2**31 - 1


@dataclass(frozen=True)
class SearchConfig:
    """Runtime knobs for the lookahead search (depth/width vs strength/latency).

    horizon_hands: hands to plan through. 1 = play the current hand exactly then
        leaf-evaluate the board (cheap, the high-ROI floor; no chance nodes). 2+
        adds expectimax over that many future hands.
    beam: max board states kept between within-hand plies / boundary end-states.
    samples: next-hand Monte-Carlo samples per chance node (horizon_hands>=2).
    value_weight: scale applied to the value-net leaf (its output is normalized
        shaped return; this converts it toward the score units of the exact part).
    leaf_samples: hands resampled & averaged when a leaf needs a hand (value_net).
    """

    horizon_hands: int = 1
    beam: int = 16
    samples: int = 8
    value_weight: float = 30.0
    leaf_samples: int = 1


# --------------------------------------------------------------------------- #
# Board health (shared by the beam proxy and the greedy_health leaf)
# --------------------------------------------------------------------------- #

def _board_health(state: GameState) -> float:
    """A cheap board-quality penalty: fewer occupied cells and holes is better.

    Returned as a non-positive number (0 = empty board). Used both as the
    within-hand beam ranking proxy and as the ``greedy_health`` leaf future."""
    matrix = state.grid.to_matrix()
    occupied = state.grid.count_occupied()  # bitboard popcount, not a cell sum
    holes = _count_holes(matrix)
    return -(float(occupied) + 2.0 * holes)


def _beam_key(state: GameState) -> float:
    """Rank within-hand lines by banked exact score plus board cleanliness."""
    return float(state.score) + _board_health(state)


# --------------------------------------------------------------------------- #
# Leaf evaluators (swappable — the Phase 3a bake-off)
# --------------------------------------------------------------------------- #

def score_only_leaf(state: GameState, rng: random.Random) -> float:
    """Myopic: no future estimate. The objective collapses to horizon score."""
    return 0.0


def greedy_health_leaf(state: GameState, rng: random.Random) -> float:
    """Greedy's board-health heuristic as the future estimate (board-only)."""
    return _board_health(state)


def make_value_leaf(
    checkpoint: str | Path,
    device: str = "cpu",
    weight: float = 30.0,
    leaf_samples: int = 1,
) -> LeafFn:
    """Build a value-net leaf from a Phase 2 checkpoint.

    Resamples ``leaf_samples`` next hands (avoiding the cloned-RNG peek) and
    averages the critic's value, scaled by ``weight`` into score-ish units.
    """
    import torch

    from rl.policy import BlockBlastPolicy

    dev = torch.device(device)
    policy = BlockBlastPolicy().to(dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    @torch.no_grad()
    def leaf(state: GameState, rng: random.Random) -> float:
        states = [state.resample_hand(rng.randint(0, _RNG_MAX)) for _ in range(leaf_samples)]
        board = torch.tensor(
            [encode_obs(s) for s in states], dtype=torch.float32, device=dev
        )
        aux = torch.tensor(
            [encode_aux(s) for s in states], dtype=torch.float32, device=dev
        )
        _, value = policy(board, aux)
        return weight * float(value.mean().item())

    return leaf


LEAF_FACTORIES: dict[str, Callable[..., LeafFn]] = {
    "score_only": lambda **_: score_only_leaf,
    "greedy_health": lambda **_: greedy_health_leaf,
    # value_net is built via make_value_leaf (needs a checkpoint).
}


# --------------------------------------------------------------------------- #
# Search core
# --------------------------------------------------------------------------- #

def _beam(items: list[tuple[GameState, object]], k: int) -> list[tuple[GameState, object]]:
    if len(items) <= k:
        return items
    items.sort(key=lambda sa: _beam_key(sa[0]), reverse=True)
    return items[:k]


def _within_hand_endstates(
    root: GameState, beam: int
) -> list[tuple[GameState, tuple[int, int, int]]]:
    """Beam-search every way to play out ``root``'s current hand.

    Returns ``(end_state, first_action)`` pairs where ``end_state`` is either a
    state whose hand just exhausted (a chance-node boundary, hand left empty) or
    a terminal dead-end mid-hand. ``first_action`` is the move taken from ``root``
    that began the line, so the caller can recover the action to play.

    Deterministic: no RNG — the current hand is fully known. We place with
    ``deal_next=False``: nobody downstream reads the boundary hand (leaves look at
    the board; expectimax resamples), so dealing it is pure waste — skipping it is
    the search's single biggest speedup. See ``GameState.place``.
    """
    # (state, first_action); None marks the root level (first move not chosen yet).
    active: list[tuple[GameState, tuple[int, int, int] | None]] = [(root, None)]
    completed: list[tuple[GameState, tuple[int, int, int]]] = []

    while active:
        next_active: list[tuple[GameState, tuple[int, int, int] | None]] = []
        for state, first in active:
            for action in state.legal_actions():
                child = state.clone()
                result = child.place(*action, deal_next=False)
                line_first = action if first is None else first
                if child.game_over or result.hand_refreshed:
                    completed.append((child, line_first))
                else:
                    next_active.append((child, line_first))
        active = _beam(next_active, beam)
        completed = _beam(completed, beam)  # keep only the best end-states

    return completed


def _boundary_value(
    end_state: GameState,
    hands_remaining: int,
    cfg: SearchConfig,
    leaf: LeafFn,
    rng: random.Random,
) -> float:
    """Value of a within-hand line's end-state (already played the hand)."""
    if end_state.game_over:
        return float(end_state.score)  # dead-ended: exact, no future
    if hands_remaining <= 1:
        # Horizon: exact banked score + estimated survivability beyond it.
        return float(end_state.score) + leaf(end_state, rng)
    # Chance node: expectimax over the real next-hand distribution.
    total = 0.0
    for _ in range(cfg.samples):
        sample = end_state.resample_hand(rng.randint(0, _RNG_MAX))
        if sample.game_over:
            total += float(sample.score)
        else:
            total += _best_line(sample, hands_remaining - 1, cfg, leaf, rng)[0]
    return total / cfg.samples


def _best_line(
    state: GameState,
    hands_remaining: int,
    cfg: SearchConfig,
    leaf: LeafFn,
    rng: random.Random,
) -> tuple[float, tuple[int, int, int] | None]:
    """Best (value, first_action) over all ways to play ``state``'s hand."""
    ends = _within_hand_endstates(state, cfg.beam)
    if not ends:
        return float(state.score), None
    best_value = float("-inf")
    best_first: tuple[int, int, int] | None = None
    for end_state, first in ends:
        value = _boundary_value(end_state, hands_remaining, cfg, leaf, rng)
        if value > best_value:
            best_value, best_first = value, first
    return best_value, best_first


def choose_action(
    state: GameState,
    cfg: SearchConfig,
    leaf: LeafFn,
    rng: random.Random,
) -> int:
    """Return the encoded action the search judges best from ``state``."""
    _, first = _best_line(state, cfg.horizon_hands, cfg, leaf, rng)
    if first is None:
        raise ValueError("no legal actions — check state.game_over before searching")
    return encode_action(*first)


# --------------------------------------------------------------------------- #
# Agent-seam factory (matches make_policy_agent: state -> encoded action)
# --------------------------------------------------------------------------- #

def make_search_agent(
    leaf_kind: str = "greedy_health",
    checkpoint: str | Path | None = None,
    cfg: SearchConfig | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> Callable[[GameState], int]:
    """Build a ``choose_action(state) -> int`` search agent.

    leaf_kind: ``"score_only"``, ``"greedy_health"`` or ``"value_net"``.
    checkpoint: required for ``value_net``.
    """
    cfg = cfg or SearchConfig()
    if leaf_kind == "value_net":
        if not checkpoint:
            raise ValueError("value_net leaf requires a checkpoint path")
        leaf = make_value_leaf(
            checkpoint, device=device, weight=cfg.value_weight, leaf_samples=cfg.leaf_samples
        )
    elif leaf_kind in LEAF_FACTORIES:
        leaf = LEAF_FACTORIES[leaf_kind]()
    else:
        raise ValueError(f"unknown leaf_kind: {leaf_kind!r}")

    rng = random.Random(seed)

    def agent(state: GameState) -> int:
        return choose_action(state, cfg, leaf, rng)

    return agent
