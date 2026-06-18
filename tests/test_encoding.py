from __future__ import annotations

import pytest

from engine.block import LINE_2_DOWN, TWO_BY_TWO
from engine.game import GameState
from engine.grid import Grid
from rl.encoding import (
    NUM_ACTIONS,
    GRID_SIZE,
    NUM_SLOTS,
    decode_action,
    encode_action,
    encode_obs,
    action_mask,
)


# --- codec ---

def test_codec_roundtrip_all_actions() -> None:
    for action in range(NUM_ACTIONS):
        slot, row, col = decode_action(action)
        assert encode_action(slot, row, col) == action


def test_encode_action_values() -> None:
    assert encode_action(0, 0, 0) == 0
    assert encode_action(0, 0, 1) == 1
    assert encode_action(0, 1, 0) == GRID_SIZE
    assert encode_action(1, 0, 0) == GRID_SIZE * GRID_SIZE
    assert encode_action(2, 7, 7) == NUM_ACTIONS - 1


def test_decode_action_values() -> None:
    assert decode_action(0) == (0, 0, 0)
    assert decode_action(GRID_SIZE) == (0, 1, 0)
    assert decode_action(GRID_SIZE * GRID_SIZE) == (1, 0, 0)
    assert decode_action(NUM_ACTIONS - 1) == (NUM_SLOTS - 1, GRID_SIZE - 1, GRID_SIZE - 1)


# --- encode_obs shape and types ---

def test_obs_shape() -> None:
    state = GameState(seed=0)
    obs = encode_obs(state)
    assert len(obs) == 4
    for channel in obs:
        assert len(channel) == GRID_SIZE
        for row in channel:
            assert len(row) == GRID_SIZE


def test_obs_values_are_floats() -> None:
    state = GameState(seed=0)
    obs = encode_obs(state)
    for channel in obs:
        for row in channel:
            for val in row:
                assert isinstance(val, float)


def test_obs_values_are_zero_or_one() -> None:
    state = GameState(seed=0)
    obs = encode_obs(state)
    for channel in obs:
        for row in channel:
            for val in row:
                assert val in (0.0, 1.0)


# --- channel 0: board occupancy ---

def test_obs_channel0_matches_empty_board() -> None:
    state = GameState(seed=0)
    obs = encode_obs(state)
    assert obs[0] == [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]


def test_obs_channel0_reflects_placed_cells() -> None:
    state = GameState(seed=0, pool=(LINE_2_DOWN,))
    # Place at (0,0): cells (0,0) and (1,0) become occupied after clearing
    state.place(0, 0, 0)
    obs = encode_obs(state)
    assert obs[0][0][0] == 1.0
    assert obs[0][1][0] == 1.0
    assert obs[0][0][1] == 0.0


# --- channels 1-3: hand block planes ---

def test_obs_hand_planes_match_block_cells() -> None:
    state = GameState(seed=0)
    obs = encode_obs(state)
    for i, block in enumerate(state.hand):
        plane = obs[i + 1]
        if block is None:
            assert all(plane[r][c] == 0.0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))
        else:
            occupied = {(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if plane[r][c] == 1.0}
            assert occupied == set(block.cells)


def test_obs_empty_slot_is_all_zeros() -> None:
    # Use a pool where we can deplete a slot reliably
    state = GameState(seed=0, pool=(LINE_2_DOWN,))
    state.place(0, 0, 0)  # slot 0 → None (one block dealt per slot)
    obs = encode_obs(state)
    plane = obs[1]  # channel for slot 0
    assert all(plane[r][c] == 0.0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))


# --- action_mask ---

def test_mask_length() -> None:
    state = GameState(seed=0)
    mask = action_mask(state)
    assert len(mask) == NUM_ACTIONS


def test_mask_true_exactly_on_legal_actions() -> None:
    state = GameState(seed=0)
    legal = state.legal_actions()
    legal_indices = {encode_action(s, r, c) for s, r, c in legal}
    mask = action_mask(state)
    for i, val in enumerate(mask):
        assert val == (i in legal_indices)


def test_mask_all_false_when_game_over() -> None:
    # Force game over: use a block too large for any board position
    from engine.block import Block
    huge = Block("huge", [(r, c) for r in range(9) for c in range(9)])
    state = GameState(seed=0, pool=(huge,), mode="random")
    assert state.game_over
    mask = action_mask(state)
    assert not any(mask)


def test_mask_false_for_played_slot() -> None:
    state = GameState(seed=0, pool=(LINE_2_DOWN,))
    state.place(0, 0, 0)
    mask = action_mask(state)
    # All actions for slot 0 (indices 0..63) must be False
    assert not any(mask[: GRID_SIZE * GRID_SIZE])
