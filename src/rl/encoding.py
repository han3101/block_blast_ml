"""
Observation and action encoding for the RL boundary layer.

Observation shape: (4, 8, 8) — a 4-channel 8x8 "image".
  The 8x8 spatial dimensions map naturally to a CNN (convolutional nets learn
  local spatial patterns, which is exactly what block placement strategy requires).
  The 4 channels are:
    0   — board occupancy (the live grid, 0/1 per cell)
    1-3 — one plane per hand slot, with the block's shape stamped top-left anchored;
          an empty slot is an all-zero plane.
  Slots are 0-indexed (0, 1, 2), but channel 0 is taken by the board, so:
    slot i  ↔  channel i+1  ↔  actions i*64 … i*64+63
  This is implicit in encode_obs: channel0 is prepended, then hand slots are
  appended in order, so slot 0 lands at index 1, slot 1 at index 2, etc.
  GameState.legal_actions() returns (slot, row, col) triples in the same slot
  ordering, so action_mask consumes that directly with no remapping needed.

Action space: Discrete(192) = 3 slots x 8 rows x 8 cols.
  Encoded as: action = slot*64 + row*8 + col.

All functions return plain Python lists/floats — no torch dependency.
The env layer (env.py) converts to torch.float32 tensors at that boundary.
"""
from __future__ import annotations

from engine.game import GameState

GRID_SIZE = 8
NUM_SLOTS = 3
NUM_ACTIONS = NUM_SLOTS * GRID_SIZE * GRID_SIZE  # 192


def encode_action(slot: int, row: int, col: int) -> int:
    return slot * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col


def decode_action(action: int) -> tuple[int, int, int]:
    slot = action // (GRID_SIZE * GRID_SIZE)
    remainder = action % (GRID_SIZE * GRID_SIZE)
    row = remainder // GRID_SIZE
    col = remainder % GRID_SIZE
    return slot, row, col


def encode_obs(state: GameState) -> list[list[list[float]]]:
    """Return a (4, 8, 8) nested-list observation — no torch dependency.

    Channel 0: board occupancy.
    Channels 1-3: each hand slot's block shape rendered top-left anchored onto
    an 8x8 plane; an empty slot (None) is all zeros.
    """
    board = state.grid.to_matrix()
    channel0 = [[float(board[r][c]) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]

    channels = [channel0]
    for block in state.hand:
        plane = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
        if block is not None:
            for dr, dc in block.cells:
                plane[dr][dc] = 1.0
        channels.append(plane)

    return channels


def action_mask(state: GameState) -> list[bool]:
    """Return a 192-element bool list; True only at legal (slot, row, col) positions."""
    mask = [False] * NUM_ACTIONS
    for slot, row, col in state.legal_actions():
        mask[encode_action(slot, row, col)] = True
    return mask
