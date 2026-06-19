"""Trained-policy agent: loads a checkpoint and picks actions greedily (argmax)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from engine.game import GameState
from rl.encoding import NUM_ACTIONS, action_mask, encode_obs
from rl.policy import BlockBlastPolicy


def make_policy_agent(
    checkpoint: str | Path,
    device: str = "cpu",
) -> Callable[[GameState], int]:
    """Return a choose_action(state) -> int function backed by a saved checkpoint.

    Loads ``{"policy": state_dict, ...}`` format written by train_common.save_best /
    save_checkpoint.  Uses argmax (no sampling) for deterministic, comparable eval.
    """
    dev = torch.device(device)
    policy = BlockBlastPolicy().to(dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    @torch.no_grad()
    def choose_action(state: GameState) -> int:
        obs = torch.tensor(encode_obs(state), dtype=torch.float32, device=dev).unsqueeze(0)
        mask = torch.tensor(action_mask(state), dtype=torch.bool, device=dev).unsqueeze(0)
        logits, _ = policy(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        return int(logits.argmax(dim=-1).item())

    return choose_action
