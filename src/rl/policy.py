"""Residual CNN policy + value network for Block Blast.

Architecture (mirrors the reference):
  (batch, 4, 8, 8) → conv(64→128→128) with residual blocks + BatchNorm
  → Flatten → FC(512) → FC(256) → policy head (192 logits) + value head (1)

get_action_and_value() applies the action mask before sampling and computes
masked entropy (illegal actions contribute 0 to entropy naturally via -inf logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.encoding import NUM_ACTIONS

_IN_CHANNELS = 4   # board + 3 hand slots
_GRID = 8


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class BlockBlastPolicy(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(_IN_CHANNELS, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            nn.Flatten(),
            nn.Linear(128 * _GRID * _GRID, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked forward pass.

        Args:
            obs:    (B, 4, 8, 8) float32
            mask:   (B, 192) bool — True = legal
            action: (B,) long, optional; sampled from masked distribution if None

        Returns:
            action, log_prob, entropy, value  — all (B,)
        """
        logits, value = self(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
