"""Residual CNN policy + value network for Block Blast.

Architecture:
  board (batch, 4, 8, 8) → conv(64→128→128) residual blocks with GroupNorm
                         → Flatten → FC(512)            ┐
  aux   (batch, AUX_DIM) → MLP(64→64)                   ┘→ concat → FC(256)
                         → policy head (192 logits) + value head (1)

GroupNorm (not BatchNorm) is used deliberately: PPO collects rollouts under
policy.eval() and updates under policy.train(); BatchNorm's running-vs-minibatch
statistic gap makes new_logprob != old_logprob even at ratio==1, biasing the
clipped objective. GroupNorm behaves identically in train/eval, removing that
confound.

The aux branch feeds combo state and board summary scalars (see encode_aux) so
the value head can fit returns for visually identical boards with different
combo multipliers.

get_action_and_value() applies the action mask before sampling and computes
masked entropy (illegal actions contribute 0 to entropy via -inf logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.encoding import AUX_DIM, NUM_ACTIONS

_IN_CHANNELS = 4   # board + 3 hand slots
_GRID = 8
_GN_GROUPS = 8     # GroupNorm groups (divides 64 and 128 evenly)


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(_GN_GROUPS, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(_GN_GROUPS, ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class BlockBlastPolicy(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, aux_dim: int = AUX_DIM) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(_IN_CHANNELS, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_GN_GROUPS, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(_GN_GROUPS, 128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(_GN_GROUPS, 128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            nn.Flatten(),
            nn.Linear(128 * _GRID * _GRID, 512),
            nn.ReLU(inplace=True),
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self, board: torch.Tensor, aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([self.backbone(board), self.aux_mlp(aux)], dim=-1)
        feat = self.shared(feat)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)

    def get_action_and_value(
        self,
        board: torch.Tensor,
        aux: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked forward pass.

        Args:
            board:  (B, 4, 8, 8) float32
            aux:    (B, AUX_DIM) float32
            mask:   (B, 192) bool — True = legal
            action: (B,) long, optional; sampled from masked distribution if None

        Returns:
            action, log_prob, entropy, value  — all (B,)
        """
        logits, value = self(board, aux)
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
