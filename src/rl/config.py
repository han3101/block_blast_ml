"""Training configuration: schema, YAML load, CLI-override merge.

Resolution order: built-in defaults → YAML file → CLI overrides (later wins).
The merged config is frozen after construction so it can be safely shared and saved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    # PPO
    total_timesteps: int = 2_000_000
    n_envs: int = 16
    rollout_steps: int = 128
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Optimizer
    lr: float = 3e-4
    lr_warmup_frac: float = 0.05
    target_kl: float = 0.02
    # Reward shaping
    line_clear_bonus: float = 0.0
    game_over_penalty: float = 0.0
    # Environment
    env_mode: str = "at_least_one"
    seed: int = 42
    # Hardware
    device: str = "auto"
    amp: bool = False
    compile: bool = False
    # Logging / checkpointing
    log_interval: int = 1
    checkpoint_interval: int = 100
    keep_checkpoints: int = 5

    @classmethod
    def build(
        cls,
        yaml_path: Path | str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "TrainConfig":
        valid = {f.name for f in fields(cls)}
        merged: dict[str, Any] = {f.name: f.default for f in fields(cls)}

        if yaml_path is not None:
            data = yaml.safe_load(Path(yaml_path).read_text()) or {}
            merged.update({k: v for k, v in data.items() if k in valid})

        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None and k in valid})

        return cls(**merged)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))
