"""PPO training entrypoint.

Usage:
    uv run python -m rl.train_ppo                                     # defaults
    uv run python -m rl.train_ppo --config config/long_train.yaml
    uv run python -m rl.train_ppo --total-timesteps 100000 --n-envs 8  # smoke test
    uv run python -m rl.train_ppo --resume runs/<run_id>/checkpoints/ckpt_<step>.pt
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import gymnasium

from rl.config import TrainConfig
from rl.env import AutoResetEnv, BlockBlastEnv
from rl.hardware import default_n_envs, detect_device, summary
from rl.train_common import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Block Blast PPO agent")
    p.add_argument("--config", type=Path, default=None, help="YAML config file (default: config/default.yaml)")
    p.add_argument("--total-timesteps", type=int,   dest="total_timesteps")
    p.add_argument("--n-envs",          type=int,   dest="n_envs")
    p.add_argument("--batch-size",      type=int,   dest="batch_size")
    p.add_argument("--rollout-steps",   type=int,   dest="rollout_steps")
    p.add_argument("--n-epochs",        type=int,   dest="n_epochs")
    p.add_argument("--lr",              type=float)
    p.add_argument("--device",          type=str)
    p.add_argument("--amp",    action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--compile",action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--seed",            type=int)
    p.add_argument("--env-mode",        type=str,   dest="env_mode")
    p.add_argument("--line-clear-bonus",type=float, dest="line_clear_bonus")
    p.add_argument("--game-over-penalty",type=float,dest="game_over_penalty")
    p.add_argument("--checkpoint-interval", type=int, dest="checkpoint_interval")
    p.add_argument("--keep-checkpoints",    type=int, dest="keep_checkpoints")
    p.add_argument("--log-interval",        type=int, dest="log_interval")
    p.add_argument("--resume",              type=Path, default=None)
    p.add_argument("--run-id",          type=str,   default=None, dest="run_id")
    return p.parse_args()


def _make_envs(cfg: TrainConfig):
    def _factory(idx: int):
        def _init():
            env = BlockBlastEnv(
                seed=cfg.seed + idx,
                mode=cfg.env_mode,
                line_clear_bonus=cfg.line_clear_bonus,
                game_over_penalty=cfg.game_over_penalty,
            )
            return AutoResetEnv(env)
        return _init

    return gymnasium.vector.SyncVectorEnv([_factory(i) for i in range(cfg.n_envs)])


if __name__ == "__main__":
    args = _parse_args()

    # Resolve device first so hardware defaults can use it
    requested_device = args.device or "auto"
    resolved_device = detect_device(requested_device)

    overrides: dict = {k: v for k, v in vars(args).items()
                       if v is not None and k not in ("config", "resume", "run_id")}
    overrides["device"] = resolved_device
    if "n_envs" not in overrides:
        overrides["n_envs"] = default_n_envs()

    yaml_path = args.config or Path("config/default.yaml")
    cfg = TrainConfig.build(yaml_path=yaml_path, overrides=overrides)

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass

    run_id = args.run_id or f"{datetime.now():%Y%m%d_%H%M%S}-{git_sha}"
    project_root = Path(__file__).parent.parent.parent
    run_dir = project_root / "runs" / run_id

    print(f"run_id: {run_id}")
    print(f"hardware: {summary()}")

    train(cfg, _make_envs, run_dir, resume_from=args.resume)
