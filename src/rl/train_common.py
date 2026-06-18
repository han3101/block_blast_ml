"""Shared PPO training loop: rollout buffer, update, logging, checkpointing."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, cast

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rl.config import TrainConfig
from rl.encoding import NUM_ACTIONS
from rl.policy import BlockBlastPolicy

_OBS_SHAPE = (4, 8, 8)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one rollout (T steps × N envs) and computes GAE advantages."""

    def __init__(self, rollout_steps: int, n_envs: int) -> None:
        T, N = rollout_steps, n_envs
        self.T, self.N = T, N
        self.obs      = np.zeros((T, N, *_OBS_SHAPE), dtype=np.float32)
        self.actions  = np.zeros((T, N), dtype=np.int64)
        self.rewards  = np.zeros((T, N), dtype=np.float32)
        self.dones    = np.zeros((T, N), dtype=np.float32)
        self.values   = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.masks    = np.zeros((T, N, NUM_ACTIONS), dtype=bool)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns    = np.zeros((T, N), dtype=np.float32)

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """GAE in my convention: dones[t] = done flag AFTER step t.

        When dones[t]=1, the episode ended at step t. The next obs (at t+1 or
        last_value) belongs to a new episode; (1-dones[t]) zeros out that
        bootstrap contribution so the two episodes don't bleed into each other.
        """
        last_gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def as_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Flatten (T, N, ...) → (T*N, ...) and move to device."""
        B = self.T * self.N
        return {
            "obs":        torch.from_numpy(self.obs.reshape(B, *_OBS_SHAPE)).to(device),
            "actions":    torch.from_numpy(self.actions.reshape(B)).to(device),
            "log_probs":  torch.from_numpy(self.log_probs.reshape(B)).to(device),
            "advantages": torch.from_numpy(self.advantages.reshape(B)).to(device),
            "returns":    torch.from_numpy(self.returns.reshape(B)).to(device),
            "masks":      torch.from_numpy(self.masks.reshape(B, NUM_ACTIONS)).to(device),
        }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_dir: Path,
    policy: BlockBlastPolicy,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    keep: int,
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / f"ckpt_{global_step:010d}.pt"
    torch.save(
        {"global_step": global_step, "policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )
    for old in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-keep]:
        old.unlink(missing_ok=True)
    return path


def save_best(run_dir: Path, policy: BlockBlastPolicy, score: float) -> None:
    torch.save({"policy": policy.state_dict(), "score": score}, run_dir / "best_model.pt")


def load_checkpoint(
    path: Path,
    policy: BlockBlastPolicy,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(path, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["global_step"])


# ---------------------------------------------------------------------------
# Info extraction (gymnasium vector env compatibility)
# ---------------------------------------------------------------------------

def _extract_masks(infos: dict | list, n_envs: int) -> np.ndarray:
    """Pull action_mask out of gymnasium vector env info dict."""
    if isinstance(infos, dict):
        masks = infos.get("action_mask")
        if masks is not None:
            arr = np.asarray(masks, dtype=bool)
            if arr.shape == (n_envs, NUM_ACTIONS):
                return arr
    if isinstance(infos, (list, tuple)) and len(infos) == n_envs:
        return np.array(
            [info.get("action_mask", np.ones(NUM_ACTIONS, dtype=bool)) for info in infos],
            dtype=bool,
        )
    # shouldn't happen if env is correct, but fail safe > silent wrong
    raise RuntimeError(
        f"Could not extract action_mask from infos (type={type(infos)}). "
        "Make sure the env returns 'action_mask' in its info dict."
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    make_envs: Callable[["TrainConfig"], gymnasium.vector.VectorEnv],
    run_dir: Path,
    resume_from: Path | None = None,
) -> None:
    device = torch.device(
        cfg.device if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    envs = make_envs(cfg)
    policy = BlockBlastPolicy().to(device)

    if cfg.compile:
        try:
            policy = cast(BlockBlastPolicy, torch.compile(policy))
        except Exception as e:
            print(f"torch.compile unavailable ({e}), continuing without it")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    use_autocast = cfg.amp and device.type in ("cuda", "mps")
    use_scaler = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore[attr-defined]

    # LR: linear warmup then linear decay to 0
    total_rollouts = max(1, cfg.total_timesteps // (cfg.n_envs * cfg.rollout_steps))
    warmup_rollouts = int(total_rollouts * cfg.lr_warmup_frac)

    def lr_lambda(rollout_idx: int) -> float:
        if rollout_idx < warmup_rollouts:
            return rollout_idx / max(1, warmup_rollouts)
        remaining = total_rollouts - warmup_rollouts
        return max(0.0, 1.0 - (rollout_idx - warmup_rollouts) / max(1, remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_mean_reward = -float("inf")
    if resume_from:
        global_step = load_checkpoint(resume_from, policy, optimizer)
        print(f"resumed from {resume_from} at step {global_step:,}")

    obs_np, infos = envs.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])  # type: ignore[arg-type]
    mask_np = _extract_masks(infos, cfg.n_envs)

    buf = RolloutBuffer(cfg.rollout_steps, cfg.n_envs)

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_rew_running = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_len_running = np.zeros(cfg.n_envs, dtype=np.int32)
    done_np = np.zeros(cfg.n_envs, dtype=np.float32)

    rollout_idx = 0
    t_start = time.perf_counter()

    print(
        f"Training on {device} | {cfg.n_envs} envs | "
        f"{cfg.total_timesteps:,} steps | amp={use_autocast}"
    )

    while global_step < cfg.total_timesteps:
        frac = global_step / cfg.total_timesteps
        ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac

        # ---- rollout ----
        policy.eval()
        for t in range(cfg.rollout_steps):
            with torch.no_grad():
                obs_t  = torch.from_numpy(obs_np).to(device)
                mask_t = torch.from_numpy(mask_np).to(device)
                action_t, log_prob_t, _, value_t = policy.get_action_and_value(obs_t, mask_t)

            action_np = action_t.cpu().numpy()
            obs_next, rew_np, term_np, trunc_np, infos_next = envs.step(action_np)
            mask_next = _extract_masks(infos_next, cfg.n_envs)
            done_np = (term_np | trunc_np).astype(np.float32)

            buf.obs[t]       = obs_np
            buf.actions[t]   = action_np
            buf.rewards[t]   = rew_np
            buf.dones[t]     = done_np
            buf.values[t]    = value_t.cpu().numpy()
            buf.log_probs[t] = log_prob_t.cpu().numpy()
            buf.masks[t]     = mask_np

            ep_rew_running += rew_np
            ep_len_running += 1
            for i in np.where(done_np)[0]:
                ep_rewards.append(float(ep_rew_running[i]))
                ep_lengths.append(int(ep_len_running[i]))
                ep_rew_running[i] = 0.0
                ep_len_running[i] = 0

            obs_np, mask_np = obs_next, mask_next
            global_step += cfg.n_envs

        # bootstrap value for the state after the rollout
        with torch.no_grad():
            _, _, _, last_val = policy.get_action_and_value(
                torch.from_numpy(obs_np).to(device),
                torch.from_numpy(mask_np).to(device),
            )
        buf.compute_gae(last_val.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        # ---- PPO update ----
        policy.train()
        flat = buf.as_tensors(device)
        B = cfg.n_envs * cfg.rollout_steps

        pg_losses, vf_losses, entropies, kls, clip_fracs = [], [], [], [], []
        stop_early = False

        for _ in range(cfg.n_epochs):
            if stop_early:
                break
            for mb_idx in torch.randperm(B, device=device).split(cfg.batch_size):
                mb_obs   = flat["obs"][mb_idx]
                mb_act   = flat["actions"][mb_idx]
                mb_lp    = flat["log_probs"][mb_idx]
                mb_adv   = flat["advantages"][mb_idx]
                mb_ret   = flat["returns"][mb_idx]
                mb_mask  = flat["masks"][mb_idx]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with torch.autocast(device_type=device.type, enabled=use_autocast):
                    _, new_lp, entropy, new_val = policy.get_action_and_value(
                        mb_obs, mb_mask, mb_act
                    )
                    log_ratio = new_lp - mb_lp
                    ratio = log_ratio.exp()
                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                    ).mean()
                    vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + cfg.vf_coef * vf_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)
                clip_fracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

                if cfg.target_kl and approx_kl > cfg.target_kl:
                    stop_early = True
                    break

        scheduler.step()
        rollout_idx += 1

        # ---- logging ----
        if rollout_idx % cfg.log_interval == 0 and ep_rewards:
            recent_rew = ep_rewards[-100:]
            recent_len = ep_lengths[-100:]
            mean_rew   = float(np.mean(recent_rew))
            max_rew    = float(np.max(recent_rew))
            mean_len   = float(np.mean(recent_len))
            elapsed    = time.perf_counter() - t_start
            sps        = global_step / elapsed
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            mean_pg   = float(np.mean(pg_losses))
            mean_vf   = float(np.mean(vf_losses))
            mean_ent  = float(np.mean(entropies))
            mean_kl   = float(np.mean(kls))
            mean_clip = float(np.mean(clip_fracs))
            total_loss = mean_pg + cfg.vf_coef * mean_vf - ent_coef * mean_ent

            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                save_best(run_dir, policy, mean_rew)

            writer.add_scalar("train/avg_score",        mean_rew,   global_step)
            writer.add_scalar("train/max_score",        max_rew,    global_step)
            writer.add_scalar("train/best_score",       best_mean_reward, global_step)
            writer.add_scalar("train/avg_length",       mean_len,   global_step)
            writer.add_scalar("train/policy_loss",      mean_pg,    global_step)
            writer.add_scalar("train/value_loss",       mean_vf,    global_step)
            writer.add_scalar("train/entropy",          mean_ent,   global_step)
            writer.add_scalar("train/total_loss",       total_loss, global_step)
            writer.add_scalar("train/approx_kl",        mean_kl,    global_step)
            writer.add_scalar("train/clip_fraction",    mean_clip,  global_step)
            writer.add_scalar("train/ent_coef",         ent_coef,   global_step)
            writer.add_scalar("train/lr",               scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("perf/fps",               sps,        global_step)

            print(
                f"[Step {global_step:,}] [{elapsed_str}]\n"
                f"  fps:            {sps:.0f}\n"
                f"  avg_score:      {mean_rew:.4f}\n"
                f"  max_score:      {max_rew:.0f}\n"
                f"  best_score:     {best_mean_reward:.4f}\n"
                f"  avg_length:     {mean_len:.4f}\n"
                f"  policy_loss:    {mean_pg:.4f}\n"
                f"  value_loss:     {mean_vf:.4f}\n"
                f"  entropy:        {mean_ent:.4f}\n"
                f"  total_loss:     {total_loss:.4f}\n"
                f"  approx_kl:      {mean_kl:.4f}\n"
                f"  clip_fraction:  {mean_clip:.4f}\n"
            )

        # ---- checkpoint ----
        if rollout_idx % cfg.checkpoint_interval == 0:
            save_checkpoint(run_dir, policy, optimizer, global_step, cfg.keep_checkpoints)

    save_checkpoint(run_dir, policy, optimizer, global_step, cfg.keep_checkpoints)
    writer.close()
    envs.close()
    print(f"done. run saved to {run_dir}")
