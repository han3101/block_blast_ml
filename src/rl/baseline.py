"""Benchmark any agent over N seeded games and report score statistics.

Usage
-----
Greedy:
    uv run python -m rl.baseline --model greedy

Policy:
    uv run python -m rl.baseline --model policy --checkpoint runs/2b_fixed_2M/best_model.pt

Save results to JSON:
    uv run python -m rl.baseline --model policy --checkpoint <path> --save
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from engine.game import GameState
from engine.generator import Mode
from rl.agents.greedy import choose_action as greedy_choose
from rl.encoding import decode_action
from rl.types import BenchmarkResult


def _run_agent(
    agent_name: str,
    choose_action: Callable[[GameState], int],
    n_games: int,
    seed_offset: int,
    mode: Mode,
) -> BenchmarkResult:
    scores: list[int] = []
    lengths: list[int] = []
    for i in range(n_games):
        state = GameState(seed=seed_offset + i, mode=mode)
        steps = 0
        while not state.game_over:
            action = choose_action(state)
            slot, row, col = decode_action(action)
            state.place(slot, row, col)
            steps += 1
        scores.append(state.score)
        lengths.append(steps)
    return BenchmarkResult.from_plays(agent_name, scores, lengths, seed_offset=seed_offset)


def run_greedy_baseline(
    n_games: int = 100,
    seed_offset: int = 0,
    mode: Mode = "at_least_one",
) -> BenchmarkResult:
    return _run_agent("greedy", greedy_choose, n_games, seed_offset, mode)


def run_policy_baseline(
    checkpoint: str | Path,
    n_games: int = 100,
    seed_offset: int = 0,
    mode: Mode = "at_least_one",
    device: str = "cpu",
) -> BenchmarkResult:
    from rl.agents.policy_agent import make_policy_agent
    choose = make_policy_agent(checkpoint, device=device)
    name = f"policy:{Path(checkpoint).parent.name}"
    return _run_agent(name, choose, n_games, seed_offset, mode)


def _print_result(result: BenchmarkResult, elapsed: float) -> None:
    print(f"\n{result.agent}  ({result.n_games} games, {elapsed:.1f}s)")
    print(f"  score  — mean={result.mean:.1f}  median={result.median:.1f}  "
          f"stdev={result.stdev:.1f}  max={result.max}  min={result.min}")
    print(f"  length — mean={result.avg_length:.1f}  median={result.median_length:.1f}  "
          f"max={result.max_length}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Benchmark agents over N seeded games.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", choices=["greedy", "policy"], required=True,
                        help="which agent to benchmark")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a .pt checkpoint (required for --model policy)")
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--mode", choices=["at_least_one", "random", "solvable"],
                        default="at_least_one")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true", default=False,
                        help="write results to runs/<name>_<sha>_v<N>.json")
    args = parser.parse_args()

    if args.model == "policy" and not args.checkpoint:
        parser.error("--model policy requires --checkpoint")

    print(f"Running {args.model} over {args.n_games} games (seeds {args.seed_offset}–"
          f"{args.seed_offset + args.n_games - 1}, mode={args.mode}) …")
    t0 = time.perf_counter()

    if args.model == "greedy":
        result = run_greedy_baseline(args.n_games, args.seed_offset, args.mode)
    else:
        result = run_policy_baseline(
            args.checkpoint, args.n_games, args.seed_offset, args.mode, args.device
        )

    elapsed = time.perf_counter() - t0
    _print_result(result, elapsed)

    if args.save:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        timestamp = datetime.now(timezone.utc).isoformat()
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        runs_dir.mkdir(exist_ok=True)

        record = {
            "git_commit": git_commit,
            "timestamp": timestamp,
            **result.to_dict(),
        }
        base = f"{result.agent.replace(':', '_')}_{git_commit}"
        version = 0
        while (runs_dir / f"{base}_v{version}.json").exists():
            version += 1
        out_path = runs_dir / f"{base}_v{version}.json"
        out_path.write_text(json.dumps(record, indent=2))
        print(f"  saved → {out_path}")
