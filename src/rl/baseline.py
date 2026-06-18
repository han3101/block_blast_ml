"""Run the greedy agent over N seeded games and report score statistics."""
from __future__ import annotations

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from engine.game import GameState
from engine.generator import Mode
from rl.agents.greedy import choose_action
from rl.encoding import decode_action

# uv run python -m rl.baseline
def run_greedy_baseline(
    n_games: int = 100,
    seed_offset: int = 0,
    mode: Mode = "at_least_one",
) -> dict:
    """Run the greedy agent over N seeded games and report score statistics."""
    scores = []
    for i in range(n_games):
        state = GameState(seed=seed_offset + i, mode=mode)
        while not state.game_over:
            action = choose_action(state)
            slot, row, col = decode_action(action)
            state.place(slot, row, col)
        scores.append(state.score)
    return {
        "n_games": n_games,
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "max": max(scores),
        "min": min(scores),
        "scores": scores,
    }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Run greedy baseline over N seeded games.")
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    t0 = time.perf_counter()
    results = run_greedy_baseline(n_games=args.n_games)
    elapsed = time.perf_counter() - t0

    print(f"Greedy baseline over {results['n_games']} games ({elapsed:.2f}s):")
    print(
        f"  mean={results['mean']:.1f}  median={results['median']:.1f}"
        f"  stdev={results['stdev']:.1f}  max={results['max']}  min={results['min']}"
    )

    if args.save:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "agent": "greedy",
            "git_commit": git_commit,
            "n_games": results["n_games"],
            "seed_offset": 0,
            "timestamp": timestamp,
            "elapsed_s": round(elapsed, 2),
            "stats": {
                "mean": results["mean"],
                "median": results["median"],
                "stdev": results["stdev"],
                "max": results["max"],
                "min": results["min"],
            },
            "scores": results["scores"],
        }
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        runs_dir.mkdir(exist_ok=True)
        base = f"greedy_{git_commit}"
        version = 0
        while (runs_dir / f"{base}_v{version}.json").exists():
            version += 1
        out_path = runs_dir / f"{base}_v{version}.json"
        out_path.write_text(json.dumps(record, indent=2))
        print(f"  saved → {out_path}")
