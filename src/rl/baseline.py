"""Run the greedy agent over N seeded games and report score statistics."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from engine.game import GameState
from engine.generator import Mode
from rl.agents.greedy import choose_action
from rl.encoding import decode_action
from rl.types import BenchmarkResult

# uv run python -m rl.baseline
def run_greedy_baseline(
    n_games: int = 100,
    seed_offset: int = 0,
    mode: Mode = "at_least_one",
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
    return BenchmarkResult.from_plays("greedy", scores, lengths, seed_offset=seed_offset)


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

    print(f"Greedy baseline over {results.n_games} games ({elapsed:.2f}s):")
    print(
        f"  mean={results.mean:.1f}  median={results.median:.1f}"
        f"  stdev={results.stdev:.1f}  max={results.max}  min={results.min}"
    )
    print(
        f"  avg_length={results.avg_length:.1f}  median_length={results.median_length:.1f}"
        f"  max_length={results.max_length}"
    )

    if args.save:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "git_commit": git_commit,
            "timestamp": timestamp,
            "elapsed_s": round(elapsed, 2),
            **results.to_dict(),
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
