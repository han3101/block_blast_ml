"""Run the greedy agent over N seeded games and report score statistics."""
from __future__ import annotations

import statistics

from engine.game import GameState
from engine.generator import Mode
from rl.agents.greedy import choose_action
from rl.encoding import decode_action


def run_greedy_baseline(
    n_games: int = 100,
    seed_offset: int = 0,
    mode: Mode = "at_least_one",
) -> dict:
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
        "max": max(scores),
        "min": min(scores),
    }


if __name__ == "__main__":
    results = run_greedy_baseline(n_games=100)
    print(f"Greedy baseline over {results['n_games']} games:")
    print(
        f"  mean={results['mean']:.1f}  median={results['median']:.1f}"
        f"  max={results['max']}  min={results['min']}"
    )
