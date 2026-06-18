"""Shared result types for all agents and eval scripts."""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of running any agent over N seeded games.

    Scores and lengths are tuples (immutable) so the dataclass stays hashable.
    Use to_dict() for JSON serialization.
    """

    agent: str
    n_games: int
    seed_offset: int
    mean: float
    median: float
    stdev: float
    max: int
    min: int
    avg_length: float
    median_length: float
    max_length: int
    scores: tuple[int, ...]
    lengths: tuple[int, ...]

    @classmethod
    def from_plays(
        cls,
        agent: str,
        scores: list[int],
        lengths: list[int],
        seed_offset: int = 0,
    ) -> "BenchmarkResult":
        n = len(scores)
        return cls(
            agent=agent,
            n_games=n,
            seed_offset=seed_offset,
            mean=statistics.mean(scores),
            median=statistics.median(scores),
            stdev=statistics.stdev(scores) if n > 1 else 0.0,
            max=max(scores),
            min=min(scores),
            avg_length=statistics.mean(lengths),
            median_length=statistics.median(lengths),
            max_length=max(lengths),
            scores=tuple(scores),
            lengths=tuple(lengths),
        )

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "n_games": self.n_games,
            "seed_offset": self.seed_offset,
            "stats": {
                "mean": self.mean,
                "median": self.median,
                "stdev": self.stdev,
                "max": self.max,
                "min": self.min,
                "avg_length": self.avg_length,
                "median_length": self.median_length,
                "max_length": self.max_length,
            },
            "scores": list(self.scores),
            "lengths": list(self.lengths),
        }
