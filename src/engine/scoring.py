from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    @property
    def score(self) -> int: ...
    def score_placement(self, block_placed: int, lines_cleared: int) -> int: ...
    def reset(self) -> None: ...


class SimpleScorer:
    def __init__(self, cell_points: int = 1, line_points: int = 10) -> None:
        self._cell_points = cell_points
        self._line_points = line_points
        self._score = 0

    @property
    def score(self) -> int:
        return self._score

    def score_placement(self, block_placed: int, lines_cleared: int) -> int:
        self._score += block_placed * self._cell_points + lines_cleared * self._line_points
        return self._score

    def reset(self) -> None:
        self._score = 0

# TODO: add combo bonuses, etc.
