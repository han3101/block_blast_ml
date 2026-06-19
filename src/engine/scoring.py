from __future__ import annotations

from typing import Protocol, runtime_checkable

# Base line-clear score indexed by number of simultaneously cleared lines (0–8).
# Formula: 10 for L=1, 10·L·(L−1) for L≥2. Sized to the 8×8 board max.
BASE_LINE_SCORES = [0, 10, 20, 60, 120, 200, 300, 420, 560]


@runtime_checkable
class Scorer(Protocol):
    @property
    def score(self) -> int: ...
    def score_placement(self, block_placed: int, lines_cleared: int) -> int: ...
    def end_round(self) -> None: ...
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

    def end_round(self) -> None:
        pass

    def reset(self) -> None:
        self._score = 0


class ComboScorer:
    """Real Block Blast scorer with round-level combo streak.

    Per placement: move_score = cells_placed + BASE_LINE_SCORES[lines_cleared] * (N + 1)
    Per round: N += 1 if any placement in the round cleared lines, else N = 0.
    All clears within a round share the same N+1 multiplier. Streak is uncapped.
    """

    def __init__(self) -> None:
        self._score = 0
        self._N = 0                     # combo count: streak of consecutive clearing rounds
        self._cleared_this_round = False

    @property
    def score(self) -> int:
        return self._score

    @property
    def combo(self) -> int:
        """Current streak length N (multiplier applied this round is N+1)."""
        return self._N

    @property
    def cleared_this_round(self) -> bool:
        """Whether any placement in the current (in-progress) round cleared a line."""
        return self._cleared_this_round

    def score_placement(self, block_placed: int, lines_cleared: int) -> int:
        self._score += block_placed
        if lines_cleared > 0:
            self._score += BASE_LINE_SCORES[lines_cleared] * (self._N + 1)
            self._cleared_this_round = True
        return self._score

    def end_round(self) -> None:
        """Call once per round after the 3rd placement is scored."""
        if self._cleared_this_round:
            self._N += 1
        else:
            self._N = 0
        self._cleared_this_round = False

    def reset(self) -> None:
        self._score = 0
        self._N = 0
        self._cleared_this_round = False
