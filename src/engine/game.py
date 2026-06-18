from __future__ import annotations

from dataclasses import dataclass

from engine.block import ALL_BLOCKS, Block
from engine.generator import HandGenerator, Mode
from engine.grid import Grid
from engine.scoring import ComboScorer, Scorer, SimpleScorer


@dataclass(frozen=True, slots=True)
class StepResult:
    score: int
    lines_cleared: int
    cells_placed: int
    game_over: bool
    hand_refreshed: bool
    prev_score: int = 0
    combo: int = 0


class GameState:
    def __init__(
        self,
        seed: int | None = None,
        pool: tuple[Block, ...] = ALL_BLOCKS,
        mode: Mode = "at_least_one",
        scorer: Scorer | None = None,
    ) -> None:
        self._generator = HandGenerator(seed=seed, pool=pool, mode=mode)
        self._scorer: Scorer = scorer if scorer is not None else ComboScorer()
        self._grid = Grid()
        self._hand: list[Block | None] = []
        self.game_over: bool = False
        self.last_result: StepResult | None = None
        self._deal_hand()

    def _deal_hand(self) -> None:
        dealt = self._generator.deal(self._grid)
        if dealt is None:
            self._hand = [None, None, None]
            self.game_over = True
        else:
            self._hand = list(dealt)
            # at_least_one/solvable guarantee a playable piece, so this only
            # fires in random mode where the generator ignores board state.
            self.game_over = not bool(self.legal_actions())

    @property
    def hand(self) -> tuple[Block | None, ...]:
        return tuple(self._hand)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def score(self) -> int:
        return self._scorer.score

    def legal_actions(self) -> list[tuple[int, int, int]]:
        """Return all (slot, row, col) placements valid on the current grid."""
        actions: list[tuple[int, int, int]] = []
        for slot, block in enumerate(self._hand):
            if block is not None:
                for row, col in self._grid.placements(block):
                    actions.append((slot, row, col))
        return actions

    def place(self, slot: int, row: int, col: int) -> StepResult:
        if self.game_over:
            raise ValueError("game is over — call reset() to start a new game")
        if slot not in range(3):
            raise ValueError(f"slot must be 0, 1, or 2; got {slot!r}")
        block = self._hand[slot]
        if block is None:
            raise ValueError(f"slot {slot} is already empty")
        if not self._grid.can_place(block, row, col):
            raise ValueError(f"cannot place {block.name!r} at ({row}, {col})")
        if self._scorer.score is not None:
            prev_score = self._scorer.score
        else:
            prev_score = 0

        self._grid.place(block, row, col)
        lines_cleared = self._grid.clear_full_lines()
        cells_placed = len(block.cells)
        self._scorer.score_placement(cells_placed, lines_cleared)
        combo = getattr(self._scorer, "combo", 0)
        self._hand[slot] = None

        hand_refreshed = False
        if all(b is None for b in self._hand):
            self._scorer.end_round()
            self._deal_hand()
            hand_refreshed = True
        else:
            self.game_over = not bool(self.legal_actions())

        self.last_result = StepResult(
            score=self._scorer.score,
            lines_cleared=lines_cleared,
            cells_placed=cells_placed,
            game_over=self.game_over,
            hand_refreshed=hand_refreshed,
            prev_score=prev_score,
            combo=combo,
        )
        return self.last_result

    def reset(self) -> None:
        self._grid = Grid()
        self._scorer.reset()
        self.game_over = False
        self.last_result = None
        self._deal_hand()

    def snapshot(self) -> dict:
        return {
            "grid": self._grid.to_matrix(),
            "hand": [b.name if b is not None else None for b in self._hand],
            "score": self._scorer.score,
            "game_over": self.game_over,
        }
