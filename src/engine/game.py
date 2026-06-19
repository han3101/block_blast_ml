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

    @property
    def combo(self) -> int:
        """Current combo streak N (multiplier applied this round is N+1).

        0 for scorers without a combo concept (e.g. SimpleScorer)."""
        return getattr(self._scorer, "combo", 0)

    @property
    def cleared_this_round(self) -> bool:
        """Whether any placement in the in-progress round has cleared a line."""
        return getattr(self._scorer, "cleared_this_round", False)

    def legal_actions(self) -> list[tuple[int, int, int]]:
        """Return all (slot, row, col) placements valid on the current grid."""
        actions: list[tuple[int, int, int]] = []
        for slot, block in enumerate(self._hand):
            if block is not None:
                for row, col in self._grid.placements(block):
                    actions.append((slot, row, col))
        return actions

    def place(self, slot: int, row: int, col: int, *, deal_next: bool = True) -> StepResult:
        """Place ``self._hand[slot]`` at (row, col) and advance the game one step.

        ``deal_next`` (search hot-path knob): when the placement exhausts the
        hand, the engine normally deals the next hand immediately. That deal runs
        an ``at_least_one`` board scan over the whole pool (the search's single
        biggest cost). A lookahead that only reads the *board* at a hand boundary
        — every leaf evaluator does, and expectimax resamples its own hand — never
        uses that freshly dealt hand, so it passes ``deal_next=False`` to skip it.
        The hand is left empty and ``game_over`` is *not* recomputed at the
        boundary (a fully-blocked board then surfaces through the leaf's
        board-health penalty rather than a bare-score terminal short-circuit).
        Mid-hand dead-ends are unaffected — that branch still sets ``game_over``.
        """
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
            if deal_next:
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

    def clone(self) -> "GameState":
        """Return a fully independent deep copy for search rollouts.

        Unlike ``snapshot()`` (which is lossy — grid + hand names + score, no
        RNG, no restore), ``clone()`` captures *everything* needed to continue
        play identically: grid cells, hand, scorer state (score, combo streak,
        in-round clear flag), the generator's RNG bit-state, and the
        game_over/last_result flags. Mutating the clone (via ``place``) never
        touches the original, so the original can keep playing the real game
        while the clone explores a hypothetical line.

        ``hand`` holds frozen ``Block`` objects and ``last_result`` is a frozen
        ``StepResult``, so both are shared by reference (immutable — safe).
        """
        new = GameState.__new__(GameState)
        new._generator = self._generator.clone()
        new._scorer = self._scorer.clone()
        new._grid = self._grid.clone()
        new._hand = list(self._hand)
        new.game_over = self.game_over
        new.last_result = self.last_result
        return new

    def restore(self, other: "GameState") -> None:
        """Overwrite this state in-place with an independent copy of ``other``.

        The inverse direction of ``clone()``: for a depth-first search that
        wants to undo a rollout without reallocating, do
        ``saved = state.clone(); ...mutate state...; state.restore(saved)``.
        ``other`` is left untouched (its grid/scorer/generator are re-copied),
        so the same ``saved`` snapshot can be restored repeatedly.
        """
        self._generator = other._generator.clone()
        self._scorer = other._scorer.clone()
        self._grid = other._grid.clone()
        self._hand = list(other._hand)
        self.game_over = other.game_over
        self.last_result = other.last_result

    def resample_hand(self, seed: int | None) -> "GameState":
        """Return a clone with its hand re-dealt for the *current board*.

        Search chance-node primitive: ``clone()`` reproduces the one next hand
        the live RNG would deal, but expectimax needs an *expectation* over the
        next-hand distribution. Each ``resample_hand`` reseeds the cloned
        generator and re-deals on the same board, so drawing many gives iid
        samples from the true marginal next-hand distribution (mode + pool +
        board filtering all preserved). ``game_over`` is recomputed for the new
        hand (random mode can deal an unplaceable hand → terminal).

        Note this redeals from the *current* hand state — call it on a state
        whose board is the one the next hand should be dealt against (e.g. a
        state that just exhausted its hand and refreshed).
        """
        new = self.clone()
        new._generator.reseed(seed)
        new._deal_hand()
        return new

    def snapshot(self) -> dict:
        return {
            "grid": self._grid.to_matrix(),
            "hand": [b.name if b is not None else None for b in self._hand],
            "score": self._scorer.score,
            "game_over": self.game_over,
        }
