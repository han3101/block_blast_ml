"""One-step lookahead greedy agent for Block Blast. Not learned."""
from __future__ import annotations

from engine.game import GameState
from engine.grid import Grid
from rl.encoding import encode_action


def _count_holes(matrix: list[list[int]]) -> int:
    """Empty cells with a filled cell above them in the same column."""
    size = len(matrix)
    holes = 0
    for col in range(size):
        found_filled = False
        for row in range(size):
            if matrix[row][col] == 1:
                found_filled = True
            elif found_filled:
                holes += 1
    return holes


def choose_action(state: GameState) -> int:
    """Return the encoded action that maximises a one-step lookahead heuristic.

    Ranking key (lexicographic, higher is better):
      1. lines cleared — more clears = better
      2. occupied cells after placement — fewer = better
      3. holes (empty cells with filled above) — fewer = better
    """
    best_action: int | None = None
    best_key: tuple | None = None

    for slot, row, col in state.legal_actions():
        block = state.hand[slot]
        assert block is not None
        grid_copy = Grid(cells=state.grid.to_matrix())
        grid_copy.place(block, row, col)
        lines = grid_copy.clear_full_lines()
        matrix = grid_copy.to_matrix()
        occupied = sum(matrix[r][c] for r in range(grid_copy.size) for c in range(grid_copy.size))
        holes = _count_holes(matrix)
        key = (lines, -occupied, -holes)
        if best_key is None or key > best_key:
            best_key = key
            best_action = encode_action(slot, row, col)

    if best_action is None:
        raise ValueError("no legal actions — check state.game_over before calling choose_action")
    return best_action
