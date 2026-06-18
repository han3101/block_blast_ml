"""Terminal runner: watch an agent play Block Blast step by step.

Usage:
    uv run python -m app.cli_game                          # greedy, random seed
    uv run python -m app.cli_game --seed 7                 # deterministic replay
    uv run python -m app.cli_game --mode random            # pure random hand generation
    uv run python -m app.cli_game --mode solvable          # all 3 pieces always placeable
    uv run python -m app.cli_game --delay 0                # as fast as possible
    uv run python -m app.cli_game --delay 1                # 1 second delay
    uv run python -m app.cli_game --step                   # press Enter between moves
"""
from __future__ import annotations

import argparse
import os
import time

from engine.block import Block
from engine.game import GameState
from engine.generator import Mode
from rl.agents.greedy import choose_action
from rl.encoding import decode_action

# Agent registry — add entries here as new agents are trained/implemented.
AGENTS = {
    "greedy": choose_action,
}


# --- rendering helpers ---

def _render_block(block: Block | None, height: int = 6) -> list[str]:
    if block is None:
        return ["(empty)"] + [""] * (height - 1)
    rows = block.height
    cols = block.width
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for r, c in block.cells:
        grid[r][c] = "#"
    lines = [" ".join(row) for row in grid]
    while len(lines) < height:
        lines.append("")
    return lines


def _render_board(matrix: list[list[int]]) -> list[str]:
    return [" ".join("#" if c else "." for c in row) for row in matrix]


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _render_frame(state: GameState, step: int, seed: int, last_info: str, agent: str = "greedy", mode: str = "at_least_one") -> str:
    snap = state.snapshot()
    board_lines = _render_board(snap["grid"])

    hand_height = 6
    col_width = 18
    hand_cols: list[list[str]] = []
    for i, block in enumerate(state.hand):
        label = f"[{i}] {block.name if block else '—':12}"
        block_lines = _render_block(block, height=hand_height)
        hand_cols.append([label] + block_lines[:hand_height])

    hand_rows: list[str] = []
    for row_i in range(hand_height + 1):
        row = "  ".join(
            (col[row_i] if row_i < len(col) else "").ljust(col_width)
            for col in hand_cols
        )
        hand_rows.append(row)

    lines: list[str] = []
    lines.append(f"Block Blast — {agent} / {mode}   seed={seed}   step={step}")
    lines.append("─" * 60)

    n = max(len(board_lines), len(hand_rows))
    for i in range(n):
        bl = board_lines[i] if i < len(board_lines) else " " * 15
        hr = hand_rows[i] if i < len(hand_rows) else ""
        lines.append(f"{bl}    {'Hand:' if i == 0 else hr}")

    lines.append("─" * 60)
    lines.append(f"Score: {snap['score']}   {last_info}")
    if snap["game_over"]:
        lines.append("\n  *** GAME OVER ***")
    return "\n".join(lines)


def run(seed: int, delay: float, step_mode: bool, mode: Mode, agent: str) -> None:
    state = GameState(seed=seed, mode=mode)
    choose = AGENTS[agent]
    step = 0
    last_info = "—"

    _clear()
    print(_render_frame(state, step, seed, last_info, agent=agent, mode=mode))

    while not state.game_over:
        if step_mode:
            input("\nEnter to step...")
        else:
            time.sleep(delay)

        action = choose(state)
        slot, row, col = decode_action(action)
        placed = state.hand[slot]
        block_name = placed.name if placed else "?"
        result = state.place(slot, row, col)

        step += 1
        score_delta = result.score - result.prev_score
        last_info = (
            f"placed {block_name} @ ({row},{col})  "
            f"+{score_delta} pts  "
            f"{result.lines_cleared} line(s) cleared"
        )

        _clear()
        print(_render_frame(state, step, seed, last_info, agent=agent, mode=mode))

    print(f"\nFinal score: {state.score}  steps: {step}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the greedy agent play Block Blast.")
    parser.add_argument("--seed", type=int, default=None, help="fix the seed for a deterministic replay")
    parser.add_argument("--mode", choices=["at_least_one", "random", "solvable"], default="at_least_one")
    parser.add_argument("--agent", choices=list(AGENTS), default="greedy")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="seconds between moves (ignored with --step)")
    parser.add_argument("--step", action="store_true",
                        help="press Enter between each move instead of auto-advancing")
    args = parser.parse_args()
    import random as _random
    seed = args.seed if args.seed is not None else _random.randint(0, 2**31)
    run(seed=seed, delay=args.delay, step_mode=args.step, mode=args.mode, agent=args.agent)


if __name__ == "__main__":
    main()
