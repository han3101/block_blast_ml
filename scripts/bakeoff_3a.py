"""Phase 3a leaf-evaluator bake-off (the value-head diagnostic).

Runs greedy + the Phase 2 policy + the lookahead search under each swappable
leaf evaluator on a *common* seed set, and reports score / placements / latency.
The question it answers (plans/phase-3.md, "Prerequisite"): is the Phase 2
critic (explained_variance ~0.16) good enough to be a leaf evaluator, or does
greedy's board-health heuristic beat it? That result gates the BC-on-greedy
contingency and tells 3b whether it can lean on the existing value head.

Two groupings:
  * Main bake-off  — all leaves at horizon 1 on seeds 0..N-1 (apples-to-apples).
  * Horizon probe  — greedy_health vs value_net at h1 vs h2 on the first M seeds,
    to sketch the depth/strength/latency curve for the definition of done.

Usage:
    uv run python scripts/bakeoff_3a.py [--checkpoint PATH] [--games 15] [--h2-games 5]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl.baseline import run_greedy_baseline, run_policy_baseline, run_search_baseline  # noqa: E402
from rl.types import BenchmarkResult  # noqa: E402

DEFAULT_CKPT = "runs/phase2_v3_curriculum/best_model.pt"


def _timed(fn, *args, **kwargs) -> tuple[BenchmarkResult, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _row(label: str, r: BenchmarkResult, elapsed: float) -> str:
    per_move = 1000.0 * elapsed / max(1, sum(r.lengths))
    return (
        f"{label:<26} {r.mean:>10.1f} {r.median:>9.1f} {r.max:>8} "
        f"{r.avg_length:>8.1f} {r.max_length:>6} {elapsed:>8.1f} {per_move:>9.1f}"
    )


HEADER = (
    f"{'agent':<26} {'mean':>10} {'median':>9} {'max':>8} "
    f"{'avg_len':>8} {'maxlen':>6} {'secs':>8} {'ms/move':>9}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--games", type=int, default=15, help="games for the h1 main bake-off")
    ap.add_argument("--h2-games", type=int, default=5, help="games for the slower h2 horizon probe")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = args.checkpoint
    N, M = args.games, args.h2_games
    results: dict[str, tuple[BenchmarkResult, float]] = {}

    def run(label, fn, *a, **k):
        print(f"  … {label}", flush=True)
        results[label] = _timed(fn, *a, **k)

    print(f"=== Main bake-off — horizon 1, seeds 0–{N - 1} ===", flush=True)
    run("greedy", run_greedy_baseline, N)
    run("policy:v3", run_policy_baseline, ckpt, N, device=args.device)
    run("search:score_only:h1", run_search_baseline, "score_only", None, N, horizon_hands=1)
    run("search:greedy_health:h1", run_search_baseline, "greedy_health", None, N, horizon_hands=1)
    run("search:value_net:h1", run_search_baseline, "value_net", ckpt, N,
        device=args.device, horizon_hands=1)

    print(f"\n=== Horizon probe — seeds 0–{M - 1} ===", flush=True)
    run("search:greedy_health:h2", run_search_baseline, "greedy_health", None, M,
        horizon_hands=2, beam=8, samples=4)
    run("search:value_net:h2", run_search_baseline, "value_net", ckpt, M,
        device=args.device, horizon_hands=2, beam=6, samples=3)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * len(HEADER))
    print("MAIN BAKE-OFF (horizon 1, all agents on the same seed set)")
    print(HEADER)
    for label in ("greedy", "policy:v3", "search:score_only:h1",
                  "search:greedy_health:h1", "search:value_net:h1"):
        print(_row(label, *results[label]))

    print("\nHORIZON PROBE (h1 sliced to first M seeds vs fresh h2 runs)")
    print(HEADER)
    for leaf in ("greedy_health", "value_net"):
        h1, h1_secs = results[f"search:{leaf}:h1"]
        sub = list(h1.scores[:M])
        sub_len = list(h1.lengths[:M])
        sliced = BenchmarkResult.from_plays(f"search:{leaf}:h1[:{M}]", sub, sub_len)
        # latency for the slice is unknown precisely; report full-run ms/move instead.
        print(_row(f"search:{leaf}:h1 (first {M})", sliced,
                   h1_secs * sum(sub_len) / max(1, sum(h1.lengths))))
        print(_row(f"search:{leaf}:h2", *results[f"search:{leaf}:h2"]))

    # ---- persist ----------------------------------------------------------
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    out = {
        "git_commit": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt,
        "games": N,
        "h2_games": M,
        "results": {label: {**r.to_dict(), "elapsed_s": secs}
                    for label, (r, secs) in results.items()},
    }
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    path = runs_dir / f"bakeoff_3a_{sha}.json"
    v = 0
    while path.exists():
        v += 1
        path = runs_dir / f"bakeoff_3a_{sha}_v{v}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {path}")


if __name__ == "__main__":
    main()
