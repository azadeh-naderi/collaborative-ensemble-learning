"""
Analysis for the repeated short-cycle experiment (run_repeated_cycle.py).

Reports, per K (mean +/- std across seeds):
  - final Oracle-gain-style sum and peer-gain-style sum
  - whether/when the Oracle-gain-style cumulative sum sustains a negative
    crossing (analog of CEL-Net's Fig 1 zero-crossing), as an exposure index
  - a checkpoint table of the Oracle-gain-style cumulative sum at fixed
    exposure indices, to see the trend shape (matching CEL-Net's Fig 1:
    does it rise-then-fall for large K, stay rising for small K?)
  - cosine at the first vs. last CE round, to see whether repeated cycling
    drifts the angle toward/past zero over time

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/analyze_repeated_cycle.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load_by_k(results_dir: Path):
    by_k: dict[int, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("k*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_k.setdefault(data["k"], []).append(data)
    return by_k


def ce_round_cosines(data):
    """cos values logged specifically at CE rounds, in chronological order."""
    return [e["cos"] for e in data["ce_kl_cosine"] if e.get("is_ce_round")]


def oracle_gain_at_exposure(data, exposure_idx):
    """oracle_gain_cumulative value right after the exposure_idx-th CE round
    (1-indexed). Returns None if the run has fewer exposures than that."""
    ce_positions = [i for i, p in enumerate(data["phase"]) if p == "ce"]
    if exposure_idx > len(ce_positions):
        return None
    return data["oracle_gain_cumulative"][ce_positions[exposure_idx - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--checkpoint_exposures", type=int, nargs="+",
                     default=[5, 10, 15, 20])
    args = ap.parse_args()

    by_k = load_by_k(Path(args.results_dir))
    if not by_k:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print("Final Oracle-gain-style and peer-gain-style sums, per K (mean +/- std):\n")
    header = (f"{'K':>4} | {'seeds':>5} | {'total_rounds':>12} | "
              f"{'final Oracle-gain':>20} | {'final peer-gain':>18} | {'sustained crossing (exp #)':>26}")
    print(header)
    print("-" * len(header))
    for k in sorted(by_k):
        runs = by_k[k]
        oracle_finals = [r["final_oracle_gain"] for r in runs]
        peer_finals = [r["final_peer_gain"] for r in runs]
        crossings = [r["oracle_gain_sustained_crossing_exposure"] for r in runs]
        n_crossed = sum(1 for c in crossings if c is not None)
        crossing_str = (f"{stats.mean([c for c in crossings if c is not None]):.1f} "
                         f"({n_crossed}/{len(runs)} seeds)") if n_crossed else f"never (0/{len(runs)})"
        oracle_str = f"{stats.mean(oracle_finals):+.2f} +/- {stats.pstdev(oracle_finals) if len(oracle_finals) > 1 else 0:.2f}"
        peer_str = f"{stats.mean(peer_finals):+.2f} +/- {stats.pstdev(peer_finals) if len(peer_finals) > 1 else 0:.2f}"
        total_rounds = runs[0]["total_rounds"]
        print(f"{k:>4} | {len(runs):>5} | {total_rounds:>12} | {oracle_str:>20} | {peer_str:>18} | {crossing_str:>26}")

    print(f"\nOracle-gain-style cumulative sum at fixed exposure checkpoints (mean across seeds):")
    header2 = "K".rjust(4) + "".join(f"{'exp'+str(e):>10}" for e in args.checkpoint_exposures)
    print(header2)
    for k in sorted(by_k):
        runs = by_k[k]
        row = f"{k:>4}"
        for e in args.checkpoint_exposures:
            vals = [oracle_gain_at_exposure(r, e) for r in runs]
            vals = [v for v in vals if v is not None]
            row += f"{stats.mean(vals):>10.2f}" if vals else f"{'--':>10}"
        print(row)

    print(f"\ncos(g_CE, g_KL) at CE rounds: first exposure vs. last exposure (mean across seeds):")
    header3 = f"{'K':>4} | {'first CE cos':>14} | {'last CE cos':>13}"
    print(header3)
    print("-" * len(header3))
    for k in sorted(by_k):
        runs = by_k[k]
        firsts, lasts = [], []
        for r in runs:
            cs = ce_round_cosines(r)
            if cs:
                firsts.append(cs[0]); lasts.append(cs[-1])
        f_str = f"{stats.mean(firsts):.3f}" if firsts else "--"
        l_str = f"{stats.mean(lasts):.3f}" if lasts else "--"
        print(f"{k:>4} | {f_str:>14} | {l_str:>13}")


if __name__ == "__main__":
    main()
