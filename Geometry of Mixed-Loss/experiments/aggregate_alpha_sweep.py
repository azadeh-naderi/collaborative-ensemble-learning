"""
Aggregates Exp 1 (alpha-sweep) results across seeds and prints a Table-3-style
summary: T* and initial Delta_CE, mean +/- std across 5 seeds, per alpha.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/aggregate_alpha_sweep.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load_results(results_dir: Path):
    by_alpha: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("alpha*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            print(f"  [skip] {run_dir.name}: no results.json")
            continue
        data = json.loads(f.read_text())
        by_alpha.setdefault(data["alpha"], []).append(data)
    return by_alpha


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--- (no drop in any seed)"
    if len(vals) == 1:
        return f"{vals[0]:.1f} (n=1)"
    return f"{stats.mean(vals):.1f} +/- {stats.pstdev(vals):.1f} (n={len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    by_alpha = load_results(Path(args.results_dir))

    print(f"{'alpha':>6} | {'seeds':>5} | {'T* (rounds into CE phase)':>28} | {'initial Delta_CE (%)':>22}")
    print("-" * 72)
    for alpha in sorted(by_alpha):
        runs = by_alpha[alpha]
        t_stars = [r["t_star"] for r in runs]
        deltas = [r["initial_ce_delta"] for r in runs]
        n_incompatible = sum(1 for t in t_stars if t is not None)
        print(f"{alpha:>6} | {len(runs):>5} | "
              f"{summarize(t_stars):>28} | {stats.mean(deltas):+.2f} avg "
              f"({n_incompatible}/{len(runs)} seeds showed a drop)")

    print("\nPer-seed detail:")
    for alpha in sorted(by_alpha):
        for r in by_alpha[alpha]:
            print(f"  alpha={alpha} seed={r['seed']:>5}  T*={str(r['t_star']):>4}  "
                  f"initial_delta={r['initial_ce_delta']:+.2f}%")


if __name__ == "__main__":
    main()
