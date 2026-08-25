"""
Prints student validation accuracy at fixed checkpoints across the Exp 1
trajectory (KL phase 1-100, CE phase 101-160), mean +/- std across seeds
per alpha -- a text-only alternative to plot_accuracy_trajectory.py for
when you can't easily pull a PDF off the cluster.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/print_accuracy_checkpoints.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load_by_alpha(results_dir: Path):
    by_alpha: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("alpha*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_alpha.setdefault(data["alpha"], []).append(data)
    return by_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--checkpoints", type=int, nargs="+",
                     default=[1, 20, 50, 80, 100, 105, 110, 120, 140, 160])
    args = ap.parse_args()

    by_alpha = load_by_alpha(Path(args.results_dir))
    if not by_alpha:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print("Checkpoints <=100 are in the KL phase; >100 are in the CE phase "
          "(switch happens right after round 100).\n")
    header = "alpha".rjust(6) + "".join(f"{'r'+str(c):>9}" for c in args.checkpoints)
    print(header)
    print("-" * len(header))
    for alpha in sorted(by_alpha):
        runs = by_alpha[alpha]
        row = f"{alpha:>6}"
        for c in args.checkpoints:
            vals = [r["val_acc"][c - 1] for r in runs if c - 1 < len(r["val_acc"])]
            row += f"{stats.mean(vals):>9.2f}" if vals else f"{'--':>9}"
        print(row)
    print(f"\n(mean across seeds; n per alpha: " +
          ", ".join(f"{a}={len(by_alpha[a])}" for a in sorted(by_alpha)) + ")")


if __name__ == "__main__":
    main()
