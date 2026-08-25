"""
Plots student validation accuracy across the full 160-round Exp 1 trajectory
(100 KL-phase rounds + 60 CE-phase rounds), one line per alpha, mean +/- std
across seeds, with a vertical line marking the KL -> CE switch at round 100.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_accuracy_trajectory.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --out "Geometry of Mixed-Loss/results/alpha_sweep/analysis/accuracy_trajectory.pdf"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--switch_round", type=int, default=100,
                     help="round at which KL phase ends and CE phase begins")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_alpha = load_by_alpha(results_dir)
    if not by_alpha:
        raise SystemExit(f"No results.json files found under {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis([i / max(1, len(by_alpha) - 1) for i in range(len(by_alpha))])

    for color, alpha in zip(colors, sorted(by_alpha)):
        runs = by_alpha[alpha]
        max_len = max(len(r["val_acc"]) for r in runs)
        means, stds, xs = [], [], []
        for i in range(max_len):
            vals = [r["val_acc"][i] for r in runs if i < len(r["val_acc"])]
            if not vals:
                continue
            xs.append(i + 1)
            means.append(stats.mean(vals))
            stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)
        ax.plot(xs, means, color=color, lw=1.8, label=f"α={alpha}")
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)

    ax.axvline(args.switch_round, color="black", ls="--", lw=1.0)
    ax.text(args.switch_round + 1, ax.get_ylim()[0], "KL → CE switch",
            rotation=90, va="bottom", fontsize=8, color="#555")

    ax.set_xlabel("Round")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Exp 1: student accuracy across KL phase (1–100) and CE phase (101–160)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
