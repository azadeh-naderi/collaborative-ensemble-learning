"""
Plots cumulative Oracle-gain-style (CE updates, dashed) vs. cumulative
peer-gain-style (KL updates, solid) accuracy gain over the course of the
repeated short-cycle experiment -- the direct analog of Figure 1 in the
paper (CEL-Net's own Oracle gain / peer gain decomposition), one line pair
per K, mean across seeds.

oracle_gain_cumulative and peer_gain_cumulative are logged every round in
results.json (each holds its value flat between updates of its own kind,
exactly like CEL-Net's own cumulative curves), so this just averages those
two series across seeds, per K, and plots them directly.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_repeated_cycle.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/analysis/oracle_vs_peer_gain.pdf"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_by_k(results_dir: Path):
    by_k: dict[int, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("k*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_k.setdefault(data["k"], []).append(data)
    return by_k


def mean_series(runs, key):
    max_len = max(len(r[key]) for r in runs)
    means = []
    for i in range(max_len):
        vals = [r[key][i] for r in runs if i < len(r[key])]
        means.append(stats.mean(vals))
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_k = load_by_k(results_dir)
    if not by_k:
        raise SystemExit(f"No results.json files found under {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma([i / max(1, len(by_k) - 1) for i in range(len(by_k))])

    for color, k in zip(colors, sorted(by_k)):
        runs = by_k[k]
        oracle_mean = mean_series(runs, "oracle_gain_cumulative")
        peer_mean = mean_series(runs, "peer_gain_cumulative")
        xs_o = list(range(1, len(oracle_mean) + 1))
        xs_p = list(range(1, len(peer_mean) + 1))
        ax.plot(xs_o, oracle_mean, color=color, lw=1.8, ls=(0, (5, 3)),
                label=f"K={k} Oracle-gain-style (CE)")
        ax.plot(xs_p, peer_mean, color=color, lw=1.8,
                label=f"K={k} peer-gain-style (KL)")

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Accuracy Gain")
    ax.set_title("Repeated short-cycle: cumulative CE-update vs. KL-update gain")
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
