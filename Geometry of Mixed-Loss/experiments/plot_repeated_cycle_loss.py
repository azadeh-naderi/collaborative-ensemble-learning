"""
Plots the CE-round loss change (After - Before), train and val, across the
20 CE exposures of the repeated short-cycle experiment -- the single-model
analog of the CEL-Net measurement in
experiments/Oracle_Gain_Experiments/D. Measure loss/, which found the
Oracle CE update frequently *raises* CE loss rather than lowering it, not
just accuracy. Requires results.json produced by the updated
run_repeated_cycle.py (fields ce_train_loss_before/after,
ce_val_loss_before/after) -- older results.json files from before this
logging was added will not have these fields.

One color per K, mean +/- std across seeds, two panels (train delta, val
delta), one results_dir (one teacher condition) per invocation.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_repeated_cycle_loss.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/analysis/ce_loss_delta.pdf"
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
        if "ce_train_loss_before" not in data:
            raise SystemExit(
                f"{f} has no ce_train_loss_before field -- it was produced "
                "before loss logging was added to run_repeated_cycle.py. "
                "Re-run the experiment to regenerate results.json with loss data."
            )
        by_k.setdefault(data["k"], []).append(data)
    return by_k


def deltas(data, before_key, after_key):
    return [a - b for a, b in zip(data[after_key], data[before_key])]


def plot_panel(ax, by_k, before_key, after_key, title):
    ks = sorted(by_k)
    colors = plt.cm.plasma([i / max(1, len(ks) - 1) for i in range(len(ks))])
    for color, k in zip(colors, ks):
        runs = by_k[k]
        per_seed = [deltas(r, before_key, after_key) for r in runs]
        n_exp = max(len(d) for d in per_seed)
        means, stds = [], []
        for i in range(n_exp):
            vals = [d[i] for d in per_seed if i < len(d)]
            means.append(stats.mean(vals))
            stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)
        xs = list(range(1, n_exp + 1))
        ax.plot(xs, means, color=color, lw=1.9, label=f"K={k}")
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)], color=color, alpha=0.12)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("CE exposure #")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plot_panel(ax1, by_k, "ce_train_loss_before", "ce_train_loss_after",
               "Train CE loss change (After - Before)")
    plot_panel(ax2, by_k, "ce_val_loss_before", "ce_val_loss_after",
               "Val CE loss change (After - Before)")
    ax1.set_ylabel("Loss Change")
    fig.suptitle("Repeated short-cycle: CE-round loss change per exposure, by K")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
