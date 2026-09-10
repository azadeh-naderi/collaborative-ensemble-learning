"""
Plots cumulative CE-phase Delta_acc (mean +/- std across seeds), one line
per dominance-variable group, side by side for both teacher conditions.
Works from the CSVs produced by export_full_delta_acc_csv.py.

Uses a fixed, high-contrast color list rather than a continuous colormap
-- plasma/viridis's brightest end (pale yellow) is nearly invisible on a
white background, which matters here since the highest-dominance group
(often the most important line) tends to land on it.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_cumulative_ce_both_teachers.py" \
        --csv_a "Geometry of Mixed-Loss/results/alpha_sweep/analysis/full_delta_acc.csv" \
        --label_a "CE-pretrained teacher" \
        --csv_b "Geometry of Mixed-Loss/results/alpha_sweep_kl_teacher/analysis/full_delta_acc.csv" \
        --label_b "KL-self-distilled teacher" \
        --group_prefix "alpha=" --xlabel "CE round" \
        --title "Alpha-sweep: cumulative CE-phase accuracy gain (mean +/- std across seeds)" \
        --out "Geometry of Mixed-Loss/results/alpha_sweep/analysis/cumulative_ce_both_teachers.pdf"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# High-contrast, colorblind-friendly, all clearly visible on white
DISTINCT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
]


def cum_by_group(path):
    df = pd.read_csv(path)
    df["group"] = df["group"].astype(float)
    ce = df[df.phase == "ce"].sort_values(["group", "seed", "round_index"]).copy()
    ce["exposure_idx"] = ce.groupby(["group", "seed"]).cumcount() + 1
    ce["cum"] = ce.groupby(["group", "seed"])["delta_acc"].cumsum()
    return ce


def plot_panel(ax, csv_path, group_prefix, title):
    ce = cum_by_group(csv_path)
    groups = sorted(ce["group"].unique())
    for color, g in zip(DISTINCT_COLORS, groups):
        sub = ce[ce["group"] == g]
        agg = sub.groupby("exposure_idx")["cum"].agg(["mean", "std"])
        label = f"{group_prefix}{g:g}"
        ax.plot(agg.index, agg["mean"], color=color, lw=2.0, label=label)
        ax.fill_between(agg.index, agg["mean"] - agg["std"].fillna(0),
                         agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True)
    ap.add_argument("--label_a", required=True)
    ap.add_argument("--csv_b", required=True)
    ap.add_argument("--label_b", required=True)
    ap.add_argument("--group_prefix", default="")
    ap.add_argument("--xlabel", default="round")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_panel(axes[0], args.csv_a, args.group_prefix, args.label_a)
    plot_panel(axes[1], args.csv_b, args.group_prefix, args.label_b)
    axes[0].set_ylabel("Cumulative CE-phase Delta_acc")
    for ax in axes:
        ax.set_xlabel(args.xlabel)
    fig.suptitle(args.title)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
