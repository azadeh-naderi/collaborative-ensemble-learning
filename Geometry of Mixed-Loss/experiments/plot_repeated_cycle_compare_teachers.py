"""
Side-by-side comparison of cumulative Oracle-gain-style (CE updates,
dashed) vs. cumulative peer-gain-style (KL updates, solid) accuracy gain
for the repeated short-cycle experiment, one subplot per teacher
condition (CE-pretrained vs. KL-self-distilled), sharing a y-axis so the
two are directly comparable -- one line pair per K, mean across seeds.

Line style always encodes update type (solid = KL, dashed = CE); color
always encodes K. This is made explicit via two separate legends (one for
line style, one for color) plus a direct in-line label at the end of each
curve, so the KL-vs-CE distinction doesn't rely on reading a crowded
per-line legend.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_repeated_cycle_compare_teachers.py" \
        --ce_teacher_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --kl_teacher_dir "Geometry of Mixed-Loss/results/repeated_cycle_kl_teacher" \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/analysis/teacher_comparison.pdf"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def plot_one(ax, results_dir: Path, title: str):
    by_k = load_by_k(results_dir)
    if not by_k:
        raise SystemExit(f"No results.json files found under {results_dir}")

    ks = sorted(by_k)
    colors = plt.cm.plasma([i / max(1, len(ks) - 1) for i in range(len(ks))])
    for color, k in zip(colors, ks):
        runs = by_k[k]
        oracle_mean = mean_series(runs, "oracle_gain_cumulative")
        peer_mean = mean_series(runs, "peer_gain_cumulative")
        xs_o = list(range(1, len(oracle_mean) + 1))
        xs_p = list(range(1, len(peer_mean) + 1))
        ax.plot(xs_o, oracle_mean, color=color, lw=2.2, ls=(0, (4, 2)))
        ax.plot(xs_p, peer_mean, color=color, lw=2.6)
        # direct in-line labels at the end of each curve
        ax.annotate(f"K={k} KL", xy=(xs_p[-1], peer_mean[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=8, fontweight="bold", va="center")
        ax.annotate(f"K={k} CE", xy=(xs_o[-1], oracle_mean[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=8, va="center")

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Round")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    ax.margins(x=0.12)  # leave room for the in-line end labels

    color_legend = ax.legend(
        handles=[Line2D([0], [0], color=c, lw=2.6, label=f"K={k}")
                 for c, k in zip(colors, ks)],
        loc="upper left", fontsize=8, frameon=True, title="Color = K",
        title_fontsize=8,
    )
    ax.add_artist(color_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="gray", lw=2.6, ls="-",
                   label="solid = KL updates"),
            Line2D([0], [0], color="gray", lw=2.2, ls=(0, (4, 2)),
                   label="dashed = CE updates"),
        ],
        loc="center left", fontsize=8, frameon=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_teacher_dir", required=True)
    ap.add_argument("--kl_teacher_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    plot_one(ax1, Path(args.ce_teacher_dir), "CE-pretrained teacher")
    plot_one(ax2, Path(args.kl_teacher_dir), "KL-self-distilled teacher")
    ax1.set_ylabel("Cumulative Accuracy Gain")
    fig.suptitle("Repeated short-cycle: cumulative KL-update vs. CE-update gain, both teacher conditions")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
