"""
Combined diversity comparison across all CelNet policies + independent baseline
+ traditional KD baseline.

Usage:
    python experiments/diversity/plot_diversity_combined.py --out results/diversity_combined/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

METHODS = {
    "MWM_AccDiff":   "results/diversity_mwm_acc_1073995",
    "MWM_ClassDist": "results/diversity_MWM_ClassDist_1073996_0",
    "AccDiff":       "results/diversity_AccDiff_1073996_1",
    "ClassDist":     "results/diversity_ClassDist_1073996_2",
    "RRG_AccDiff":   "results/diversity_RRG_AccDiff_1073996_3",
    "RRg_Random":    "results/diversity_RRg_Random_1073996_4",
    "Independent":   "results/diversity_independent_1073997",
    "Traditional KD": "results/diversity_trad_kd_1073998",
    # Experiment A: DML full-label baseline — job ID filled in after run
    # "DML":           "results/diversity_dml_<JOBID>",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#7f7f7f", "#17becf", "#e377c2"]


def load(data_dir: Path, fname: str):
    return torch.load(data_dir / fname, map_location="cpu", weights_only=True).numpy()


def plot_disagreement_combined(out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, path), color in zip(METHODS.items(), COLORS):
        dis = load(Path(path), "disagreement_per_round.pt")
        rounds = np.arange(len(dis))
        ax.plot(rounds, dis, color=color, linewidth=1.5, label=name)
    ax.set_title("Pairwise Disagreement over Rounds — All Methods", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg pairwise disagreement")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out_dir / "combined_disagreement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_diversity_combined(out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, path), color in zip(METHODS.items(), COLORS):
        div = load(Path(path), "diversity_per_round.pt")
        rounds = np.arange(len(div))
        ax.plot(rounds, div, color=color, linewidth=1.5, label=name)
    ax.set_title("Diversity Gain over Rounds — All Methods", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Diversity (avg ind. error − ensemble error)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out_dir / "combined_diversity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_final_summary_bars(out_dir: Path):
    names = list(METHODS.keys())
    final_dis, final_div, final_ens = [], [], []
    for name, path in METHODS.items():
        dis = load(Path(path), "disagreement_per_round.pt")
        div = load(Path(path), "diversity_per_round.pt")
        ens = load(Path(path), "ensemble_error_per_round.pt")
        final_dis.append(dis[-1])
        final_div.append(div[-1])
        final_ens.append(1 - ens[-1])  # final ensemble accuracy fraction

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title, ylabel in [
        (axes[0], final_dis, "Final Pairwise Disagreement", "Disagreement"),
        (axes[1], final_div, "Final Diversity Gain", "Diversity"),
        (axes[2], final_ens, "Final Ensemble Accuracy", "Accuracy"),
    ]:
        bars = ax.bar(names, vals, color=COLORS)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out_dir / "combined_final_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for combined plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_disagreement_combined(out_dir)
    plot_diversity_combined(out_dir)
    plot_final_summary_bars(out_dir)
    print("\nAll combined diversity plots saved.")


if __name__ == "__main__":
    main()
