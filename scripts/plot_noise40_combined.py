"""
Combined plot for noise40 500-round experiments.

Usage (run from repo root on Wulver):
    # 6 policies + independent
    python scripts/plot_noise40_combined.py --out results/noise40_500r_combined.png

    # POM only
    python scripts/plot_noise40_combined.py --pom-only --out results/noise40_500r_pom.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# 6 collaborative policies + independent (500-round jobs)
POLICIES = [
    ("MWM_AccDiff",   "noise40_500r_mwm_acc_1071100",      "tensor_ens_acc_10_resnet_MWM_AccDiff_r1.pt",   "tensor_avg_learner_acc_10_resnet_MWM_AccDiff_r1.pt"),
    ("MWM_ClassDist", "noise40_500r_MWM_ClassDist_1071101_0", "tensor_ens_acc_10_resnet_MWM_ClassDist_r1.pt", "tensor_avg_learner_acc_10_resnet_MWM_ClassDist_r1.pt"),
    ("AccDiff",       "noise40_500r_AccDiff_1071101_1",    "tensor_ens_acc_10_resnet_AccDiff_r1.pt",       "tensor_avg_learner_acc_10_resnet_AccDiff_r1.pt"),
    ("ClassDist",     "noise40_500r_ClassDist_1071101_2",  "tensor_ens_acc_10_resnet_ClassDist_r1.pt",     "tensor_avg_learner_acc_10_resnet_ClassDist_r1.pt"),
    ("RRG_AccDiff",   "noise40_500r_RRG_AccDiff_1071101_3","tensor_ens_acc_10_resnet_RRG_AccDiff_r1.pt",   "tensor_avg_learner_acc_10_resnet_RRG_AccDiff_r1.pt"),
    ("RRg_Random",    "noise40_500r_RRg_Random_1071101_4", "tensor_ens_acc_10_resnet_RRg_Random_r1.pt",    "tensor_avg_learner_acc_10_resnet_RRg_Random_r1.pt"),
    ("Independent",   "noise40_500r_independent_1071102",  "tensor_ens_acc_9_resnet_independent_r1.pt",    "tensor_avg_learner_acc_9_resnet_independent_r1.pt"),
]

POM_ENTRY = ("POM", "noise40_500r_pom_1071132",
             "tensor_ens_acc_10_resnet_POM_r1.pt",
             "tensor_avg_learner_acc_10_resnet_POM_r1.pt")

COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
LINESTYLES = ["-", "-", "-", "-", "-", "-", "--"]
LINEWIDTHS = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


def load(base: Path, subdir: str, fname: str):
    path = base / subdir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def style_ax(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_entries(axes, entries, colors, linestyles, linewidths, base):
    for i, (name, subdir, ens_file, learner_file) in enumerate(entries):
        ens_acc     = load(base, subdir, ens_file)
        learner_acc = load(base, subdir, learner_file)
        rounds = range(len(ens_acc))
        c, ls, lw = colors[i], linestyles[i], linewidths[i]
        axes[0].plot(rounds, ens_acc,     color=c, linestyle=ls, linewidth=lw, label=name)
        axes[1].plot(rounds, learner_acc, color=c, linestyle=ls, linewidth=lw, label=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default="results/")
    parser.add_argument("--out", default="results/noise40_500r_combined.png")
    parser.add_argument("--pom-only", action="store_true", help="Plot POM only")
    args = parser.parse_args()

    base = Path(args.results_base)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if args.pom_only:
        fig.suptitle("ResNet-18 / CIFAR-10 — 40% Noise — POM (500 rounds)", fontsize=13)
        plot_entries(axes, [POM_ENTRY], ["#e6194b"], ["-"], [2.0], base)
    else:
        fig.suptitle("ResNet-18 / CIFAR-10 — 40% Symmetric Label Noise (500 rounds)", fontsize=13)
        plot_entries(axes, POLICIES, COLORS, LINESTYLES, LINEWIDTHS, base)

    style_ax(axes[0], "Ensemble Accuracy (val, no oracle)")
    style_ax(axes[1], "Avg Learner Accuracy (val, no oracle)")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
