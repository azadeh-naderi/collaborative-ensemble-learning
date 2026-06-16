"""
Combined plot for all noise40 experiments:
  - 6 collaborative policies (MWM_AccDiff, MWM_ClassDist, AccDiff, ClassDist, RRG_AccDiff, RRg_Random)
  - 1 independent baseline (9 models, no KD)

Usage (run from repo root on Wulver):
    python scripts/plot_noise40_combined.py --results-base results/ --out results/noise40_combined.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

POLICIES = [
    ("MWM_AccDiff",   "noise40_mwm_acc_1070470",          "tensor_ens_acc_10_resnet_MWM_AccDiff_r1.pt",    "tensor_avg_learner_acc_10_resnet_MWM_AccDiff_r1.pt"),
    ("MWM_ClassDist", "noise40_MWM_ClassDist_1070471_0",   "tensor_ens_acc_10_resnet_MWM_ClassDist_r1.pt",  "tensor_avg_learner_acc_10_resnet_MWM_ClassDist_r1.pt"),
    ("AccDiff",       "noise40_AccDiff_1070471_1",         "tensor_ens_acc_10_resnet_AccDiff_r1.pt",        "tensor_avg_learner_acc_10_resnet_AccDiff_r1.pt"),
    ("ClassDist",     "noise40_ClassDist_1070471_2",       "tensor_ens_acc_10_resnet_ClassDist_r1.pt",      "tensor_avg_learner_acc_10_resnet_ClassDist_r1.pt"),
    ("RRG_AccDiff",   "noise40_RRG_AccDiff_1070471_3",     "tensor_ens_acc_10_resnet_RRG_AccDiff_r1.pt",    "tensor_avg_learner_acc_10_resnet_RRG_AccDiff_r1.pt"),
    ("RRg_Random",    "noise40_RRg_Random_1070471_4",      "tensor_ens_acc_10_resnet_RRg_Random_r1.pt",     "tensor_avg_learner_acc_10_resnet_RRg_Random_r1.pt"),
    ("Independent",   "noise40_independent_1070485",       "tensor_ens_acc_9_resnet_independent_r1.pt",     "tensor_avg_learner_acc_9_resnet_independent_r1.pt"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
LINESTYLES = ["-", "-", "-", "-", "-", "-", "--"]


def load(base: Path, subdir: str, fname: str):
    path = base / subdir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return torch.load(path, map_location="cpu").numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default="results/")
    parser.add_argument("--out", default="results/noise40_combined.png")
    args = parser.parse_args()

    base = Path(args.results_base)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResNet-18 / CIFAR-10 — 40% Symmetric Label Noise (220 rounds)", fontsize=13)

    for i, (name, subdir, ens_file, learner_file) in enumerate(POLICIES):
        ens_acc    = load(base, subdir, ens_file)
        learner_acc = load(base, subdir, learner_file)
        rounds = range(len(ens_acc))
        c, ls = COLORS[i], LINESTYLES[i]
        axes[0].plot(rounds, ens_acc,    color=c, linestyle=ls, linewidth=1.5, label=name)
        axes[1].plot(rounds, learner_acc, color=c, linestyle=ls, linewidth=1.5, label=name)

    for ax, title, ylabel in [
        (axes[0], "Ensemble Accuracy (val, no oracle)",   "Accuracy (%)"),
        (axes[1], "Avg Learner Accuracy (val, no oracle)", "Accuracy (%)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
