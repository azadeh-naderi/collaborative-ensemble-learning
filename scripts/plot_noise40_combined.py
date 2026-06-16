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
import matplotlib.ticker as ticker
import torch

POLICIES = [
    ("MWM_AccDiff",   "noise40_mwm_acc_1070470",        "tensor_ens_acc_10_resnet_MWM_AccDiff_r1.pt",    "tensor_avg_learner_acc_10_resnet_MWM_AccDiff_r1.pt"),
    ("MWM_ClassDist", "noise40_MWM_ClassDist_1070471_0", "tensor_ens_acc_10_resnet_MWM_ClassDist_r1.pt",  "tensor_avg_learner_acc_10_resnet_MWM_ClassDist_r1.pt"),
    ("AccDiff",       "noise40_AccDiff_1070471_1",       "tensor_ens_acc_10_resnet_AccDiff_r1.pt",        "tensor_avg_learner_acc_10_resnet_AccDiff_r1.pt"),
    ("ClassDist",     "noise40_ClassDist_1070471_2",     "tensor_ens_acc_10_resnet_ClassDist_r1.pt",      "tensor_avg_learner_acc_10_resnet_ClassDist_r1.pt"),
    ("RRG_AccDiff",   "noise40_RRG_AccDiff_1070471_3",   "tensor_ens_acc_10_resnet_RRG_AccDiff_r1.pt",    "tensor_avg_learner_acc_10_resnet_RRG_AccDiff_r1.pt"),
    ("RRg_Random",    "noise40_RRg_Random_1070471_4",    "tensor_ens_acc_10_resnet_RRg_Random_r1.pt",     "tensor_avg_learner_acc_10_resnet_RRg_Random_r1.pt"),
    ("Independent",   "noise40_independent_1070485",     "tensor_ens_acc_9_resnet_independent_r1.pt",     "tensor_avg_learner_acc_9_resnet_independent_r1.pt"),
]

# Distinct colors + the independent baseline gets black dashed
COLORS     = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#000000"]
LINESTYLES = ["-",       "-",       "-",       "-",       "-",       "-",       "--"]
LINEWIDTHS = [2.0,        2.0,       2.0,       2.0,       2.0,       2.0,       2.5]


def load(base: Path, subdir: str, fname: str):
    path = base / subdir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def style_ax(ax, title, xlabel="Round", ylabel="Accuracy (%)"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(fontsize=9, framealpha=0.85, loc="lower right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default="results/")
    parser.add_argument("--out", default="results/noise40_combined.png")
    args = parser.parse_args()

    base = Path(args.results_base)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "ResNet-18 / CIFAR-10  —  40% Symmetric Label Noise  (220 rounds)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, (name, subdir, ens_file, learner_file) in enumerate(POLICIES):
        ens_acc     = load(base, subdir, ens_file)
        learner_acc = load(base, subdir, learner_file)
        rounds = range(len(ens_acc))
        c, ls, lw = COLORS[i], LINESTYLES[i], LINEWIDTHS[i]
        axes[0].plot(rounds, ens_acc,     color=c, linestyle=ls, linewidth=lw, label=name)
        axes[1].plot(rounds, learner_acc, color=c, linestyle=ls, linewidth=lw, label=name)

    style_ax(axes[0], "Ensemble Accuracy (val, no oracle)")
    style_ax(axes[1], "Avg Learner Accuracy (val, no oracle)")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
