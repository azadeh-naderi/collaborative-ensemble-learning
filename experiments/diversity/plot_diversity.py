"""
Generate 4 diversity plots from tensors saved by run_diversity.py.

Usage:
    python experiments/diversity/plot_diversity.py --data results/diversity_mwm_acc/ --out results/diversity_mwm_acc/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CIFAR10_CLASSES = ["airplane", "auto", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load(data_dir: Path, fname: str):
    return torch.load(data_dir / fname, map_location="cpu", weights_only=True).numpy()


def plot_disagreement(data_dir: Path, out_dir: Path):
    dis = load(data_dir, "disagreement_per_round.pt")
    rounds = np.arange(len(dis))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, dis, color="#1f77b4", linewidth=1.8)
    ax.set_title("Pairwise Disagreement over Rounds (MWM_AccDiff)", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg pairwise disagreement")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out_dir / "diversity_disagreement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_heatmap(data_dir: Path, out_dir: Path):
    heatmap = load(data_dir, "per_class_acc_heatmap.pt")  # (num_models, 10)
    n_models = heatmap.shape[0]

    fig, ax = plt.subplots(figsize=(9, max(3, n_models * 0.5 + 1)))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([f"M{i+1}" for i in range(n_models)], fontsize=9)
    ax.set_title("Per-class Accuracy Heatmap — Final Models (MWM_AccDiff)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Accuracy")
    # annotate cells
    for i in range(n_models):
        for j in range(10):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if heatmap[i, j] < 0.75 else "white")
    plt.tight_layout()
    path = out_dir / "diversity_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ambiguity(data_dir: Path, out_dir: Path):
    ens_err = load(data_dir, "ensemble_error_per_round.pt")
    avg_err = load(data_dir, "avg_ind_error_per_round.pt")
    div     = load(data_dir, "diversity_per_round.pt")
    rounds  = np.arange(len(ens_err))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(rounds, 0, ens_err, alpha=0.7, color="#1f77b4", label="Ensemble error")
    ax.fill_between(rounds, ens_err, avg_err, alpha=0.6, color="#ff7f0e", label="Diversity gain")
    ax.plot(rounds, avg_err, color="#d62728", linewidth=1.5, linestyle="--", label="Avg individual error")
    ax.set_title("Ambiguity Decomposition over Rounds (MWM_AccDiff)", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Error rate")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out_dir / "diversity_ambiguity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(data_dir: Path, out_dir: Path):
    correct_mat = load(data_dir, "correct_matrix.pt")  # (num_models, N_test)
    n_models = correct_mat.shape[0]

    # PCA on correctness vectors
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(correct_mat)  # (num_models, 2)

    # t-SNE (only meaningful if num_models > 5; perplexity must be < num_models)
    perp = min(5, n_models - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords_tsne = tsne.fit_transform(correct_mat)

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Prediction Space (MWM_AccDiff) — final non-oracle models", fontsize=12)

    for ax, coords, title in [
        (axes[0], coords_pca,  f"PCA (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)"),
        (axes[1], coords_tsne, "t-SNE"),
    ]:
        for i in range(n_models):
            ax.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=120, zorder=3,
                       label=f"M{i+1}", edgecolors="white", linewidths=0.8)
            ax.annotate(f"M{i+1}", (coords[i, 0], coords[i, 1]),
                        fontsize=9, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    path = out_dir / "diversity_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory with saved tensors")
    parser.add_argument("--out",  required=True, help="Output directory for plots")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_disagreement(data_dir, out_dir)
    plot_heatmap(data_dir, out_dir)
    plot_ambiguity(data_dir, out_dir)
    plot_scatter(data_dir, out_dir)
    print("\nAll 4 diversity plots saved.")


if __name__ == "__main__":
    main()
