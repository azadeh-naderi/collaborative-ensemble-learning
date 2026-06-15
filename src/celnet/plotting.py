from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_tensor(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return t


def save_gain_plots(n_rounds, oracle_gain_student, non_oracle_gain_mean, oracle_gain_student_cum, non_oracle_gain_mean_cum, num_models, model_name, pairing_strategy, run_id, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rounds = np.arange(1, n_rounds + 1)
    rounds_cum = np.arange(1, n_rounds + 2)

    fig1 = results_dir / f"{num_models}_mean_{model_name}_{pairing_strategy}_r{run_id}.png"
    plt.figure(figsize=(9, 5))
    plt.plot(rounds, _to_numpy(oracle_gain_student), label="Oracle Gain student", linewidth=2)
    plt.plot(rounds, _to_numpy(non_oracle_gain_mean), label="Non-Oracle Gain Mean per Model", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Accuracy Gain")
    plt.title("Oracle vs Non-Oracle Gain per Round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    fig2 = results_dir / f"{num_models}_cumulative_mean_{model_name}_{pairing_strategy}_r{run_id}.png"
    plt.figure(figsize=(9, 5))
    plt.plot(rounds_cum, _to_numpy(oracle_gain_student_cum), label="Cumulative Oracle Gain student", linewidth=2)
    plt.plot(rounds_cum, _to_numpy(non_oracle_gain_mean_cum), label="Cumulative Non-Oracle Gain Mean per Model", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Accuracy Gain")
    plt.title("Aggregate Progress by Session Type")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    return str(fig1), str(fig2)


def save_per_model_plots(per_model_acc, oracle_gain_per_model, non_oracle_gain_per_model, oracle_gain_per_model_cum, non_oracle_gain_per_model_cum, num_models, model_name, pairing_strategy, run_id, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_rounds = oracle_gain_per_model.shape[1]
    round_axis = np.arange(0, n_rounds + 1)
    rounds = np.arange(1, n_rounds + 1)

    per_model_acc = _to_numpy(per_model_acc)
    og = _to_numpy(oracle_gain_per_model)
    ng = _to_numpy(non_oracle_gain_per_model)
    og_cum = _to_numpy(oracle_gain_per_model_cum)
    ng_cum = _to_numpy(non_oracle_gain_per_model_cum)

    paths = []

    # Per-model accuracy across rounds
    fig_acc = results_dir / f"{num_models}_per_model_acc_{model_name}_{pairing_strategy}_r{run_id}.png"
    plt.figure(figsize=(14, 8))
    for mid in range(num_models):
        plt.plot(round_axis, per_model_acc[mid], label=f"M{mid}", linewidth=1.8)
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Model Accuracy Across Rounds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_acc, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(str(fig_acc))

    # Cumulative oracle/non-oracle gain per model
    fig_cum = results_dir / f"{num_models}_per_model_oracle_vs_nonoracle_gain_{model_name}_{pairing_strategy}_r{run_id}.png"
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for mid in range(num_models):
        axes[0].plot(round_axis, og_cum[mid], label=f"M{mid}", linewidth=1.8)
        axes[1].plot(round_axis, ng_cum[mid], label=f"M{mid}", linewidth=1.8)
    axes[0].set_title("Cumulative Oracle Gain per Model")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy Gain")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Cumulative Non-Oracle Gain per Model")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy Gain")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_cum, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(str(fig_cum))

    # Heatmaps of per-round gains
    fig_heat = results_dir / f"{num_models}_heatmap_og_ng_per_model_{model_name}_{pairing_strategy}_r{run_id}.png"
    og_plot = np.nan_to_num(og, nan=0.0)
    ng_plot = np.nan_to_num(ng, nan=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    im1 = axes[0].imshow(og_plot, aspect="auto", interpolation="nearest")
    axes[0].set_title("Oracle Gain per Round per Model")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Model")
    axes[0].set_yticks(np.arange(num_models))
    axes[0].set_yticklabels([f"M{i}" for i in range(num_models)])
    axes[0].set_xticks(np.arange(0, n_rounds, max(1, n_rounds // 10)))
    im2 = axes[1].imshow(ng_plot, aspect="auto", interpolation="nearest")
    axes[1].set_title("Non-Oracle Gain per Round per Model")
    axes[1].set_xlabel("Round")
    axes[1].set_yticks(np.arange(num_models))
    axes[1].set_yticklabels([f"M{i}" for i in range(num_models)])
    axes[1].set_xticks(np.arange(0, n_rounds, max(1, n_rounds // 10)))
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label("Accuracy Gain")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label("Accuracy Gain")
    plt.tight_layout()
    plt.savefig(fig_heat, dpi=200, bbox_inches="tight")
    plt.close()
    paths.append(str(fig_heat))

    # Bar chart of total gains per model
    fig_bar = results_dir / f"{num_models}_total_og_ng_per_model_{model_name}_{pairing_strategy}_r{run_id}.png"
    oracle_total = np.nan_to_num(og, nan=0.0).sum(axis=1)
    non_oracle_total = np.nan_to_num(ng, nan=0.0).sum(axis=1)
    x = np.arange(num_models)
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, oracle_total, width=width, label="Oracle Gain")
    plt.bar(x + width / 2, non_oracle_total, width=width, label="Non-Oracle Gain")
    plt.xlabel("Model")
    plt.ylabel("Total Accuracy Gain")
    plt.title("Total Oracle vs Non-Oracle Gain per Model")
    plt.xticks(x, [f"M{i}" for i in range(num_models)])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_bar, dpi=200)
    plt.close()
    paths.append(str(fig_bar))

    return paths


def save_policy_comparison_plot(policy_tensors: Mapping[str, Tuple[object, object]], output_path: str | Path, title: str = "Aggregate Progress Across Pairing Policies"):
    """
    policy_tensors maps policy name -> (oracle_cumulative_tensor, non_oracle_cumulative_tensor)
    Each value may be a torch.Tensor, numpy array, or list.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for policy_name, (oracle_vals, non_oracle_vals) in policy_tensors.items():
        plt.plot(_to_numpy(oracle_vals), linestyle="--", label=f"{policy_name} (Oracle)")
        plt.plot(_to_numpy(non_oracle_vals), linewidth=2, label=f"{policy_name} (Non-Oracle)")

    plt.xlabel("Round")
    plt.ylabel("Cumulative Accuracy Gain")
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)
