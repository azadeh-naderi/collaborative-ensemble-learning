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
