"""
Compare CelNet / CE / KD trained models by random-perturbation sharpness.

For each model, we repeatedly add relative Gaussian noise to its weights
(perturbation magnitude sigma * |w| per element) and measure how much the
test accuracy / loss degrades. A model sitting in a flatter (wider) minimum
degrades less for the same sigma -> lower sharpness, better generalization.

Each --*-path points to a file produced via:
    torch.save([(model.state_dict(), idx) for model, idx in updated_models], path)

Usage:
    python scripts/flatness_experiment.py \
        --celnet-path /path/to/trained_models_10_resnet_MWM_AccDiff_New \
        --ce-path /path/to/trained_CE_models_10_resnet_122epochs \
        --kd-path /path/to/trained_KD_models_10_resnet_122epochs \
        --sigmas 0.0 0.01 0.02 0.05 0.1 0.2 \
        --repeats 5 \
        --output results/flatness
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celnet.data import cifar_dataset  # noqa: E402
from celnet.models import ResNet  # noqa: E402
from celnet.plotting import save_sharpness_plot  # noqa: E402


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def perturb_inplace(model, sigma, generator):
    """Add relative Gaussian noise N(0, (sigma * |w|)^2) to every parameter."""
    for p in model.parameters():
        if sigma == 0.0:
            continue
        noise = torch.randn(p.shape, generator=generator, device="cpu").to(p.device)
        p.add_(noise * p.abs() * sigma)


def load_models(path, input_channels, num_classes, pretrained, device):
    entries = torch.load(path, map_location=device)
    models = []
    for entry in entries:
        state_dict, idx = entry
        model = ResNet(input_channels, pretrained, num_classes).to(device)
        model.load_state_dict(state_dict)
        models.append((model, idx))
    return models


def run_method(name, path, loader, sigmas, repeats, input_channels, num_classes, pretrained, device, base_seed):
    rows = []
    models = load_models(path, input_channels, num_classes, pretrained, device)
    print(f"[{name}] loaded {len(models)} models from {path}")

    for model, idx in models:
        base_acc, base_loss = eval_model(model, loader, device)
        rows.append({"method": name, "model_idx": idx, "sigma": 0.0, "repeat": 0,
                      "accuracy": base_acc, "loss": base_loss,
                      "acc_drop": 0.0, "loss_increase": 0.0})
        print(f"  M{idx}: base_acc={base_acc:.2f} base_loss={base_loss:.4f}")

        for sigma in sigmas:
            if sigma == 0.0:
                continue
            for r in range(repeats):
                gen = torch.Generator().manual_seed(base_seed + idx * 1000 + r)
                perturbed = copy.deepcopy(model)
                perturb_inplace(perturbed, sigma, gen)
                acc, loss = eval_model(perturbed, loader, device)
                rows.append({"method": name, "model_idx": idx, "sigma": sigma, "repeat": r,
                              "accuracy": acc, "loss": loss,
                              "acc_drop": base_acc - acc, "loss_increase": loss - base_loss})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sharpness/flatness comparison across training methods")
    parser.add_argument("--celnet-path", type=str, required=True)
    parser.add_argument("--ce-path", type=str, required=True)
    parser.add_argument("--kd-path", type=str, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/flatness")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = cifar_dataset(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, root=args.data_root
    )

    all_rows = []
    for name, path in [("CelNet", args.celnet_path), ("CE", args.ce_path), ("KD", args.kd_path)]:
        all_rows.extend(run_method(
            name, path, test_loader, args.sigmas, args.repeats,
            args.input_channels, args.num_classes, False, device, args.seed,
        ))

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "flatness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    summary = (
        df[df["sigma"] > 0]
        .groupby(["method", "sigma"])[["accuracy", "acc_drop", "loss_increase"]]
        .agg(["mean", "std"])
    )
    summary_path = output_dir / "flatness_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")
    print(summary)

    # A single scalar sharpness score per method: mean accuracy drop across all sigmas > 0
    score = df[df["sigma"] > 0].groupby("method")["acc_drop"].mean().sort_values()
    print("\nSharpness score (lower = flatter minimum):")
    print(score)
    score.to_csv(output_dir / "sharpness_score.csv")

    plot_path = save_sharpness_plot(df, output_dir / "sharpness_comparison.png")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
