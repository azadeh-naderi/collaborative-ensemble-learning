"""
Diversity experiment — Independent baseline: 9 models, CE on (optionally noisy)
labels, no KD. Tracks same diversity metrics as run_diversity.py.

Run from repo root:
    python experiments/diversity/run_diversity_independent.py \
        --config configs/diversity_mwm_acc_220r.yaml \
        --out results/diversity_independent/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.celnet.config import ExperimentConfig, load_config_file
from src.celnet.data import cifar_dataset, noisy_loader
from src.celnet.metrics import (
    accuracy, average_ensemble_accuracy, average_learner_accuracy,
    ensemble_accuracy, pairwise_disagreement_matrix,
)
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import TimeLogger, seed_everything, save_json

from experiments.diversity.run_diversity import (
    per_class_accuracy, collect_predictions, collect_softmax,
)


def run(cfg: ExperimentConfig, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size, seed=cfg.run_seed,
        num_workers=cfg.num_workers, root=cfg.data_root,
    )

    noisy_train_loader = train_loader
    if cfg.label_noise_rate > 0:
        noisy_train_loader, actual = noisy_loader(train_loader, cfg.num_classes, cfg.label_noise_rate, cfg.noise_seed)
        print(f"Label noise: actual={actual:.4f}")

    model_class = get_model_class(cfg.model)
    n_models = cfg.num_models
    models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes,
                         seed=cfg.model_seeds[i], device=device)
        for i in range(n_models)
    ]

    n_train_per_round = n_models // 2
    expected_steps = int(cfg.n_rounds * n_train_per_round / n_models)
    optimizers, schedulers = [], []
    for m in models:
        opt = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        s1 = max(1, int(0.55 * expected_steps))
        s2 = max(s1 + 1, int(0.80 * expected_steps))
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=cfg.gamma)
        optimizers.append(opt)
        schedulers.append(sch)

    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator().manual_seed(cfg.run_seed)

    disagreement_per_round    = torch.zeros(cfg.n_rounds + 1)
    ensemble_error_per_round  = torch.zeros(cfg.n_rounds + 1)
    avg_ind_error_per_round   = torch.zeros(cfg.n_rounds + 1)
    diversity_per_round       = torch.zeros(cfg.n_rounds + 1)

    models_with_idx = [(m, i) for i, m in enumerate(models)]

    # round-0
    ens0 = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
    avg0 = average_learner_accuracy(models_with_idx, val_loader)
    _, _, dis0 = pairwise_disagreement_matrix(models_with_idx, val_loader, cfg.num_classes)
    ensemble_error_per_round[0] = 1 - ens0 / 100
    avg_ind_error_per_round[0]  = 1 - avg0 / 100
    diversity_per_round[0]      = avg_ind_error_per_round[0] - ensemble_error_per_round[0]
    disagreement_per_round[0]   = dis0

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} (Independent) ===")
            perm = torch.randperm(n_models, generator=rng).tolist()
            train_indices = perm[:n_train_per_round]

            for i in train_indices:
                m = models[i]
                m.train()
                for images, labels in noisy_train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizers[i].zero_grad()
                    criterion(m(images), labels).backward()
                    optimizers[i].step()
                schedulers[i].step()

            ens  = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
            avg  = average_learner_accuracy(models_with_idx, val_loader)
            _, _, dis = pairwise_disagreement_matrix(models_with_idx, val_loader, cfg.num_classes)
            r = round_idx + 1
            ensemble_error_per_round[r] = 1 - ens / 100
            avg_ind_error_per_round[r]  = 1 - avg / 100
            diversity_per_round[r]      = avg_ind_error_per_round[r] - ensemble_error_per_round[r]
            disagreement_per_round[r]   = dis
            print(f"  ens_acc={ens:.2f}  avg_acc={avg:.2f}  disagreement={dis:.4f}")

        time.sleep(5)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(disagreement_per_round,   out_dir / "disagreement_per_round.pt")
    torch.save(ensemble_error_per_round, out_dir / "ensemble_error_per_round.pt")
    torch.save(avg_ind_error_per_round,  out_dir / "avg_ind_error_per_round.pt")
    torch.save(diversity_per_round,      out_dir / "diversity_per_round.pt")

    heatmap = torch.stack([per_class_accuracy(m, test_loader, cfg.num_classes, device) for m in models])
    torch.save(heatmap, out_dir / "per_class_acc_heatmap.pt")

    preds = collect_predictions(models_with_idx, test_loader, device)
    labels_all = torch.cat([y for _, y in test_loader])
    torch.save((preds == labels_all.unsqueeze(0)).float(), out_dir / "correct_matrix.pt")
    torch.save(collect_softmax(models_with_idx, test_loader, device), out_dir / "softmax_vecs.pt")

    final_ens = ensemble_accuracy(models_with_idx, test_loader, mode="logprob")
    final_avg = average_learner_accuracy(models_with_idx, test_loader)
    _, _, final_dis = pairwise_disagreement_matrix(models_with_idx, test_loader, cfg.num_classes)
    final_div = (1 - final_avg / 100) - (1 - final_ens / 100)
    print(f"\nFinal — ens_acc={final_ens:.2f}  avg={final_avg:.2f}  disagreement={final_dis:.4f}  diversity={final_div:.4f}")
    save_json({"final_ensemble_acc": final_ens, "final_avg_learner_acc": final_avg,
               "final_disagreement": final_dis, "final_diversity": float(final_div),
               "config": cfg.to_dict()}, out_dir / "diversity_summary.json")
    print(f"All tensors saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    defaults = ExperimentConfig()
    file_cfg = load_config_file(args.config)
    cfg = ExperimentConfig(**{**defaults.to_dict(), **file_cfg})
    cfg.num_models = 9
    run(cfg, Path(args.out))
