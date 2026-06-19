"""
Diversity experiment: MWM_AccDiff, 220 rounds, ResNet-18 / CIFAR-10.

Tracks per-round:
  - pairwise disagreement
  - ensemble error (1 - ens_acc)
  - avg individual error (1 - avg_learner_acc)
  - diversity = avg_individual_error - ensemble_error

At end of training saves final model predictions on test set for heatmap and
t-SNE / PCA scatter plots.

Run from repo root:
    python experiments/diversity/run_diversity.py \
        --config configs/diversity_mwm_acc_220r.yaml \
        --out results/diversity_mwm_acc/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.celnet.config import ExperimentConfig, build_config_from_args, load_config_file
from src.celnet.data import cifar_dataset, noisy_loader, split_validation_set_kfold
from src.celnet.metrics import (
    accuracy, average_ensemble_accuracy, average_ensemble_confidence,
    average_learner_accuracy, ensemble_accuracy, ensemble_val_acc_no_oracle,
    get_models_no_oracle, pairwise_disagreement_matrix,
)
from src.celnet.models import get_model_class, initialize_model
from src.celnet.pairing import build_pairing_methods, make_fixed_friend_groups_by_id
from src.celnet.training import OptimizerRegistry, train_student_kd, train_student_oracle
from src.celnet.utils import TimeLogger, seed_everything, save_json


@torch.no_grad()
def per_class_accuracy(model, data_loader, num_classes, device):
    model.eval()
    correct = torch.zeros(num_classes, device=device)
    total   = torch.zeros(num_classes, device=device)
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for c in range(num_classes):
            mask = labels == c
            correct[c] += (preds[mask] == labels[mask]).sum()
            total[c]   += mask.sum()
    acc = torch.zeros(num_classes)
    nz = total > 0
    acc[nz] = (correct[nz] / total[nz]).cpu()
    return acc


@torch.no_grad()
def collect_predictions(models_with_idx, data_loader, device):
    """Return (num_models, N) tensor of predicted class labels on data_loader."""
    all_preds = []
    for model, _ in models_with_idx:
        model.eval()
        preds = []
        for images, _ in data_loader:
            images = images.to(device)
            preds.append(model(images).argmax(dim=1).cpu())
        all_preds.append(torch.cat(preds))
    return torch.stack(all_preds)  # (num_models, N)


@torch.no_grad()
def collect_softmax(models_with_idx, data_loader, device):
    """Return (num_models, num_classes) mean softmax probability vector per model."""
    vecs = []
    for model, _ in models_with_idx:
        model.eval()
        probs = []
        for images, _ in data_loader:
            images = images.to(device)
            probs.append(F.softmax(model(images), dim=1).cpu())
        vecs.append(torch.cat(probs).mean(dim=0))
    return torch.stack(vecs)  # (num_models, num_classes)


def run(cfg: ExperimentConfig, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size, seed=cfg.run_seed,
        num_workers=cfg.num_workers, root=cfg.data_root,
    )
    val_splits = split_validation_set_kfold(val_loader.dataset, 10, cfg.run_seed)

    oracle_train_loader = train_loader
    if cfg.label_noise_rate > 0:
        oracle_train_loader, actual = noisy_loader(train_loader, cfg.num_classes, cfg.label_noise_rate, cfg.noise_seed)
        print(f"Oracle label noise: actual={actual:.4f}")

    model_class = get_model_class(cfg.model)
    init_models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes,
                         seed=s, device=device)
        for s in cfg.model_seeds[:cfg.num_models]
    ]
    updated_models = [(m, i) for i, m in enumerate(init_models)]
    oracle_id = 0

    # tracking tensors
    disagreement_per_round    = torch.zeros(cfg.n_rounds + 1)
    ensemble_error_per_round  = torch.zeros(cfg.n_rounds + 1)
    avg_ind_error_per_round   = torch.zeros(cfg.n_rounds + 1)
    diversity_per_round       = torch.zeros(cfg.n_rounds + 1)

    registry = OptimizerRegistry()
    fixed_group_ids = None
    DISAGREE_INTERVAL = 5  # compute disagreement every N rounds to reduce overhead

    # round-0
    models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
    ens_acc0  = average_ensemble_accuracy(models_no_oracle, val_loader, cfg.num_classes)
    avg_acc0  = average_learner_accuracy(models_no_oracle, val_loader)
    _, _, dis0 = pairwise_disagreement_matrix(models_no_oracle, val_loader, cfg.num_classes)
    ensemble_error_per_round[0] = 1 - ens_acc0 / 100
    avg_ind_error_per_round[0]  = 1 - avg_acc0 / 100
    diversity_per_round[0]      = avg_ind_error_per_round[0] - ensemble_error_per_round[0]
    disagreement_per_round[0]   = dis0
    last_dis = dis0

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            pairing_methods = build_pairing_methods(
                updated_models=updated_models, val_loader=val_loader,
                val_splits=val_splits, num_classes=cfg.num_classes,
                degree=cfg.degree, run_seed=cfg.run_seed,
                batch_size=cfg.batch_size, round_idx=round_idx,
                fixed_group_ids=fixed_group_ids,
            )
            pairs = pairing_methods[cfg.pairing_strategy]()
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} ({cfg.pairing_strategy}) ===")

            for (m1, id1), (m2, id2) in pairs:
                if id1 == oracle_id or id2 == oracle_id:
                    student, student_id = (m2, id2) if id1 == oracle_id else (m1, id1)
                    opt, sch = registry.get(student_id, student, cfg.optimizer_type, cfg.learning_rate,
                                            cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma)
                    student = train_student_oracle(student, oracle_train_loader, opt, device, sch)
                    for i, (_m, mid) in enumerate(updated_models):
                        if mid == student_id:
                            updated_models[i] = (student, student_id); break
                    continue

                acc1 = float(accuracy(m1, val_loader))
                acc2 = float(accuracy(m2, val_loader))
                if acc2 > acc1:
                    teacher, teacher_id = m2, id2
                    student, student_id = m1, id1
                else:
                    teacher, teacher_id = m1, id1
                    student, student_id = m2, id2
                opt, sch = registry.get(student_id, student, cfg.optimizer_type, cfg.learning_rate,
                                        cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma)
                student = train_student_kd(student, teacher, train_loader, opt, device, cfg.temperature, cfg.alpha, sch)
                for i, (_m, mid) in enumerate(updated_models):
                    if mid == student_id:
                        updated_models[i] = (student, student_id); break

            models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
            ens_acc = average_ensemble_accuracy(models_no_oracle, val_loader, cfg.num_classes)
            avg_acc = average_learner_accuracy(models_no_oracle, val_loader)

            r = round_idx + 1
            if r % DISAGREE_INTERVAL == 0 or r == cfg.n_rounds:
                _, _, last_dis = pairwise_disagreement_matrix(models_no_oracle, val_loader, cfg.num_classes)
            disagreement_per_round[r]   = last_dis
            ensemble_error_per_round[r] = 1 - ens_acc / 100
            avg_ind_error_per_round[r]  = 1 - avg_acc / 100
            diversity_per_round[r]      = avg_ind_error_per_round[r] - ensemble_error_per_round[r]
            print(f"  ens_acc={ens_acc:.2f}  avg_acc={avg_acc:.2f}  disagreement={last_dis:.4f}  diversity={diversity_per_round[r]:.4f}")

        time.sleep(5)

    # save tensors
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(disagreement_per_round,   out_dir / "disagreement_per_round.pt")
    torch.save(ensemble_error_per_round, out_dir / "ensemble_error_per_round.pt")
    torch.save(avg_ind_error_per_round,  out_dir / "avg_ind_error_per_round.pt")
    torch.save(diversity_per_round,      out_dir / "diversity_per_round.pt")

    # final-state metrics on test set
    final_no_oracle = get_models_no_oracle(updated_models, oracle_id)

    # per-class accuracy heatmap: (num_models, num_classes)
    heatmap = torch.stack([
        per_class_accuracy(m, test_loader, cfg.num_classes, device)
        for m, _ in final_no_oracle
    ])
    torch.save(heatmap, out_dir / "per_class_acc_heatmap.pt")

    # predictions for scatter: (num_models, N_test) — correctness per sample
    preds = collect_predictions(final_no_oracle, test_loader, device)
    labels_all = torch.cat([y for _, y in test_loader])
    correct_mat = (preds == labels_all.unsqueeze(0)).float()  # (num_models, N)
    torch.save(correct_mat, out_dir / "correct_matrix.pt")

    # mean softmax per model for t-SNE input: (num_models, num_classes)
    softmax_vecs = collect_softmax(final_no_oracle, test_loader, device)
    torch.save(softmax_vecs, out_dir / "softmax_vecs.pt")

    # final accuracy numbers
    final_ens = ensemble_accuracy(final_no_oracle, test_loader, mode="logprob")
    final_avg = average_learner_accuracy(final_no_oracle, test_loader)
    _, _, final_dis = pairwise_disagreement_matrix(final_no_oracle, test_loader, cfg.num_classes)
    final_div = (1 - final_avg / 100) - (1 - final_ens / 100)
    print(f"\nFinal — ens_acc={final_ens:.2f}  avg_learner={final_avg:.2f}  disagreement={final_dis:.4f}  diversity={final_div:.4f}")

    save_json({
        "final_ensemble_acc": final_ens,
        "final_avg_learner_acc": final_avg,
        "final_disagreement": final_dis,
        "final_diversity": float(final_div),
        "config": cfg.to_dict(),
    }, out_dir / "diversity_summary.json")

    print(f"\nAll tensors saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairing-strategy", dest="pairing_strategy", default=None)
    args = parser.parse_args()

    defaults = ExperimentConfig()
    file_cfg = load_config_file(args.config)
    cfg = ExperimentConfig(**{**defaults.to_dict(), **file_cfg})
    if args.pairing_strategy:
        cfg.pairing_strategy = args.pairing_strategy
    run(cfg, Path(args.out))
