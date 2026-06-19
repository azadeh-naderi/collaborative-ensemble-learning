"""
Diversity experiment — Traditional KD: one fixed teacher (model 0 = oracle),
9 student models all distill from the same teacher every round. Expected to
produce low-diversity ensemble since all students chase the same target.

Run from repo root:
    python experiments/diversity/run_diversity_trad_kd.py \
        --config configs/diversity_mwm_acc_220r.yaml \
        --out results/diversity_trad_kd/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.celnet.config import ExperimentConfig, load_config_file
from src.celnet.data import cifar_dataset
from src.celnet.metrics import (
    accuracy, average_ensemble_accuracy, average_learner_accuracy,
    ensemble_accuracy, pairwise_disagreement_matrix,
)
from src.celnet.models import get_model_class, initialize_model
from src.celnet.training import train_student_kd
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

    model_class = get_model_class(cfg.model)
    n_students = cfg.num_models  # 9 students
    # teacher: separate model with a distinct seed
    teacher = initialize_model(model_class, cfg.input_channels, cfg.pretrained,
                               cfg.num_classes, seed=999999, device=device)
    # pre-train teacher for 10 epochs before collaboration
    print("Pre-training teacher (10 epochs)...")
    t_opt = torch.optim.SGD(teacher.parameters(), lr=cfg.learning_rate,
                             momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    import torch.nn as nn
    ce = nn.CrossEntropyLoss()
    for ep in range(10):
        teacher.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            t_opt.zero_grad()
            ce(teacher(images), labels).backward()
            t_opt.step()
    teacher.eval()
    teacher_acc = accuracy(teacher, val_loader)
    print(f"Teacher val acc after pre-training: {teacher_acc:.2f}")

    students = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained,
                         cfg.num_classes, seed=cfg.model_seeds[i], device=device)
        for i in range(n_students)
    ]
    optimizers, schedulers = [], []
    for s in students:
        opt = torch.optim.SGD(s.parameters(), lr=cfg.learning_rate,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        m1 = max(1, int(0.55 * cfg.n_rounds))
        m2 = max(m1 + 1, int(0.80 * cfg.n_rounds))
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=cfg.gamma)
        optimizers.append(opt)
        schedulers.append(sch)

    models_with_idx = [(s, i) for i, s in enumerate(students)]

    disagreement_per_round    = torch.zeros(cfg.n_rounds + 1)
    ensemble_error_per_round  = torch.zeros(cfg.n_rounds + 1)
    avg_ind_error_per_round   = torch.zeros(cfg.n_rounds + 1)
    diversity_per_round       = torch.zeros(cfg.n_rounds + 1)

    # round-0
    ens0 = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
    avg0 = average_learner_accuracy(models_with_idx, val_loader)
    _, _, dis0 = pairwise_disagreement_matrix(models_with_idx, val_loader, cfg.num_classes)
    ensemble_error_per_round[0] = 1 - ens0 / 100
    avg_ind_error_per_round[0]  = 1 - avg0 / 100
    diversity_per_round[0]      = avg_ind_error_per_round[0] - ensemble_error_per_round[0]
    disagreement_per_round[0]   = dis0
    last_dis = dis0
    DISAGREE_INTERVAL = 5

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} (TradKD) ===")
            student_id = round_idx % n_students
            student = students[student_id]
            student = train_student_kd(student, teacher, train_loader,
                                       optimizers[student_id], device,
                                       cfg.temperature, cfg.alpha, schedulers[student_id])
            students[student_id] = student
            models_with_idx[student_id] = (student, student_id)

            ens  = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
            avg  = average_learner_accuracy(models_with_idx, val_loader)
            r = round_idx + 1
            if r % DISAGREE_INTERVAL == 0 or r == cfg.n_rounds:
                _, _, last_dis = pairwise_disagreement_matrix(models_with_idx, val_loader, cfg.num_classes)
            ensemble_error_per_round[r] = 1 - ens / 100
            avg_ind_error_per_round[r]  = 1 - avg / 100
            diversity_per_round[r]      = avg_ind_error_per_round[r] - ensemble_error_per_round[r]
            disagreement_per_round[r]   = last_dis
            print(f"  student={student_id}  ens_acc={ens:.2f}  avg_acc={avg:.2f}  disagreement={last_dis:.4f}")

        time.sleep(5)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(disagreement_per_round,   out_dir / "disagreement_per_round.pt")
    torch.save(ensemble_error_per_round, out_dir / "ensemble_error_per_round.pt")
    torch.save(avg_ind_error_per_round,  out_dir / "avg_ind_error_per_round.pt")
    torch.save(diversity_per_round,      out_dir / "diversity_per_round.pt")

    heatmap = torch.stack([per_class_accuracy(s, test_loader, cfg.num_classes, device) for s in students])
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
               "teacher_val_acc": float(teacher_acc), "config": cfg.to_dict()},
              out_dir / "diversity_summary.json")
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
