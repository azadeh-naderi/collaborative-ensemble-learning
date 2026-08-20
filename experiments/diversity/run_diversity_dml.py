"""
Diversity experiment — Deep Mutual Learning (DML) baseline.

All 9 models have full access to true labels at every round (CE with ground
truth), PLUS peer KL-distillation from a randomly-selected partner. This is
the DML regime (Zhang et al., 2018): every model minimises

    L = (1-alpha)*CE(output, true_label) + alpha*T^2*KL(output || peer_output)

Round-robin partner selection: model i pairs with model (i+1) % n_models each
round, rotating so every model sees every other model as a peer over time.

Expected outcome: models converge toward the same supervised optimum → lower
pairwise disagreement and diversity gain compared to CEL-Net (where true-label
access is restricted to Oracle sessions only).

Run from repo root:
    python experiments/diversity/run_diversity_dml.py \
        --config configs/diversity_mwm_acc_220r.yaml \
        --out results/diversity_dml/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.celnet.config import ExperimentConfig, load_config_file
from src.celnet.data import cifar_dataset
from src.celnet.metrics import (
    accuracy, average_ensemble_accuracy, average_learner_accuracy,
    ensemble_accuracy, pairwise_disagreement_matrix,
)
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import TimeLogger, seed_everything, save_json

from experiments.diversity.run_diversity import (
    per_class_accuracy, collect_predictions, collect_softmax,
)


def train_dml_pair(m1, m2, train_loader, opt1, opt2, device,
                   temperature, alpha, sch1=None, sch2=None):
    """Both models train with CE(true labels) + KL(peer soft predictions)."""
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    m1.train(); m2.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # freeze peer logits (no gradient through the other model)
        with torch.no_grad():
            logits1 = m1(images)
            logits2 = m2(images)

        # m1: CE(true) + KL(m2 soft)
        out1 = m1(images)
        soft2 = F.softmax(logits2 / temperature, dim=1)
        loss1 = ((1 - alpha) * criterion_ce(out1, labels)
                 + alpha * temperature**2
                 * criterion_kl(F.log_softmax(out1 / temperature, dim=1), soft2))
        opt1.zero_grad(); loss1.backward(); opt1.step()

        # m2: CE(true) + KL(m1 soft)
        out2 = m2(images)
        soft1 = F.softmax(logits1 / temperature, dim=1)
        loss2 = ((1 - alpha) * criterion_ce(out2, labels)
                 + alpha * temperature**2
                 * criterion_kl(F.log_softmax(out2 / temperature, dim=1), soft1))
        opt2.zero_grad(); loss2.backward(); opt2.step()

    if sch1: sch1.step()
    if sch2: sch2.step()
    return m1, m2


def run(cfg: ExperimentConfig, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size, seed=cfg.run_seed,
        num_workers=cfg.num_workers, root=cfg.data_root,
    )

    model_class = get_model_class(cfg.model)
    n_models = cfg.num_models  # 9
    models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained,
                         cfg.num_classes, seed=cfg.model_seeds[i], device=device)
        for i in range(n_models)
    ]

    optimizers, schedulers = [], []
    for m in models:
        opt = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        s1 = max(1, int(0.55 * cfg.n_rounds))
        s2 = max(s1 + 1, int(0.80 * cfg.n_rounds))
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=cfg.gamma)
        optimizers.append(opt)
        schedulers.append(sch)

    models_with_idx = [(m, i) for i, m in enumerate(models)]

    disagreement_per_round   = torch.zeros(cfg.n_rounds + 1)
    ensemble_error_per_round = torch.zeros(cfg.n_rounds + 1)
    avg_ind_error_per_round  = torch.zeros(cfg.n_rounds + 1)
    diversity_per_round      = torch.zeros(cfg.n_rounds + 1)

    DISAGREE_INTERVAL = 5

    # round-0
    ens0 = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
    avg0 = average_learner_accuracy(models_with_idx, val_loader)
    _, _, dis0 = pairwise_disagreement_matrix(models_with_idx, val_loader, cfg.num_classes)
    ensemble_error_per_round[0] = 1 - ens0 / 100
    avg_ind_error_per_round[0]  = 1 - avg0 / 100
    diversity_per_round[0]      = avg_ind_error_per_round[0] - ensemble_error_per_round[0]
    disagreement_per_round[0]   = dis0
    last_dis = dis0

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} (DML) ===")
            # round-robin pairing: (0,1), (2,3), ... then (1,2), (3,4), ...
            offset = round_idx % 2
            pairs = []
            for i in range(offset, n_models - 1, 2):
                pairs.append((i, i + 1))

            for i, j in pairs:
                models[i], models[j] = train_dml_pair(
                    models[i], models[j], train_loader,
                    optimizers[i], optimizers[j], device,
                    cfg.temperature, cfg.alpha,
                    schedulers[i], schedulers[j],
                )
                models_with_idx[i] = (models[i], i)
                models_with_idx[j] = (models[j], j)

            ens = average_ensemble_accuracy(models_with_idx, val_loader, cfg.num_classes)
            avg = average_learner_accuracy(models_with_idx, val_loader)
            r = round_idx + 1
            if r % DISAGREE_INTERVAL == 0 or r == cfg.n_rounds:
                _, _, last_dis = pairwise_disagreement_matrix(
                    models_with_idx, val_loader, cfg.num_classes)
            ensemble_error_per_round[r] = 1 - ens / 100
            avg_ind_error_per_round[r]  = 1 - avg / 100
            diversity_per_round[r]      = avg_ind_error_per_round[r] - ensemble_error_per_round[r]
            disagreement_per_round[r]   = last_dis
            print(f"  ens_acc={ens:.2f}  avg_acc={avg:.2f}  disagreement={last_dis:.4f}"
                  f"  diversity={diversity_per_round[r]:.4f}")

        time.sleep(5)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(disagreement_per_round,   out_dir / "disagreement_per_round.pt")
    torch.save(ensemble_error_per_round, out_dir / "ensemble_error_per_round.pt")
    torch.save(avg_ind_error_per_round,  out_dir / "avg_ind_error_per_round.pt")
    torch.save(diversity_per_round,      out_dir / "diversity_per_round.pt")

    heatmap = torch.stack([per_class_accuracy(m, test_loader, cfg.num_classes, device)
                           for m in models])
    torch.save(heatmap, out_dir / "per_class_acc_heatmap.pt")

    preds = collect_predictions(models_with_idx, test_loader, device)
    labels_all = torch.cat([y for _, y in test_loader])
    torch.save((preds == labels_all.unsqueeze(0)).float(), out_dir / "correct_matrix.pt")
    torch.save(collect_softmax(models_with_idx, test_loader, device), out_dir / "softmax_vecs.pt")

    final_ens = ensemble_accuracy(models_with_idx, test_loader, mode="logprob")
    final_avg = average_learner_accuracy(models_with_idx, test_loader)
    _, _, final_dis = pairwise_disagreement_matrix(models_with_idx, test_loader, cfg.num_classes)
    final_div = (1 - final_avg / 100) - (1 - final_ens / 100)
    print(f"\nFinal — ens_acc={final_ens:.2f}  avg={final_avg:.2f}"
          f"  disagreement={final_dis:.4f}  diversity={final_div:.4f}")
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
