from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .data import cifar_dataset, noisy_loader, split_validation_set_kfold
from .metrics import accuracy, average_learner_accuracy, ensemble_accuracy
from .models import get_model_class, initialize_model
from .utils import TimeLogger, calculate_std_dev, save_json, seed_everything


def run_independent_baseline(cfg: ExperimentConfig):
    """
    Independent baseline: N models each train with CE on noisy labels every round.
    No oracle, no peer KD — pure lower-bound reference for collaborative experiments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size,
        seed=cfg.run_seed,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
    )

    noisy_train_loader, actual_noise_rate = noisy_loader(
        train_loader, cfg.num_classes, cfg.label_noise_rate, cfg.noise_seed
    )
    print(f"Label noise: requested={cfg.label_noise_rate:.2f}, actual={actual_noise_rate:.4f}")

    model_class = get_model_class(cfg.model)
    n_models = cfg.num_models
    assert n_models <= len(cfg.model_seeds), "Not enough seeds for requested num_models."

    models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes,
                         seed=cfg.model_seeds[i], device=device)
        for i in range(n_models)
    ]

    # Each model trains ~(n_models//2)/n_models fraction of rounds
    n_train_per_round = n_models // 2
    expected_train_steps = int(cfg.n_rounds * n_train_per_round / n_models)

    optimizers = []
    schedulers = []
    for m in models:
        opt = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        s1 = max(1, int(0.55 * expected_train_steps))
        s2 = max(s1 + 1, int(0.80 * expected_train_steps))
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=cfg.gamma)
        optimizers.append(opt)
        schedulers.append(sch)

    criterion = nn.CrossEntropyLoss()

    per_model_acc   = torch.zeros((n_models, cfg.n_rounds + 1), dtype=torch.float32)
    ens_acc_per_round = torch.zeros(cfg.n_rounds + 1)
    avg_learner_acc_per_round = torch.zeros(cfg.n_rounds + 1)

    models_with_idx = [(m, i) for i, m in enumerate(models)]

    # round-0 baseline
    for i, m in enumerate(models):
        per_model_acc[i, 0] = accuracy(m, val_loader)
    ens_acc_per_round[0] = ensemble_accuracy(models_with_idx, val_loader, mode="logprob")
    avg_learner_acc_per_round[0] = average_learner_accuracy(models_with_idx, val_loader)

    rng = torch.Generator().manual_seed(cfg.run_seed)

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} (Independent) ===")

            # Match collaborative training frequency: in a 10-model run, 5 pairs form
            # per round but only the students (half) update. Mirror that here by training
            # floor(n_models / 2) randomly selected models each round.
            n_train = n_models // 2
            perm = torch.randperm(n_models, generator=rng).tolist()
            train_indices = perm[:n_train]

            for i in train_indices:
                m = models[i]
                m.train()
                for images, labels in noisy_train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizers[i].zero_grad()
                    loss = criterion(m(images), labels)
                    loss.backward()
                    optimizers[i].step()
                schedulers[i].step()

            for i, m in enumerate(models):
                per_model_acc[i, round_idx + 1] = accuracy(m, val_loader)
            print(f"  trained models: {sorted(train_indices)}")

            ens_acc_per_round[round_idx + 1] = ensemble_accuracy(models_with_idx, val_loader, mode="logprob")
            avg_learner_acc_per_round[round_idx + 1] = average_learner_accuracy(models_with_idx, val_loader)
            print(f"  ens_acc={ens_acc_per_round[round_idx + 1]:.2f}  avg_learner={avg_learner_acc_per_round[round_idx + 1]:.2f}")

        time.sleep(5)

    results_dir = Path(cfg.results_dir)
    model_name = model_class.__name__.lower()
    prefix = f"{n_models}_{model_name}_independent_r{cfg.run_id}"

    torch.save(ens_acc_per_round,         results_dir / f"tensor_ens_acc_{prefix}.pt")
    torch.save(avg_learner_acc_per_round, results_dir / f"tensor_avg_learner_acc_{prefix}.pt")
    torch.save(per_model_acc,             results_dir / f"tensor_per_model_acc_{prefix}.pt")

    # final evaluation on test set
    ens_acc_logprob = ensemble_accuracy(models_with_idx, test_loader, mode="logprob")
    ens_acc_prob    = ensemble_accuracy(models_with_idx, test_loader, mode="prob")
    ens_acc_logit   = ensemble_accuracy(models_with_idx, test_loader, mode="logit")
    avg_learner_test = average_learner_accuracy(models_with_idx, test_loader)
    print(f"\nEnsemble (logprob) Test Accuracy: {ens_acc_logprob:.2f}")
    print(f"Ensemble (prob)    Test Accuracy: {ens_acc_prob:.2f}")
    print(f"Ensemble (logit)   Test Accuracy: {ens_acc_logit:.2f}")
    print(f"Avg learner test accuracy: {avg_learner_test:.2f}")

    accuracies = []
    for i, m in enumerate(models):
        acc = accuracy(m, test_loader)
        print(f"M{i}_acc= {acc:.2f}")
        accuracies.append(acc)

    std_dev = calculate_std_dev(accuracies)
    acc_var = float(np.var(accuracies))
    print(f"std_dev= {std_dev:.2f}")
    print(f"acc_var= {acc_var:.4f}")

    summary = {
        "config": cfg.to_dict(),
        "actual_noise_rate": actual_noise_rate,
        "ensemble_accuracy_logprob": ens_acc_logprob,
        "ensemble_accuracy_prob": ens_acc_prob,
        "ensemble_accuracy_logit": ens_acc_logit,
        "avg_learner_test_accuracy": avg_learner_test,
        "individual_model_accuracies": accuracies,
        "std_dev": std_dev,
        "variance": acc_var,
    }
    save_json(summary, results_dir / f"summary_r{cfg.run_id}.json")
    return summary
