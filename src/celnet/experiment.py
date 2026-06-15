from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import cifar_dataset, split_validation_set_kfold
from .metrics import (
    accuracy,
    average_ensemble_accuracy,
    average_ensemble_confidence,
    average_learner_accuracy,
    ensemble_val_acc_no_oracle,
    get_models_no_oracle,
    pairwise_disagreement_matrix,
)
from .models import get_model_class, initialize_model
from .pairing import build_pairing_methods, make_fixed_friend_groups_by_id
from .plotting import save_gain_plots, save_per_model_plots
from .training import OptimizerRegistry, distill_ensemble_to_student, train_student_kd, train_student_oracle
from .utils import TimeLogger, calculate_std_dev, save_json, seed_everything


def run_experiment(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size,
        seed=cfg.run_seed,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
    )
    val_splits = split_validation_set_kfold(val_loader.dataset, 10, cfg.run_seed)

    model_class = get_model_class(cfg.model)
    assert cfg.num_models <= len(cfg.model_seeds), "Not enough seeds for requested num_models."

    init_models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes, seed=s, device=device)
        for s in cfg.model_seeds[: cfg.num_models]
    ]
    updated_models = [(model, idx) for idx, model in enumerate(init_models)]
    oracle_id = 0

    avg_val_acc_tensor = torch.zeros(cfg.n_rounds + 1, device=device)
    avg_val_conf_tensor = torch.zeros(cfg.n_rounds + 1, device=device)
    avg_learner_acc_tensor = torch.zeros(cfg.n_rounds + 1, device=device)
    oracle_gain_student = torch.zeros(cfg.n_rounds, dtype=torch.float32)
    oracle_gain_student_cum = torch.zeros(cfg.n_rounds + 1, dtype=torch.float32)
    non_oracle_gain_mean = torch.zeros(cfg.n_rounds, dtype=torch.float32)
    non_oracle_gain_mean_cum = torch.zeros(cfg.n_rounds + 1, dtype=torch.float32)

    per_model_acc = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)
    oracle_gain_per_model = torch.full((cfg.num_models, cfg.n_rounds), float("nan"), dtype=torch.float32)
    non_oracle_gain_per_model = torch.full((cfg.num_models, cfg.n_rounds), float("nan"), dtype=torch.float32)
    oracle_gain_per_model_cum = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)
    non_oracle_gain_per_model_cum = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)

    fixed_group_ids = None
    if cfg.pairing_strategy.startswith("friend"):
        fixed_group_ids = make_fixed_friend_groups_by_id(updated_models, group_size=cfg.friend_group_size, seed=cfg.run_seed)

    registry = OptimizerRegistry()

    with TimeLogger():
        avg_val_acc_tensor[0] = average_ensemble_accuracy(updated_models, val_loader, cfg.num_classes)
        avg_val_conf_tensor[0] = average_ensemble_confidence(updated_models, val_loader, cfg.num_classes)
        avg_learner_acc_tensor[0] = average_learner_accuracy(get_models_no_oracle(updated_models, oracle_id), val_loader)
        for model, idx in updated_models:
            per_model_acc[idx, 0] = accuracy(model, val_loader)

        for round_idx in range(cfg.n_rounds):
            pairing_methods = build_pairing_methods(
                updated_models=updated_models,
                val_loader=val_loader,
                val_splits=val_splits,
                num_classes=cfg.num_classes,
                degree=cfg.degree,
                run_seed=cfg.run_seed,
                batch_size=cfg.batch_size,
                round_idx=round_idx,
                fixed_group_ids=fixed_group_ids,
            )
            if cfg.pairing_strategy not in pairing_methods:
                raise ValueError(f"Invalid pairing strategy: {cfg.pairing_strategy}")

            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} ({cfg.pairing_strategy}) ===")
            _round_start_acc = ensemble_val_acc_no_oracle(updated_models, val_loader, cfg.num_classes, oracle_id)
            pairs = pairing_methods[cfg.pairing_strategy]()

            round_kd_total_gain = 0.0
            round_kd_student_count = 0

            for (m1, id1), (m2, id2) in pairs:
                if id1 == oracle_id or id2 == oracle_id:
                    student, student_id = (m2, id2) if id1 == oracle_id else (m1, id1)
                    stu_acc_before_oracle = accuracy(student, val_loader)
                    opt, sch = registry.get(
                        student_id, student, cfg.optimizer_type, cfg.learning_rate,
                        cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma,
                    )
                    print(f"  oracle-train: oracle -> student {student_id}")
                    student = train_student_oracle(student, train_loader, opt, device, sch)
                    for i, (_m, mid) in enumerate(updated_models):
                        if mid == student_id:
                            updated_models[i] = (student, student_id)
                            break
                    gain = accuracy(student, val_loader) - stu_acc_before_oracle
                    oracle_gain_student[round_idx] = gain
                    oracle_gain_student_cum[round_idx + 1] = oracle_gain_student_cum[round_idx] + gain
                    oracle_gain_per_model[student_id, round_idx] = gain
                    continue

                acc1 = float(accuracy(m1, val_loader))
                acc2 = float(accuracy(m2, val_loader))
                if acc2 > acc1:
                    teacher, teacher_id, teacher_acc = m2, id2, acc2
                    student, student_id, student_acc = m1, id1, acc1
                else:
                    teacher, teacher_id, teacher_acc = m1, id1, acc1
                    student, student_id, student_acc = m2, id2, acc2

                opt, sch = registry.get(
                    student_id, student, cfg.optimizer_type, cfg.learning_rate,
                    cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma,
                )
                print(f"  kd-train: teacher {teacher_id} ({teacher_acc:.2f}) -> student {student_id} ({student_acc:.2f})")
                student = train_student_kd(student, teacher, train_loader, opt, device, cfg.temperature, cfg.alpha, sch)
                student_acc_after = float(accuracy(student, val_loader))
                kd_gain = student_acc_after - student_acc
                non_oracle_gain_per_model[student_id, round_idx] = kd_gain
                round_kd_total_gain += kd_gain
                round_kd_student_count += 1
                for i, (_m, mid) in enumerate(updated_models):
                    if mid == student_id:
                        updated_models[i] = (student, student_id)
                        break

            models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
            avg_val_acc_tensor[round_idx + 1] = average_ensemble_accuracy(models_no_oracle, val_loader, cfg.num_classes)
            avg_val_conf_tensor[round_idx + 1] = average_ensemble_confidence(models_no_oracle, val_loader, cfg.num_classes)
            avg_learner_acc_tensor[round_idx + 1] = average_learner_accuracy(models_no_oracle, val_loader)
            round_kd_mean_gain = (round_kd_total_gain / round_kd_student_count) if round_kd_student_count > 0 else 0.0
            non_oracle_gain_mean[round_idx] = round_kd_mean_gain
            non_oracle_gain_mean_cum[round_idx + 1] = non_oracle_gain_mean_cum[round_idx] + round_kd_mean_gain

            for model, idx in updated_models:
                per_model_acc[idx, round_idx + 1] = accuracy(model, val_loader)

            for mid in range(cfg.num_models):
                og = oracle_gain_per_model[mid, round_idx]
                ng = non_oracle_gain_per_model[mid, round_idx]
                oracle_gain_per_model_cum[mid, round_idx + 1] = oracle_gain_per_model_cum[mid, round_idx] + (0.0 if torch.isnan(og) else og)
                non_oracle_gain_per_model_cum[mid, round_idx + 1] = non_oracle_gain_per_model_cum[mid, round_idx] + (0.0 if torch.isnan(ng) else ng)

    model_name = model_class.__name__.lower()
    fig1, fig2 = save_gain_plots(
        cfg.n_rounds,
        oracle_gain_student,
        non_oracle_gain_mean,
        oracle_gain_student_cum,
        non_oracle_gain_mean_cum,
        cfg.num_models,
        model_name,
        cfg.pairing_strategy,
        cfg.run_id,
        cfg.results_dir,
    )

    per_model_figs = save_per_model_plots(
        per_model_acc,
        oracle_gain_per_model,
        non_oracle_gain_per_model,
        oracle_gain_per_model_cum,
        non_oracle_gain_per_model_cum,
        cfg.num_models,
        model_name,
        cfg.pairing_strategy,
        cfg.run_id,
        cfg.results_dir,
    )

    results_dir = Path(cfg.results_dir)
    torch.save(oracle_gain_student_cum, results_dir / f"{cfg.num_models}_oracle_cum_tensor_{model_name}_{cfg.pairing_strategy}_r{cfg.run_id}.pt")
    torch.save(non_oracle_gain_mean_cum, results_dir / f"{cfg.num_models}_non_oracle_cum_tensor_{model_name}_{cfg.pairing_strategy}_r{cfg.run_id}.pt")
    torch.save(avg_learner_acc_tensor, results_dir / f"{cfg.num_models}_avg_learner_acc_tensor_{model_name}_{cfg.pairing_strategy}_r{cfg.run_id}.pt")

    models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
    avg_acc = average_ensemble_accuracy(models_no_oracle, test_loader, cfg.num_classes)
    avg_conf = average_ensemble_confidence(models_no_oracle, test_loader, cfg.num_classes)
    ids, d, mean_dis = pairwise_disagreement_matrix(models_no_oracle, val_loader, cfg.num_classes)

    print(f"avg_ens_acc = {avg_acc:.2f}")
    print(f"avg_ens_conf = {avg_conf:.2f}")
    print(f"Mean pairwise 0/1 disagreement: {mean_dis:.4f}")
    print("\n===== Distilling Ensemble -> Single Model =====")

    student_model = initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes, seed=999, device=device)
    student_model = distill_ensemble_to_student(
        ensemble_models=models_no_oracle,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        temperature=4.0,
        alpha=0.7,
        epochs=50,
        lr=0.01,
    )
    student_test_acc = accuracy(student_model, test_loader)
    print(f"\nDistilled Student Test Accuracy: {student_test_acc:.2f}")

    accuracies = []
    for model, idx in models_no_oracle:
        acc = accuracy(model, test_loader)
        print(f"M{idx}_acc= {acc:.2f}")
        accuracies.append(acc)

    std_dev = calculate_std_dev(accuracies)
    acc_var_test = float(np.var(accuracies))
    print(f"std_dev= {std_dev:.2f}")
    print(f"acc_var= {acc_var_test:.4f}")

    summary = {
        "config": cfg.to_dict(),
        "avg_ensemble_test_accuracy": avg_acc,
        "avg_ensemble_test_confidence": avg_conf,
        "mean_pairwise_disagreement": mean_dis,
        "distilled_student_test_accuracy": student_test_acc,
        "individual_model_accuracies": accuracies,
        "std_dev": std_dev,
        "variance": acc_var_test,
        "plots": [fig1, fig2, *per_model_figs],
    }
    save_json(summary, results_dir / f"summary_r{cfg.run_id}.json")
    return summary
