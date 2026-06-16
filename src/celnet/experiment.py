from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import cifar_dataset, noisy_loader, split_validation_set_kfold
from .metrics import (
    accuracy,
    average_ensemble_accuracy,
    average_ensemble_confidence,
    average_learner_accuracy,
    ensemble_accuracy,
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


    # Oracle train loader (may have label noise injected)
    oracle_train_loader = train_loader
    actual_noise_rate = 0.0
    if cfg.label_noise_rate > 0:
        oracle_train_loader, actual_noise_rate = noisy_loader(
            train_loader, cfg.num_classes, cfg.label_noise_rate, cfg.noise_seed
        )
        print(f"Oracle label noise: requested={cfg.label_noise_rate:.2f}, actual={actual_noise_rate:.4f}")

    model_class = get_model_class(cfg.model)
    assert cfg.num_models <= len(cfg.model_seeds), "Not enough seeds for requested num_models."

    init_models = [
        initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes, seed=s, device=device)
        for s in cfg.model_seeds[: cfg.num_models]
    ]
    updated_models = [(model, idx) for idx, model in enumerate(init_models)]
    oracle_id = 0

    # ── tracking tensors (names match monolithic script) ──────────────────────
    ens_acc_per_round       = torch.zeros(cfg.n_rounds + 1, device=device)
    avg_conf_per_round      = torch.zeros(cfg.n_rounds + 1, device=device)
    avg_learner_acc_per_round = torch.zeros(cfg.n_rounds + 1, device=device)

    oracle_gain_ens         = torch.zeros(cfg.n_rounds, dtype=torch.float32)
    oracle_gain_ens_cum     = torch.zeros(cfg.n_rounds + 1, dtype=torch.float32)
    non_oracle_gain_ens     = torch.zeros(cfg.n_rounds, dtype=torch.float32)
    non_oracle_gain_ens_cum = torch.zeros(cfg.n_rounds + 1, dtype=torch.float32)

    per_model_acc               = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)
    oracle_gain_per_model       = torch.full((cfg.num_models, cfg.n_rounds), float("nan"), dtype=torch.float32)
    non_oracle_gain_per_model   = torch.full((cfg.num_models, cfg.n_rounds), float("nan"), dtype=torch.float32)
    oracle_gain_per_model_cum   = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)
    non_oracle_gain_per_model_cum = torch.zeros((cfg.num_models, cfg.n_rounds + 1), dtype=torch.float32)
    # ─────────────────────────────────────────────────────────────────────────

    fixed_group_ids = None
    if cfg.pairing_strategy.startswith("Friend") or cfg.pairing_strategy.startswith("friend"):
        fixed_group_ids = make_fixed_friend_groups_by_id(updated_models, group_size=cfg.friend_group_size, seed=cfg.run_seed)

    registry = OptimizerRegistry()

    with TimeLogger():
        # round-0 baseline (all models, matching monolithic)
        models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
        for model, idx in updated_models:
            per_model_acc[idx, 0] = accuracy(model, val_loader)

        ens_acc_per_round[0]          = average_ensemble_accuracy(updated_models, val_loader, cfg.num_classes)
        avg_learner_acc_per_round[0]  = average_learner_accuracy(models_no_oracle, val_loader)
        avg_conf_per_round[0]         = average_ensemble_confidence(updated_models, val_loader, cfg.num_classes)

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

            round_start_acc = ensemble_val_acc_no_oracle(updated_models, val_loader, cfg.num_classes, oracle_id)
            oracle_phase_acc = round_start_acc

            pairs = pairing_methods[cfg.pairing_strategy]()

            round_kd_total_gain   = 0.0
            round_kd_student_count = 0
            oracle_gain_student_var = 0.0

            for (m1, id1), (m2, id2) in pairs:
                # Case A: oracle pair → supervised CE with (optionally noisy) true labels
                if id1 == oracle_id or id2 == oracle_id:
                    student, student_id = (m2, id2) if id1 == oracle_id else (m1, id1)

                    stu_acc_before_oracle = accuracy(student, val_loader)
                    opt, sch = registry.get(
                        student_id, student, cfg.optimizer_type, cfg.learning_rate,
                        cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma,
                    )
                    print(f"  oracle-train: oracle -> student {student_id} : {stu_acc_before_oracle:.2f}")

                    student = train_student_oracle(student, oracle_train_loader, opt, device, sch)

                    for i, (_m, mid) in enumerate(updated_models):
                        if mid == student_id:
                            updated_models[i] = (student, student_id)
                            break

                    stu_acc_after_oracle = accuracy(student, val_loader)
                    oracle_gain_student_var = stu_acc_after_oracle - stu_acc_before_oracle
                    oracle_gain_per_model[student_id, round_idx] = oracle_gain_student_var
                    print(f"  oracle-gain in this round: {oracle_gain_student_var:.2f}")
                    continue

                # Case B: peer-to-peer KD
                acc1 = float(accuracy(m1, val_loader))
                acc2 = float(accuracy(m2, val_loader))
                if acc2 > acc1:
                    teacher, teacher_id, teacher_acc = m2, id2, acc2
                    student, student_id, student_acc  = m1, id1, acc1
                else:
                    teacher, teacher_id, teacher_acc = m1, id1, acc1
                    student, student_id, student_acc  = m2, id2, acc2

                opt, sch = registry.get(
                    student_id, student, cfg.optimizer_type, cfg.learning_rate,
                    cfg.momentum, cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma,
                )
                print(f"  kd-train: teacher {teacher_id} ({teacher_acc:.2f}) -> student {student_id} ({student_acc:.2f})")

                student = train_student_kd(student, teacher, train_loader, opt, device, cfg.temperature, cfg.alpha, sch)
                student_acc_after = float(accuracy(student, val_loader))
                kd_gain = student_acc_after - student_acc

                non_oracle_gain_per_model[student_id, round_idx] = kd_gain
                round_kd_total_gain   += kd_gain
                round_kd_student_count += 1

                for i, (_m, mid) in enumerate(updated_models):
                    if mid == student_id:
                        updated_models[i] = (student, student_id)
                        break

            # end-of-round logging
            oracle_gain_ens[round_idx]         = oracle_gain_student_var
            oracle_gain_ens_cum[round_idx + 1] = oracle_gain_ens_cum[round_idx] + oracle_gain_student_var

            models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
            ens_acc_per_round[round_idx + 1]          = average_ensemble_accuracy(models_no_oracle, val_loader, cfg.num_classes)
            avg_learner_acc_per_round[round_idx + 1]  = average_learner_accuracy(models_no_oracle, val_loader)
            avg_conf_per_round[round_idx + 1]         = average_ensemble_confidence(models_no_oracle, val_loader, cfg.num_classes)

            round_kd_mean_gain = (round_kd_total_gain / round_kd_student_count) if round_kd_student_count > 0 else 0.0
            non_oracle_gain_ens[round_idx]         = round_kd_mean_gain
            non_oracle_gain_ens_cum[round_idx + 1] = non_oracle_gain_ens_cum[round_idx] + round_kd_mean_gain

            for model, idx in updated_models:
                per_model_acc[idx, round_idx + 1] = accuracy(model, val_loader)

            for mid in range(cfg.num_models):
                og = oracle_gain_per_model[mid, round_idx]
                ng = non_oracle_gain_per_model[mid, round_idx]
                oracle_gain_per_model_cum[mid, round_idx + 1]     = oracle_gain_per_model_cum[mid, round_idx]     + (0.0 if torch.isnan(og) else og)
                non_oracle_gain_per_model_cum[mid, round_idx + 1] = non_oracle_gain_per_model_cum[mid, round_idx] + (0.0 if torch.isnan(ng) else ng)

        # per-model gain summary (matching monolithic)
        oracle_total_per_model     = torch.nan_to_num(oracle_gain_per_model,     nan=0.0).sum(dim=1)
        non_oracle_total_per_model = torch.nan_to_num(non_oracle_gain_per_model, nan=0.0).sum(dim=1)
        for mid in range(cfg.num_models):
            print(
                f"M{mid}: oracle_total_gain={oracle_total_per_model[mid]:.2f}, "
                f"non_oracle_total_gain={non_oracle_total_per_model[mid]:.2f}"
            )

        time.sleep(5)

    # ── plots ─────────────────────────────────────────────────────────────────
    model_name = model_class.__name__.lower()
    fig1, fig2 = save_gain_plots(
        cfg.n_rounds,
        oracle_gain_ens,
        non_oracle_gain_ens,
        oracle_gain_ens_cum,
        non_oracle_gain_ens_cum,
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

    # ── save tensors (names match monolithic) ─────────────────────────────────
    results_dir = Path(cfg.results_dir)
    prefix = f"{cfg.num_models}_{model_name}_{cfg.pairing_strategy}_r{cfg.run_id}"
    torch.save(oracle_gain_ens,         results_dir / f"tensor_mean_oracle_{prefix}.pt")
    torch.save(non_oracle_gain_ens,     results_dir / f"tensor_mean_non_oracle_{prefix}.pt")
    torch.save(oracle_gain_ens_cum,     results_dir / f"tensor_cum_oracle_{prefix}.pt")
    torch.save(non_oracle_gain_ens_cum, results_dir / f"tensor_cum_non_oracle_{prefix}.pt")
    torch.save(ens_acc_per_round,       results_dir / f"tensor_ens_acc_{prefix}.pt")
    torch.save(avg_learner_acc_per_round, results_dir / f"tensor_avg_learner_acc_{prefix}.pt")

    # ── evaluation ────────────────────────────────────────────────────────────
    final_models_no_oracle = get_models_no_oracle(updated_models, oracle_id)

    print("\n===== Distilling Ensemble -> Single Model =====")
    student_model = initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes, seed=999, device=device)
    student_model = distill_ensemble_to_student(
        ensemble_models=final_models_no_oracle,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        temperature=4.0,
        alpha=0.9,
        epochs=100,
        lr=0.01,
    )
    student_test_acc = accuracy(student_model, test_loader)
    print(f"\nDistilled Student Test Accuracy: {student_test_acc:.2f}")

    ens_acc_logprob = ensemble_accuracy(final_models_no_oracle, test_loader, mode="logprob")
    ens_acc_prob    = ensemble_accuracy(final_models_no_oracle, test_loader, mode="prob")
    ens_acc_logit   = ensemble_accuracy(final_models_no_oracle, test_loader, mode="logit")
    print(f"Ensemble (logprob) Top-1 Accuracy: {ens_acc_logprob:.2f}")
    print(f"Ensemble (prob) Top-1 Accuracy:    {ens_acc_prob:.2f}")
    print(f"Ensemble (logit) Top-1 Accuracy:   {ens_acc_logit:.2f}")

    avg_acc  = average_ensemble_accuracy(final_models_no_oracle, test_loader, cfg.num_classes)
    avg_conf = average_ensemble_confidence(final_models_no_oracle, test_loader, cfg.num_classes)
    avg_learner_acc = average_learner_accuracy(final_models_no_oracle, test_loader)
    print(f"avg_ens_acc = {avg_acc:.2f}")
    print(f"avg_learner_acc = {avg_learner_acc:.2f}")
    print(f"avg_ens_conf = {avg_conf:.2f}")

    ids, d, mean_dis = pairwise_disagreement_matrix(final_models_no_oracle, test_loader, cfg.num_classes)
    print(f"Mean pairwise 0/1 disagreement: {mean_dis:.4f}")

    accuracies = []
    for model, idx in final_models_no_oracle:
        acc = accuracy(model, test_loader)
        print(f"M{idx}_acc= {acc:.2f}")
        accuracies.append(acc)

    std_dev     = calculate_std_dev(accuracies)
    acc_var_test = float(np.var(accuracies))
    print(f"std_dev= {std_dev:.2f}")
    print(f"acc_var= {acc_var_test:.4f}")

    summary = {
        "config": cfg.to_dict(),
        "actual_noise_rate": actual_noise_rate,
        "avg_ensemble_test_accuracy": avg_acc,
        "avg_ensemble_test_confidence": avg_conf,
        "avg_learner_test_accuracy": avg_learner_acc,
        "ensemble_accuracy_logprob": ens_acc_logprob,
        "ensemble_accuracy_prob": ens_acc_prob,
        "ensemble_accuracy_logit": ens_acc_logit,
        "mean_pairwise_disagreement": mean_dis,
        "distilled_student_test_accuracy": student_test_acc,
        "individual_model_accuracies": accuracies,
        "std_dev": std_dev,
        "variance": acc_var_test,
        "plots": [fig1, fig2, *per_model_figs],
    }
    save_json(summary, results_dir / f"summary_r{cfg.run_id}.json")
    return summary
