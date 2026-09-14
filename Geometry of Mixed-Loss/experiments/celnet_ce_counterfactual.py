"""CEL-Net rounds with a counterfactual for every oracle (CE) update.

Runs the CEL-Net training loop (run_experiment in src/celnet/experiment.py, non-POM strategies) with CEL-Net's own
data, models, pairing, optimizer registry and train functions. The only additions:
  * before each oracle CE epoch, the student is copied (weights and optimizer state) and trained one KD epoch from its
    best peer instead (the most accurate other learner at the start of the round); that copy is then discarded,
  * val accuracy and val loss are logged before and after every update (CE and KD).
The final ensemble-to-student distillation and the plots of run_experiment are skipped.

Note: the counterfactual epoch consumes one pass of the training loader, so later data orders differ from a run
without it; the real updates are otherwise unchanged.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.celnet.config import ExperimentConfig, load_config_file
from src.celnet.data import cifar_dataset, noisy_loader, split_validation_set_kfold
from src.celnet.metrics import ensemble_accuracy
from src.celnet.models import get_model_class, initialize_model
from src.celnet.pairing import build_pairing_methods, make_fixed_friend_groups_by_id
from src.celnet.training import OptimizerRegistry, train_student_kd, train_student_oracle
from src.celnet.utils import TimeLogger, save_json, seed_everything

ORACLE_ID = 0


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total, loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, loss / total


def counterfactual_kd(student, opt, teacher, train_loader, val_loader, device, temperature, alpha):
    """One KD epoch from the student's current weights and optimizer state, on a copy; the real student is untouched."""
    twin = copy.deepcopy(student)
    twin_opt = type(opt)(twin.parameters(), **opt.defaults)
    # deepcopy: load_state_dict would otherwise share the momentum buffers with the real optimizer
    twin_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    train_student_kd(twin, teacher, train_loader, twin_opt, device, temperature, alpha, lr_scheduler=None)
    out = evaluate(twin, val_loader)
    del twin, twin_opt
    return out


def load_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.config:
        cfg = ExperimentConfig(**{**cfg.to_dict(), **load_config_file(args.config)})
    for key in ["pairing_strategy", "run_seed", "n_rounds", "num_models", "num_workers", "batch_size", "data_root"]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diversity_mwm_acc_220r.yaml")
    ap.add_argument("--pairing_strategy", default=None)
    ap.add_argument("--run_seed", type=int, default=None)
    ap.add_argument("--n_rounds", type=int, default=None)
    ap.add_argument("--num_models", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--no_counterfactual", action="store_true", help="run the plain CEL-Net rounds with logging only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args)
    if cfg.pairing_strategy == "POM":
        raise SystemExit("POM (symmetric mutual KD) is not supported here")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.run_seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size=cfg.batch_size, seed=cfg.run_seed, num_workers=cfg.num_workers, root=cfg.data_root)
    val_splits = split_validation_set_kfold(val_loader.dataset, 10, cfg.run_seed)
    oracle_train_loader, actual_noise_rate = train_loader, 0.0
    if cfg.label_noise_rate > 0:
        oracle_train_loader, actual_noise_rate = noisy_loader(
            train_loader, cfg.num_classes, cfg.label_noise_rate, cfg.noise_seed)

    model_class = get_model_class(cfg.model)
    assert cfg.num_models <= len(cfg.model_seeds), "Not enough seeds for requested num_models."
    updated_models = [
        (initialize_model(model_class, cfg.input_channels, cfg.pretrained, cfg.num_classes, seed=s, device=device), idx)
        for idx, s in enumerate(cfg.model_seeds[: cfg.num_models])]
    fixed_group_ids = None
    if cfg.pairing_strategy.lower().startswith("friend"):
        fixed_group_ids = make_fixed_friend_groups_by_id(updated_models, group_size=cfg.friend_group_size,
                                                         seed=cfg.run_seed)
    registry = OptimizerRegistry()

    def model_of(model_id):
        return next(m for m, idx in updated_models if idx == model_id)

    def replace(model, model_id):
        for i, (_m, idx) in enumerate(updated_models):
            if idx == model_id:
                updated_models[i] = (model, model_id)
                return

    updates, round_start_acc = [], []
    counterfactual = not args.no_counterfactual
    log = {"config": cfg.to_dict(), "actual_noise_rate": actual_noise_rate, "counterfactual": counterfactual,
           "counterfactual_teacher_rule": "most accurate other learner at the start of the round",
           "updates": updates, "round_start_acc": round_start_acc}

    with TimeLogger():
        for round_idx in range(cfg.n_rounds):
            pairing_methods = build_pairing_methods(
                updated_models=updated_models, val_loader=val_loader, val_splits=val_splits,
                num_classes=cfg.num_classes, degree=cfg.degree, run_seed=cfg.run_seed, batch_size=cfg.batch_size,
                round_idx=round_idx, fixed_group_ids=fixed_group_ids)
            if cfg.pairing_strategy not in pairing_methods:
                raise ValueError(f"Invalid pairing strategy: {cfg.pairing_strategy}")
            start = {idx: evaluate(m, val_loader) for m, idx in updated_models if idx != ORACLE_ID}
            round_start_acc.append({str(k): v[0] for k, v in start.items()})
            print(f"\n=== Round {round_idx + 1}/{cfg.n_rounds} ({cfg.pairing_strategy}) ===")
            pairs = pairing_methods[cfg.pairing_strategy]()

            for (m1, id1), (m2, id2) in pairs:
                if ORACLE_ID in (id1, id2):
                    student, student_id = (m2, id2) if id1 == ORACLE_ID else (m1, id1)
                    pre_acc, pre_loss = evaluate(student, val_loader)
                    opt, sch = registry.get(student_id, student, cfg.optimizer_type, cfg.learning_rate, cfg.momentum,
                                            cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma)
                    lr = opt.param_groups[0]["lr"]
                    rec = {"round": round_idx + 1, "student": student_id, "phase": "ce", "teacher": None,
                           "teacher_acc": None, "lr": lr, "pre_acc": pre_acc, "pre_loss": pre_loss}
                    if counterfactual:
                        cf_id = max((acc[0], idx) for idx, acc in start.items() if idx != student_id)[1]
                        cf_acc, cf_loss = counterfactual_kd(student, opt, model_of(cf_id), train_loader, val_loader,
                                                            device, cfg.temperature, cfg.alpha)
                        rec.update({"cf_teacher": cf_id, "cf_teacher_start_acc": start[cf_id][0],
                                    "cf_teacher_acc": evaluate(model_of(cf_id), val_loader)[0],
                                    "cf_acc": cf_acc, "cf_loss": cf_loss})
                    student = train_student_oracle(student, oracle_train_loader, opt, device, sch)
                    replace(student, student_id)
                    rec["post_acc"], rec["post_loss"] = evaluate(student, val_loader)
                    updates.append(rec)
                    cf_txt = (f", KD from peer {rec['cf_teacher']} instead -> {rec['cf_acc']:.2f} "
                              f"({rec['cf_acc'] - pre_acc:+.2f})") if counterfactual else ""
                    print(f"  oracle -> student {student_id} [lr {lr:g}]: {pre_acc:.2f} -> {rec['post_acc']:.2f} "
                          f"({rec['post_acc'] - pre_acc:+.2f}){cf_txt}")
                    continue

                (acc1, loss1), (acc2, loss2) = evaluate(m1, val_loader), evaluate(m2, val_loader)
                if acc2 > acc1:
                    teacher, teacher_id, teacher_acc = m2, id2, acc2
                    student, student_id, pre_acc, pre_loss = m1, id1, acc1, loss1
                else:
                    teacher, teacher_id, teacher_acc = m1, id1, acc1
                    student, student_id, pre_acc, pre_loss = m2, id2, acc2, loss2
                opt, sch = registry.get(student_id, student, cfg.optimizer_type, cfg.learning_rate, cfg.momentum,
                                        cfg.weight_decay, cfg.use_scheduler, cfg.n_rounds, cfg.gamma)
                lr = opt.param_groups[0]["lr"]
                student = train_student_kd(student, teacher, train_loader, opt, device, cfg.temperature, cfg.alpha, sch)
                replace(student, student_id)
                post_acc, post_loss = evaluate(student, val_loader)
                updates.append({"round": round_idx + 1, "student": student_id, "phase": "kd", "teacher": teacher_id,
                                "teacher_acc": teacher_acc, "lr": lr, "pre_acc": pre_acc, "pre_loss": pre_loss,
                                "post_acc": post_acc, "post_loss": post_loss})
                print(f"  kd: teacher {teacher_id} ({teacher_acc:.2f}) -> student {student_id} [lr {lr:g}]: "
                      f"{pre_acc:.2f} -> {post_acc:.2f} ({post_acc - pre_acc:+.2f})")

    learners = [(m, idx) for m, idx in updated_models if idx != ORACLE_ID]
    log["final_val_acc"] = {str(idx): evaluate(m, val_loader)[0] for m, idx in learners}
    log["final_test_acc"] = {str(idx): evaluate(m, test_loader)[0] for m, idx in learners}
    log["ensemble_test_acc_logprob"] = ensemble_accuracy(learners, test_loader, mode="logprob")
    save_json(log, out_dir / "results.json")
    torch.save({idx: m.state_dict() for m, idx in learners}, out_dir / "learners_final.pt")
    print(f"\nfinal test acc per learner {log['final_test_acc']}  ensemble {log['ensemble_test_acc_logprob']:.2f}")


if __name__ == "__main__":
    main()
