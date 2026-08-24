"""
Exp 8 — Temperature Sweep.

Fixes α=0.9 and varies distillation temperature T ∈ {1, 2, 4, 8}.
Same KL-then-CE-switch protocol as Exp 1 (ResNet-18 student, ResNet-50 teacher,
CIFAR-10). Logs val accuracy, delta_acc, gradient cosine, and T*.

Hypothesis: higher T → softer targets → later T* (or no incompatibility);
lower T (hard targets) approximates CE pseudo-label training, also potentially
reducing incompatibility. T=4 (CEL-Net default) is expected to show the
strongest effect.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/run_temperature_sweep.py" \\
        --temperature 4.0 --kl_rounds 100 --ce_rounds 60 \\
        --out "Geometry of Mixed-Loss/results/temperature_sweep/T4_seed42/" \\
        --teacher_ckpt "Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed42.pt" \\
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.models as tvm

from src.celnet.data import cifar_dataset
from src.celnet.utils import seed_everything, save_json, TimeLogger
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_alpha_sweep import (
    pretrain_teacher, gradient_cosine, _collect_probe, val_accuracy,
    train_kl_round, train_ce_round,
)

GRAD_LOG_INTERVAL = 5
PROBE_N = 1000


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_N, device)

    # teacher (ResNet-50, shared across T values for same seed)
    teacher_path = Path(args.teacher_ckpt) if args.teacher_ckpt else out_dir / "teacher.pt"
    teacher = tvm.resnet50(weights=None, num_classes=10).to(device)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"Loaded teacher from {teacher_path}")
    else:
        print("Pre-training ResNet-50 teacher for 100 epochs …")
        pretrain_teacher(teacher, train_loader, device, epochs=100)
        torch.save(teacher.state_dict(), teacher_path)
        print(f"Teacher saved to {teacher_path}")

    # student (ResNet-18)
    torch.manual_seed(args.seed)
    student = tvm.resnet18(weights=None, num_classes=10).to(device)
    s1 = max(1, int(0.55 * args.kl_rounds))
    s2 = max(s1 + 1, int(0.80 * args.kl_rounds))
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {
        "temperature": args.temperature, "alpha": args.alpha,
        "kl_rounds": args.kl_rounds, "ce_rounds": args.ce_rounds, "seed": args.seed,
        "val_acc": [], "phase": [], "delta_acc": [],
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
    }

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== KL phase (T={args.temperature}, α={args.alpha}) ===")

    with TimeLogger():
        for r in range(1, args.kl_rounds + 1):
            train_kl_round(student, teacher, train_loader, opt, device,
                           args.temperature, args.alpha, sch)
            acc = val_accuracy(student, val_loader, device)
            log["val_acc"].append(acc)
            log["phase"].append("kl")
            log["delta_acc"].append(acc - prev_acc)
            prev_acc = acc

            if r % GRAD_LOG_INTERVAL == 0 or r == args.kl_rounds:
                cos, gce_n, gkl_n = gradient_cosine(
                    student, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
                log["ce_kl_cosine"].append({"round": r, "cos": cos})
                log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
                print(f"  KL {r:3d}: val={acc:.2f}%  Δ={acc-prev_acc:+.2f}  cos={cos:.3f}")

    print(f"\n=== CE phase ===")
    s1_ce = max(1, int(0.55 * args.ce_rounds))
    s2_ce = max(s1_ce + 1, int(0.80 * args.ce_rounds))
    sch_ce = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1_ce, s2_ce], gamma=0.1)

    for r in range(1, args.ce_rounds + 1):
        train_ce_round(student, train_loader, opt, device, sch_ce)
        acc = val_accuracy(student, val_loader, device)
        log["val_acc"].append(acc)
        log["phase"].append("ce")
        log["delta_acc"].append(acc - prev_acc)
        prev_acc = acc

        if r % GRAD_LOG_INTERVAL == 0 or r == args.ce_rounds:
            cos, gce_n, gkl_n = gradient_cosine(
                student, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
            log["ce_kl_cosine"].append({"round": args.kl_rounds + r, "cos": cos})
            log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
            print(f"  CE {r:3d}: val={acc:.2f}%  Δ={acc-prev_acc:+.2f}  cos={cos:.3f}")

    ce_deltas = [d for d, p in zip(log["delta_acc"], log["phase"]) if p == "ce"]
    t_star = next((i + 1 for i, d in enumerate(ce_deltas) if d < 0), None)
    log["t_star"] = t_star
    log["initial_ce_delta"] = ce_deltas[0] if ce_deltas else 0.0
    print(f"\nT* = {t_star}  initial_ce_delta = {log['initial_ce_delta']:+.3f}%")

    save_json(log, out_dir / "results.json")
    torch.save(student.state_dict(), out_dir / "student_final.pt")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True,
                        choices=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--alpha",       type=float, default=0.9)
    parser.add_argument("--kl_rounds",   type=int,   default=100)
    parser.add_argument("--ce_rounds",   type=int,   default=60)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--data_root",   default="./data")
    parser.add_argument("--teacher_ckpt", default="")
    run(parser.parse_args())
