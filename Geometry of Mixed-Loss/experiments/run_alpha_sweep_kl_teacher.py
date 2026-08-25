"""
Exp 1 variant -- KL-Trained Teacher (self-distilled, not CE-trained).

Tests whether Exp 1's null result depends on the teacher having been
directly CE-trained. Here the teacher is a "Gen-1" ResNet-50 trained
purely via KL self-distillation from a CE-trained "Gen-0" ResNet-50 (the
same teacher checkpoint Exp 1 already produced and cached) -- Gen-1 never
sees a true label itself, only Gen-0's soft predictions. Unlike the
mutual-peer-co-training pilot (two random models reinforcing pure noise),
Gen-1 still inherits real classification information through distillation,
so the student's subsequent training should show genuine learning rather
than being stuck at chance.

Everything else matches run_alpha_sweep.py exactly: same student
architecture, same alpha sweep, same durations, same logging -- so
aggregate_alpha_sweep.py, analyze_ce_trajectory.py,
print_cosine_at_switch.py, and print_accuracy_checkpoints.py all work
unchanged, just pointed at this experiment's --results_dir.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/run_alpha_sweep_kl_teacher.py" \
        --alpha 0.9 --kl_rounds 100 --ce_rounds 60 --seed 42 \
        --out "Geometry of Mixed-Loss/results/alpha_sweep_kl_teacher/alpha0_9_seed42/" \
        --gen0_ckpt "Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed42.pt" \
        --gen1_ckpt "Geometry of Mixed-Loss/results/teachers_kl/teacher_resnet50_kl_seed42.pt"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.celnet.data import cifar_dataset
from src.celnet.utils import seed_everything, save_json, TimeLogger
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_alpha_sweep import (
    build_resnet18, _collect_probe, gradient_cosine, val_accuracy,
    train_kl_round, train_ce_round,
)

GRAD_LOG_INTERVAL = 5
PROBE_BATCH_SIZE = 1000


def train_self_distill_epoch(gen1, gen0, train_loader, opt, device, temperature):
    """Pure KL self-distillation: Gen-1 matches Gen-0's soft predictions.
    No CE term, no true labels -- Gen-1 never touches ground truth."""
    kl = nn.KLDivLoss(reduction="batchmean")
    gen1.train(); gen0.eval()
    for images, _ in train_loader:
        images = images.to(device)
        with torch.no_grad():
            gen0_logits = gen0(images)
        logits = gen1(images)
        loss = temperature**2 * kl(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(gen0_logits / temperature, dim=1))
        opt.zero_grad(); loss.backward(); opt.step()


def build_gen1_teacher(gen0, train_loader, device, temperature, epochs, seed):
    torch.manual_seed(seed + 7777)
    gen1 = tvm.resnet50(weights=None, num_classes=10).to(device)
    opt = torch.optim.SGD(gen1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    s1 = max(1, int(0.55 * epochs)); s2 = max(s1 + 1, int(0.80 * epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)
    for ep in range(epochs):
        train_self_distill_epoch(gen1, gen0, train_loader, opt, device, temperature)
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"  Gen-1 self-distill epoch {ep+1}/{epochs}")
    gen1.eval()
    for p in gen1.parameters(): p.requires_grad_(False)
    return gen1


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_BATCH_SIZE, device)

    # ---- Gen-0: existing CE-trained teacher from Exp 1 ----
    gen0_path = Path(args.gen0_ckpt)
    if not gen0_path.exists():
        raise FileNotFoundError(
            f"Gen-0 teacher checkpoint not found: {gen0_path}. "
            "Run Exp 1 (run_alpha_sweep.py) for this seed first -- it caches "
            "the CE-trained ResNet-50 teacher this script reuses as Gen-0.")
    gen0 = tvm.resnet50(weights=None, num_classes=10).to(device)
    gen0.load_state_dict(torch.load(gen0_path, map_location=device, weights_only=True))
    gen0.eval()
    for p in gen0.parameters(): p.requires_grad_(False)
    print(f"Loaded Gen-0 (CE-trained) teacher from {gen0_path}")

    # ---- Gen-1: self-distilled from Gen-0, pure KL, no true labels ----
    gen1_path = Path(args.gen1_ckpt) if args.gen1_ckpt else out_dir / "teacher_gen1.pt"
    gen1_path.parent.mkdir(parents=True, exist_ok=True)
    if gen1_path.exists():
        gen1 = tvm.resnet50(weights=None, num_classes=10).to(device)
        gen1.load_state_dict(torch.load(gen1_path, map_location=device, weights_only=True))
        gen1.eval()
        for p in gen1.parameters(): p.requires_grad_(False)
        print(f"Loaded Gen-1 (self-distilled) teacher from {gen1_path}")
    else:
        print(f"Training Gen-1 teacher via pure KL self-distillation from Gen-0 "
              f"({args.gen1_epochs} epochs)...")
        gen1 = build_gen1_teacher(gen0, train_loader, device, args.temperature,
                                   args.gen1_epochs, args.seed)
        torch.save(gen1.state_dict(), gen1_path)
        print(f"Gen-1 teacher saved to {gen1_path}")

    gen1_acc = val_accuracy(gen1, val_loader, device)
    gen0_acc = val_accuracy(gen0, val_loader, device)
    print(f"Gen-0 val acc: {gen0_acc:.2f}%   Gen-1 val acc: {gen1_acc:.2f}%")

    # ---- student: identical protocol to run_alpha_sweep.py, teacher = Gen-1 ----
    student = build_resnet18(args.seed, device)
    s1 = max(1, int(0.55 * args.kl_rounds)); s2 = max(s1 + 1, int(0.80 * args.kl_rounds))
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {
        "alpha": args.alpha, "seed": args.seed,
        "kl_rounds": args.kl_rounds, "ce_rounds": args.ce_rounds,
        "teacher_kind": "kl_self_distilled_gen1",
        "gen0_val_acc": gen0_acc, "gen1_val_acc": gen1_acc,
        "val_acc": [], "phase": [], "delta_acc": [],
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
    }

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== KL phase (teacher=Gen-1 self-distilled, alpha={args.alpha}) ===")

    with TimeLogger():
        for r in range(1, args.kl_rounds + 1):
            train_kl_round(student, gen1, train_loader, opt, device,
                            args.temperature, args.alpha, sch)
            acc = val_accuracy(student, val_loader, device)
            log["val_acc"].append(acc)
            log["phase"].append("kl")
            log["delta_acc"].append(acc - prev_acc)
            prev_acc = acc

            if r % GRAD_LOG_INTERVAL == 0 or r == args.kl_rounds:
                cos, gce_n, gkl_n = gradient_cosine(
                    student, gen1, probe_imgs, probe_labs, args.temperature, args.alpha)
                log["ce_kl_cosine"].append({"round": r, "cos": cos})
                log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
                print(f"  KL {r:3d}: val={acc:.2f}%  Delta={acc-prev_acc:+.2f}  cos={cos:.3f}")

    print(f"\n=== CE phase ===")
    s1_ce = max(1, int(0.55 * args.ce_rounds)); s2_ce = max(s1_ce + 1, int(0.80 * args.ce_rounds))
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
                student, gen1, probe_imgs, probe_labs, args.temperature, args.alpha)
            log["ce_kl_cosine"].append({"round": args.kl_rounds + r, "cos": cos})
            log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
            print(f"  CE {r:3d}: val={acc:.2f}%  Delta={acc-prev_acc:+.2f}  cos={cos:.3f}")

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
    parser.add_argument("--alpha",       type=float, required=True)
    parser.add_argument("--kl_rounds",   type=int,   default=100)
    parser.add_argument("--ce_rounds",   type=int,   default=60)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--data_root",   default="./data")
    parser.add_argument("--gen0_ckpt",   required=True,
                         help="path to the CE-trained ResNet-50 teacher Exp 1 already cached")
    parser.add_argument("--gen1_ckpt",   default="",
                         help="cache path for the self-distilled Gen-1 teacher (reused across alpha values)")
    parser.add_argument("--gen1_epochs", type=int, default=100)
    run(parser.parse_args())
