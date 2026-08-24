"""
Exp 2 — KD Pre-Training Depth.

Trains ResNet-18 student with KD from a fixed ResNet-50 teacher for K epochs
(α=0.9), then fine-tunes with CE-only for 60 epochs. Measures per-epoch
accuracy drop at the phase switch and recovery trajectory.

Usage:
    python experiments/geometry/run_kd_finetune.py \
        --kd_epochs 100 --ce_epochs 60 \
        --out results/geometry/kd_finetune/k100_seed42/ \
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.celnet.data import cifar_dataset
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import seed_everything, save_json, TimeLogger
from experiments.geometry.run_alpha_sweep import (
    pretrain_teacher, gradient_cosine, collect_probe,
)

PROBE_N          = 1000
GRAD_LOG_INTERVAL = 5


@torch.no_grad()
def val_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100 * correct / total


def train_one_epoch_kd(model, teacher, loader, opt, device, temperature, alpha):
    ce = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    model.train(); teacher.eval()
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            tl = teacher(x); pseudo = tl.argmax(1)
        logits = model(x)
        loss = (1-alpha)*ce(logits, pseudo) + alpha*temperature**2*kl(
            F.log_softmax(logits/temperature, 1), F.softmax(tl/temperature, 1))
        opt.zero_grad(); loss.backward(); opt.step()


def train_one_epoch_ce(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = ce(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = collect_probe(val_loader, PROBE_N, device)

    # teacher
    teacher_path = Path(args.data_root).parent / "teacher_resnet50.pt"
    teacher = tvm.resnet50(weights=None, num_classes=10).to(device)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    else:
        print("Pre-training ResNet-50 teacher …")
        pretrain_teacher(teacher, train_loader, device, epochs=100)
        torch.save(teacher.state_dict(), teacher_path)

    # student
    cls = get_model_class("resnet")
    student = initialize_model(cls, 3, False, 10, seed=args.seed, device=device)
    s1 = max(1, int(0.55 * args.kd_epochs))
    s2 = max(s1+1, int(0.80 * args.kd_epochs))
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {"kd_epochs": args.kd_epochs, "ce_epochs": args.ce_epochs, "seed": args.seed,
           "alpha": args.alpha, "temperature": args.temperature,
           "val_acc": [], "phase": [], "delta_acc": [], "cosine": []}

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== KD phase ({args.kd_epochs} epochs, α={args.alpha}) ===")

    with TimeLogger():
        for ep in range(1, args.kd_epochs + 1):
            train_one_epoch_kd(student, teacher, train_loader, opt, device, args.temperature, args.alpha)
            sch.step()
            acc = val_accuracy(student, val_loader, device)
            log["val_acc"].append(acc); log["phase"].append("kd")
            log["delta_acc"].append(acc - prev_acc); prev_acc = acc
            if ep % GRAD_LOG_INTERVAL == 0 or ep == args.kd_epochs:
                cos, _, _ = gradient_cosine(student, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
                log["cosine"].append({"epoch": ep, "cos": cos})
                print(f"  KD ep {ep:3d}: val={acc:.2f}%  cos={cos:.3f}")
            # save checkpoint for post-hoc angle logging
            if ep % 10 == 0:
                torch.save(student.state_dict(), out_dir / f"student_round{ep:04d}.pt")

    print(f"\n=== CE fine-tune phase ({args.ce_epochs} epochs) ===")
    s1c = max(1, int(0.55 * args.ce_epochs))
    s2c = max(s1c+1, int(0.80 * args.ce_epochs))
    sch_ce = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1c, s2c], gamma=0.1)

    for ep in range(1, args.ce_epochs + 1):
        train_one_epoch_ce(student, train_loader, opt, device)
        sch_ce.step()
        acc = val_accuracy(student, val_loader, device)
        log["val_acc"].append(acc); log["phase"].append("ce")
        log["delta_acc"].append(acc - prev_acc); prev_acc = acc
        if ep % GRAD_LOG_INTERVAL == 0 or ep == args.ce_epochs:
            cos, _, _ = gradient_cosine(student, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
            log["cosine"].append({"epoch": args.kd_epochs + ep, "cos": cos})
            print(f"  CE ep {ep:3d}: val={acc:.2f}%  Δ={log['delta_acc'][-1]:+.3f}  cos={cos:.3f}")

    ce_deltas = [d for d, p in zip(log["delta_acc"], log["phase"]) if p == "ce"]
    log["t_star"] = next((i+1 for i, d in enumerate(ce_deltas) if d < 0), None)
    log["initial_ce_delta"] = ce_deltas[0] if ce_deltas else 0.0
    save_json(log, out_dir / "results.json")
    torch.save(student.state_dict(), out_dir / "student_final.pt")
    print(f"\nT*={log['t_star']}  init_Δ={log['initial_ce_delta']:+.3f}%  saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kd_epochs",   type=int,   default=100)
    parser.add_argument("--ce_epochs",   type=int,   default=60)
    parser.add_argument("--alpha",       type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--data_root",   default="./data")
    run(parser.parse_args())
