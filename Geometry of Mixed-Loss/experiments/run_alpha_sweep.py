"""
Exp 1 — α-Sweep: how KL weight controls loss basin incompatibility onset.

Train ResNet-18 student with pure KL distillation from a fixed ResNet-50
teacher for KL_ROUNDS rounds at varying α, then switch to CE-only for
CE_ROUNDS rounds. Log per-round val accuracy, CE/KL loss, and gradient
cosine similarity every GRAD_LOG_INTERVAL rounds.

Usage (from repo root):
    python experiments/geometry/run_alpha_sweep.py \
        --alpha 0.9 --kl_rounds 100 --ce_rounds 60 \
        --out results/geometry/alpha_sweep/alpha0.9_seed42/ \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.celnet.data import cifar_dataset
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import seed_everything, save_json, TimeLogger

GRAD_LOG_INTERVAL = 5
PROBE_BATCH_SIZE  = 1000


# ── model helpers ──────────────────────────────────────────────────────────────

def build_resnet18(seed, device, num_classes=10):
    cls = get_model_class("resnet")
    return initialize_model(cls, 3, False, num_classes, seed=seed, device=device)


def build_resnet50_teacher(device, num_classes=10):
    """Pretrain a ResNet-50 teacher with CE (100 epochs) and return it frozen."""
    import torchvision.models as tvm
    model = tvm.resnet50(weights=None, num_classes=num_classes).to(device)
    return model


def pretrain_teacher(teacher, train_loader, device, epochs=100):
    opt = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    teacher.train()
    for ep in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(teacher(images), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"  Teacher epoch {ep+1}/{epochs}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ── gradient cosine similarity ─────────────────────────────────────────────────

@torch.no_grad()
def _collect_probe(loader, n, device):
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x); labs.append(y)
        if sum(b.shape[0] for b in imgs) >= n:
            break
    return torch.cat(imgs)[:n].to(device), torch.cat(labs)[:n].to(device)


def gradient_cosine(model, teacher, probe_imgs, probe_labs, temperature, alpha):
    """Compute cos(g_CE, g_KL) on the probe batch."""
    model.train()
    # g_CE
    model.zero_grad()
    logits = model(probe_imgs)
    ce_loss = F.cross_entropy(logits, probe_labs)
    ce_loss.backward()
    g_ce = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])

    # g_KL
    model.zero_grad()
    logits = model(probe_imgs)
    with torch.no_grad():
        t_logits = teacher(probe_imgs)
    soft_t = F.softmax(t_logits / temperature, dim=1)
    kl_loss = temperature**2 * F.kl_div(
        F.log_softmax(logits / temperature, dim=1), soft_t, reduction="batchmean"
    )
    kl_loss.backward()
    g_kl = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])

    model.zero_grad()
    cos = F.cosine_similarity(g_ce.unsqueeze(0), g_kl.unsqueeze(0)).item()
    return cos, g_ce.norm().item(), g_kl.norm().item()


# ── training steps ─────────────────────────────────────────────────────────────

def train_kl_round(model, teacher, train_loader, opt, device, temperature, alpha, scheduler=None):
    """One round of KL-dominated distillation (no true labels)."""
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train(); teacher.eval()
    for images, _ in train_loader:
        images = images.to(device)
        with torch.no_grad():
            t_logits = teacher(images)
            pseudo   = t_logits.argmax(dim=1)
        logits = model(images)
        loss = (1 - alpha) * criterion_ce(logits, pseudo) + \
               alpha * temperature**2 * criterion_kl(
                   F.log_softmax(logits / temperature, dim=1),
                   F.softmax(t_logits / temperature, dim=1),
               )
        opt.zero_grad(); loss.backward(); opt.step()
    if scheduler:
        scheduler.step()


def train_ce_round(model, train_loader, opt, device, scheduler=None):
    """One round of pure CE on true labels."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        opt.zero_grad(); loss.backward(); opt.step()
    if scheduler:
        scheduler.step()


@torch.no_grad()
def val_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100 * correct / total


def compute_ce_loss(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
    return total_loss / total if total > 0 else float("nan")


# ── main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root,
    )

    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_BATCH_SIZE, device)

    # ── teacher ──
    teacher_path = Path(args.teacher_ckpt) if args.teacher_ckpt else out_dir / "teacher.pt"
    import torchvision.models as tvm
    teacher = tvm.resnet50(weights=None, num_classes=10).to(device)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"Loaded teacher from {teacher_path}")
    else:
        print("Pre-training teacher (ResNet-50) for 100 epochs …")
        pretrain_teacher(teacher, train_loader, device, epochs=100)
        torch.save(teacher.state_dict(), teacher_path)
        print(f"Teacher saved to {teacher_path}")

    # ── student ──
    student = build_resnet18(args.seed, device)
    s1 = max(1, int(0.55 * args.kl_rounds))
    s2 = max(s1+1, int(0.80 * args.kl_rounds))
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {
        "alpha": args.alpha, "seed": args.seed,
        "kl_rounds": args.kl_rounds, "ce_rounds": args.ce_rounds,
        "val_acc": [], "phase": [],        # "kl" or "ce"
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
        "delta_acc": [],                   # val_acc[t] - val_acc[t-1]
    }

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== KL phase (α={args.alpha}) ===")

    # ── KL phase ──
    with TimeLogger():
        for r in range(1, args.kl_rounds + 1):
            train_kl_round(student, teacher, train_loader, opt, device, args.temperature, args.alpha, sch)
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
                print(f"  KL round {r:3d}: val={acc:.2f}%  Δ={acc-prev_acc:+.2f}  cos={cos:.3f}")

    # reset scheduler for CE phase
    print(f"\n=== CE phase (α=0) ===")
    s1_ce = max(1, int(0.55 * args.ce_rounds))
    s2_ce = max(s1_ce+1, int(0.80 * args.ce_rounds))
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
            print(f"  CE round {r:3d}: val={acc:.2f}%  Δ={acc-prev_acc:+.2f}  cos={cos:.3f}")

    # find T* = first CE round where delta < 0
    ce_deltas = [d for d, p in zip(log["delta_acc"], log["phase"]) if p == "ce"]
    t_star = next((i+1 for i, d in enumerate(ce_deltas) if d < 0), None)
    initial_drop = ce_deltas[0] if ce_deltas else 0.0
    log["t_star"] = t_star
    log["initial_ce_delta"] = initial_drop
    print(f"\nT* = {t_star}  initial_ce_delta = {initial_drop:+.3f}%")

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
    parser.add_argument("--teacher_ckpt", default="")
    run(parser.parse_args())
