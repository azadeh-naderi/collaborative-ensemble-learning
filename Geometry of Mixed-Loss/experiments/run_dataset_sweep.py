"""
Exp 7 — Dataset & Class-Count Sweep.

Same α=0.9 KL-then-CE-switch protocol on three datasets:
  cifar10   (10 classes, 32×32)
  cifar100  (100 classes, 32×32)
  tinyimagenet10  (10-class subset of TinyImageNet, 64×64)

Teacher is always ResNet-50 pre-trained on the same dataset.
Student is always ResNet-18. Logs T*, initial Δ_CE, and gradient cosine.

TinyImageNet is expected at args.data_root/tiny-imagenet-200/ in the
standard layout (train/val directories from the official download).
The 10-class subset uses the first 10 wnids sorted alphabetically.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/run_dataset_sweep.py" \\
        --dataset cifar100 --kl_rounds 100 --ce_rounds 60 \\
        --out "Geometry of Mixed-Loss/results/dataset_sweep/cifar100_seed42/" \\
        --seed 42
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split

from src.celnet.utils import seed_everything, save_json, TimeLogger
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_alpha_sweep import (
    pretrain_teacher, gradient_cosine, _collect_probe, val_accuracy,
    train_kl_round, train_ce_round,
)

GRAD_LOG_INTERVAL = 5
PROBE_N = 1000


# ── data loaders ───────────────────────────────────────────────────────────────

def _make_loaders(dataset_name: str, data_root: str, batch_size: int,
                  seed: int, num_workers: int = 4):
    """Return (train_loader, val_loader, num_classes)."""
    root = Path(data_root)

    if dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
        ev_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        full_tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr_tf)
        full_ev = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=ev_tf)
        num_classes = 10

    elif dataset_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
        ev_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        full_tr = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tr_tf)
        full_ev = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=ev_tf)
        num_classes = 100

    elif dataset_name == "tinyimagenet10":
        tiny_root = root / "tiny-imagenet-200"
        if not tiny_root.exists():
            raise FileNotFoundError(
                f"TinyImageNet not found at {tiny_root}. "
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to that path."
            )
        mean, std = (0.480, 0.448, 0.398), (0.277, 0.269, 0.282)
        tr_tf = T.Compose([T.Resize(64), T.RandomCrop(64, padding=8),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
        ev_tf = T.Compose([T.Resize(64), T.ToTensor(), T.Normalize(mean, std)])

        full_tr_all = torchvision.datasets.ImageFolder(str(tiny_root / "train"), transform=tr_tf)
        full_ev_all = torchvision.datasets.ImageFolder(str(tiny_root / "train"), transform=ev_tf)

        # first 10 classes alphabetically
        all_classes = sorted(full_tr_all.classes)[:10]
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        kept = set(full_tr_all.class_to_idx[c] for c in all_classes)

        def filter_indices(ds):
            return [i for i, (_, label) in enumerate(ds.samples) if label in kept]

        tr_idx = filter_indices(full_tr_all)
        ev_idx = filter_indices(full_ev_all)

        # remap labels to 0..9
        class_remap = {full_tr_all.class_to_idx[c]: new for new, c in enumerate(all_classes)}

        class RemappedSubset(torch.utils.data.Dataset):
            def __init__(self, base, indices, remap):
                self.base = base; self.indices = indices; self.remap = remap
            def __len__(self): return len(self.indices)
            def __getitem__(self, i):
                x, y = self.base[self.indices[i]]
                return x, self.remap[y]

        full_tr = RemappedSubset(full_tr_all, tr_idx, class_remap)
        full_ev = RemappedSubset(full_ev_all, ev_idx, class_remap)
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    n = len(full_tr)
    train_n = int(0.9 * n)
    val_n = n - train_n
    g = torch.Generator().manual_seed(seed)
    tr_idx_split, val_idx_split = random_split(range(n), [train_n, val_n], generator=g)

    if dataset_name == "tinyimagenet10":
        # full_tr and full_ev are already RemappedSubset — build sub-subsets
        class IndexSubset(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base = base; self.indices = list(indices)
            def __len__(self): return len(self.indices)
            def __getitem__(self, i): return self.base[self.indices[i]]
        train_ds = IndexSubset(full_tr, tr_idx_split.indices)
        val_ds   = IndexSubset(full_ev, val_idx_split.indices)
    else:
        train_ds = Subset(full_tr, tr_idx_split.indices)
        val_ds   = Subset(full_ev, val_idx_split.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes


# ── main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes = _make_loaders(
        args.dataset, args.data_root, batch_size=64, seed=args.seed)
    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_N, device)

    # teacher (ResNet-50, trained on same dataset)
    teacher_path = Path(args.teacher_ckpt) if args.teacher_ckpt else out_dir / "teacher.pt"
    teacher = tvm.resnet50(weights=None, num_classes=num_classes).to(device)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"Loaded teacher from {teacher_path}")
    else:
        print(f"Pre-training ResNet-50 teacher on {args.dataset} for 100 epochs …")
        pretrain_teacher(teacher, train_loader, device, epochs=100)
        torch.save(teacher.state_dict(), teacher_path)
        print(f"Teacher saved to {teacher_path}")

    # student (ResNet-18)
    torch.manual_seed(args.seed)
    student = tvm.resnet18(weights=None, num_classes=num_classes).to(device)
    s1 = max(1, int(0.55 * args.kl_rounds))
    s2 = max(s1 + 1, int(0.80 * args.kl_rounds))
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {
        "dataset": args.dataset, "num_classes": num_classes,
        "alpha": args.alpha, "temperature": args.temperature,
        "kl_rounds": args.kl_rounds, "ce_rounds": args.ce_rounds, "seed": args.seed,
        "val_acc": [], "phase": [], "delta_acc": [],
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
    }

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== KL phase (dataset={args.dataset}, α={args.alpha}) ===")

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
    parser.add_argument("--dataset",     required=True,
                        choices=["cifar10", "cifar100", "tinyimagenet10"])
    parser.add_argument("--alpha",       type=float, default=0.9)
    parser.add_argument("--kl_rounds",   type=int,   default=100)
    parser.add_argument("--ce_rounds",   type=int,   default=60)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--data_root",   default="./data")
    parser.add_argument("--teacher_ckpt", default="")
    run(parser.parse_args())
