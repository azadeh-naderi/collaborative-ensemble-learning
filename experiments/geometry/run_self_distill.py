"""
Exp 4 — Self-Distillation Cascade.

Gen-0: train ResNet-18 with CE for KD_EPOCHS epochs.
Gen-k (k=1,2,3): train fresh ResNet-18 with KD from Gen-(k-1), α=0.9.
After each generation, reintroduce CE fine-tuning from that Gen-k checkpoint
for CE_EPOCHS epochs and record the accuracy trajectory (drop and recovery).

Usage:
    python experiments/geometry/run_self_distill.py \
        --n_gens 3 --kd_epochs 150 --ce_epochs 60 \
        --out results/geometry/self_distill/seed42/ \
        --seed 42
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.celnet.data import cifar_dataset
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import seed_everything, save_json, TimeLogger
from experiments.geometry.run_alpha_sweep import gradient_cosine, collect_probe

PROBE_N           = 1000
GRAD_LOG_INTERVAL = 5
SEEDS_PER_GEN     = [104729, 1299709, 15485863]   # different seeds per gen


@torch.no_grad()
def val_acc(model, loader, device):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return 100 * c / t


def train_ce(model, loader, opt, device, sch=None):
    model.train(); ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        l = ce(model(x), y); opt.zero_grad(); l.backward(); opt.step()
    if sch: sch.step()


def train_kd(student, teacher, loader, opt, device, temperature, alpha, sch=None):
    student.train(); teacher.eval()
    ce = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            tl = teacher(x); pseudo = tl.argmax(1)
        out = student(x)
        loss = (1-alpha)*ce(out, pseudo) + alpha*temperature**2*kl(
            F.log_softmax(out/temperature, 1), F.softmax(tl/temperature, 1))
        opt.zero_grad(); loss.backward(); opt.step()
    if sch: sch.step()


def make_opt_sch(model, n_epochs):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    s1 = max(1, int(0.55*n_epochs)); s2 = max(s1+1, int(0.80*n_epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)
    return opt, sch


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = collect_probe(val_loader, PROBE_N, device)

    cls = get_model_class("resnet")
    all_results = {}

    prev_gen = None   # teacher for next generation

    for gen in range(args.n_gens + 1):   # gen 0 = CE only
        gen_seed = SEEDS_PER_GEN[gen % len(SEEDS_PER_GEN)] + args.seed
        model = initialize_model(cls, 3, False, 10, seed=gen_seed, device=device)
        gen_log = {"gen": gen, "kd_acc": [], "ce_acc": [], "ce_delta": [], "cosine": []}

        if gen == 0:
            print(f"\n=== Gen-0: CE only ({args.kd_epochs} epochs) ===")
            opt, sch = make_opt_sch(model, args.kd_epochs)
            with TimeLogger():
                for ep in range(1, args.kd_epochs + 1):
                    train_ce(model, train_loader, opt, device, sch)
                    acc = val_acc(model, val_loader, device)
                    gen_log["kd_acc"].append(acc)
                    if ep % 20 == 0: print(f"  ep {ep}: {acc:.2f}%")
        else:
            print(f"\n=== Gen-{gen}: KD from Gen-{gen-1} ({args.kd_epochs} epochs, α={args.alpha}) ===")
            teacher = copy.deepcopy(prev_gen).eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            opt, sch = make_opt_sch(model, args.kd_epochs)
            with TimeLogger():
                for ep in range(1, args.kd_epochs + 1):
                    train_kd(model, teacher, train_loader, opt, device, args.temperature, args.alpha, sch)
                    acc = val_acc(model, val_loader, device)
                    gen_log["kd_acc"].append(acc)
                    if ep % GRAD_LOG_INTERVAL == 0:
                        cos, _, _ = gradient_cosine(model, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
                        gen_log["cosine"].append({"epoch": ep, "cos": cos})
                    if ep % 20 == 0: print(f"  ep {ep}: {acc:.2f}%")

        # save checkpoint for CE fine-tuning test
        ckpt_path = out_dir / f"gen{gen}_final.pt"
        torch.save(model.state_dict(), ckpt_path)

        # CE fine-tune phase from this gen's checkpoint
        print(f"  → CE fine-tune from Gen-{gen} checkpoint ({args.ce_epochs} epochs)")
        ft_model = initialize_model(cls, 3, False, 10, seed=gen_seed, device=device)
        ft_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        ft_opt, ft_sch = make_opt_sch(ft_model, args.ce_epochs)
        prev_acc = val_acc(ft_model, val_loader, device)

        for ep in range(1, args.ce_epochs + 1):
            train_ce(ft_model, train_loader, ft_opt, device, ft_sch)
            acc = val_acc(ft_model, val_loader, device)
            gen_log["ce_acc"].append(acc)
            gen_log["ce_delta"].append(acc - prev_acc)
            prev_acc = acc
            if ep % 5 == 0:
                print(f"    CE ep {ep}: val={acc:.2f}%  Δ={gen_log['ce_delta'][-1]:+.3f}%")

        ce_deltas = gen_log["ce_delta"]
        t_star = next((i+1 for i, d in enumerate(ce_deltas) if d < 0), None)
        gen_log["t_star"] = t_star
        gen_log["initial_ce_delta"] = ce_deltas[0] if ce_deltas else 0.0
        print(f"  Gen-{gen}: T*={t_star}  init_Δ={gen_log['initial_ce_delta']:+.3f}%")

        all_results[f"gen{gen}"] = gen_log
        prev_gen = model   # this gen becomes teacher for next

    save_json(all_results, out_dir / "results.json")
    print(f"\nAll generations saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gens",      type=int,   default=3)
    parser.add_argument("--kd_epochs",   type=int,   default=150)
    parser.add_argument("--ce_epochs",   type=int,   default=60)
    parser.add_argument("--alpha",       type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--data_root",   default="./data")
    run(parser.parse_args())
