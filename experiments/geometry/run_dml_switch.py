"""
Exp 3 — DML Loss-Regime Switch.

Train 2 ResNet-18 peers via DML for SWITCH_EPOCH epochs (α=args.alpha).
At epoch SWITCH_EPOCH, peer-1 switches to CE-only (α=0); peer-0 continues
DML. Measure per-epoch accuracy of both peers across the switch.

Usage:
    python experiments/geometry/run_dml_switch.py \
        --alpha 0.9 --switch_epoch 150 --total_epochs 210 \
        --out results/geometry/dml_switch/alpha0.9_seed42/ \
        --seed 42
"""
from __future__ import annotations

import argparse
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


@torch.no_grad()
def val_accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100 * correct / total


def train_dml_epoch(m0, m1, loader, opt0, opt1, device, temperature, alpha):
    """Standard DML: both models have CE(true labels) + KL(peer)."""
    ce = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    m0.train(); m1.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            l0 = m0(x); l1 = m1(x)
        # m0
        out0 = m0(x)
        loss0 = (1-alpha)*ce(out0,y) + alpha*temperature**2*kl(
            F.log_softmax(out0/temperature,1), F.softmax(l1/temperature,1))
        opt0.zero_grad(); loss0.backward(); opt0.step()
        # m1
        out1 = m1(x)
        loss1 = (1-alpha)*ce(out1,y) + alpha*temperature**2*kl(
            F.log_softmax(out1/temperature,1), F.softmax(l0/temperature,1))
        opt1.zero_grad(); loss1.backward(); opt1.step()


def train_ce_epoch(model, loader, opt, device):
    model.train(); ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = ce(model(x), y); opt.zero_grad(); loss.backward(); opt.step()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)

    cls = get_model_class("resnet")
    m0 = initialize_model(cls, 3, False, 10, seed=args.seed,       device=device)
    m1 = initialize_model(cls, 3, False, 10, seed=args.seed + 999,  device=device)

    s1 = max(1, int(0.55*args.total_epochs)); s2 = max(s1+1, int(0.80*args.total_epochs))
    opt0 = torch.optim.SGD(m0.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch0 = torch.optim.lr_scheduler.MultiStepLR(opt0, milestones=[s1, s2], gamma=0.1)
    sch1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[s1, s2], gamma=0.1)

    # build a dummy frozen teacher (m0 at init) just for gradient_cosine logging
    import copy
    teacher_proxy = copy.deepcopy(m1).eval()
    for p in teacher_proxy.parameters(): p.requires_grad_(False)
    probe_imgs, probe_labs = collect_probe(val_loader, PROBE_N, device)

    log = {"alpha": args.alpha, "switch_epoch": args.switch_epoch, "seed": args.seed,
           "m0_acc": [], "m1_acc": [], "m1_delta": [], "m1_cosine": []}

    prev_m1 = val_accuracy(m1, val_loader, device)

    with TimeLogger():
        for ep in range(1, args.total_epochs + 1):
            if ep <= args.switch_epoch:
                train_dml_epoch(m0, m1, train_loader, opt0, opt1, device, args.temperature, args.alpha)
                sch0.step(); sch1.step()
                phase = "dml"
            else:
                # m1 switches to CE-only; m0 continues DML (but alone — use m0 as own teacher proxy)
                train_dml_epoch(m0, m0, train_loader, opt0, opt0, device, args.temperature, args.alpha)  # m0 self-distils
                train_ce_epoch(m1, train_loader, opt1, device)
                sch0.step(); sch1.step()
                phase = "switch"

            a0 = val_accuracy(m0, val_loader, device)
            a1 = val_accuracy(m1, val_loader, device)
            log["m0_acc"].append(a0); log["m1_acc"].append(a1)
            log["m1_delta"].append(a1 - prev_m1); prev_m1 = a1

            if ep % GRAD_LOG_INTERVAL == 0 or ep == args.switch_epoch or ep == args.switch_epoch + 1:
                # use current m0 as teacher proxy for cosine measurement of m1
                for p_t, p_s in zip(teacher_proxy.parameters(), m0.parameters()):
                    p_t.data.copy_(p_s.data)
                cos, _, _ = gradient_cosine(m1, teacher_proxy, probe_imgs, probe_labs, args.temperature, args.alpha)
                log["m1_cosine"].append({"epoch": ep, "cos": cos, "phase": phase})
                print(f"  ep {ep:3d} [{phase:6s}]  m0={a0:.2f}  m1={a1:.2f}  Δm1={a1-prev_m1:+.3f}  cos={cos:.3f}")

    # find T* after switch
    post_switch_deltas = log["m1_delta"][args.switch_epoch:]
    t_star = next((i+1 for i,d in enumerate(post_switch_deltas) if d < 0), None)
    log["t_star_post_switch"] = t_star
    log["initial_switch_delta"] = post_switch_deltas[0] if post_switch_deltas else 0.0
    save_json(log, out_dir / "results.json")
    print(f"\nT*(post-switch)={t_star}  init_Δ={log['initial_switch_delta']:+.3f}%  saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",         type=float, default=0.9)
    parser.add_argument("--switch_epoch",  type=int,   default=150)
    parser.add_argument("--total_epochs",  type=int,   default=210)
    parser.add_argument("--temperature",   type=float, default=4.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--out",           required=True)
    parser.add_argument("--data_root",     default="./data")
    run(parser.parse_args())
