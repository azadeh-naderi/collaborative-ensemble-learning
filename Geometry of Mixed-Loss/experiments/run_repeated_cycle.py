"""
Repeated Short-Cycle Experiment -- tests whether loss basin incompatibility
requires *repeated* brief CE exposures without consolidation time, rather
than one long block (Exp 1's design).

Instead of 100 KL rounds then 60 CE rounds, this alternates in a repeating
cycle: K consecutive KL rounds, then 1 single CE round, repeated until a
fixed number of CE exposures (N_EXPOSURES) has occurred. K is exactly
t_eff from Corollary 1 (the number of consecutive KL rounds immediately
preceding a CE exposure) -- except here it happens repeatedly instead of
once, directly testing whether repetition compounds a per-exposure effect
that a single exposure does not show.

The number of CE exposures is held fixed across K values (so every
condition gets an equally reliable Oracle-gain-style estimate); the total
horizon scales as N_EXPOSURES * (K + 1) rounds.

Metrics mirror CEL-Net's own Oracle gain / Non-Oracle gain decomposition
exactly, computed within one continuous run instead of pooled across a
10-model cohort:
  - cumulative Oracle-gain-style sum: running total of Delta_acc on CE
    rounds only (the correct analog of CEL-Net's Oracle gain).
  - cumulative peer-gain-style sum: running total of Delta_acc on KL
    rounds only (the Non-Oracle-gain analog).
  - raw accuracy trajectory (per round), to check whether it stays smooth
    even if the Oracle-gain-style sum goes negative -- replicating the
    per-model-vs-Oracle-gain dissociation found in CEL-Net's own plots.
  - cos(g_CE, g_KL), logged at every CE round and periodically during KL
    rounds -- tests whether the angle progressively drifts toward/past
    zero as more cycles accumulate, unlike Exp 1's single-block design.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/run_repeated_cycle.py" \
        --k 20 --n_exposures 20 --alpha 0.9 --seed 42 \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/k20_seed42/" \
        --teacher_ckpt "Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed42.pt"
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
    build_resnet18, _collect_probe, gradient_cosine, val_accuracy,
    train_kl_round, train_ce_round, pretrain_teacher,
)

GRAD_LOG_INTERVAL_KL = 5   # log cosine every 5 KL rounds, in addition to every CE round
PROBE_BATCH_SIZE = 1000


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    total_rounds = args.n_exposures * (args.k + 1)
    cycle_len = args.k + 1  # K KL rounds + 1 CE round

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_BATCH_SIZE, device)

    # ---- teacher: reuse Exp 1's CE-pretrained ResNet-50 (same checkpoint) ----
    teacher_path = Path(args.teacher_ckpt)
    teacher = tvm.resnet50(weights=None, num_classes=10).to(device)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        print(f"Loaded CE-pretrained teacher from {teacher_path}")
    else:
        print(f"Teacher checkpoint not found at {teacher_path}; pretraining fresh (100 epochs)...")
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        pretrain_teacher(teacher, train_loader, device, epochs=100)
        torch.save(teacher.state_dict(), teacher_path)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # ---- student ----
    student = build_resnet18(args.seed, device)
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    s1 = max(1, int(0.55 * total_rounds)); s2 = max(s1 + 1, int(0.80 * total_rounds))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[s1, s2], gamma=0.1)

    log = {
        "k": args.k, "n_exposures": args.n_exposures, "total_rounds": total_rounds,
        "alpha": args.alpha, "seed": args.seed,
        "val_acc": [], "phase": [], "delta_acc": [],
        "oracle_gain_cumulative": [], "peer_gain_cumulative": [],
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
    }

    prev_acc = val_accuracy(student, val_loader, device)
    oracle_gain = 0.0
    peer_gain = 0.0
    n_ce_seen = 0

    print(f"\n=== Repeated short-cycle: K={args.k}, cycle_len={cycle_len}, "
          f"total_rounds={total_rounds}, alpha={args.alpha} ===")

    with TimeLogger():
        for r in range(1, total_rounds + 1):
            is_ce_round = (r % cycle_len == 0)

            if is_ce_round:
                train_ce_round(student, train_loader, opt, device, sch)
                n_ce_seen += 1
            else:
                train_kl_round(student, teacher, train_loader, opt, device,
                                args.temperature, args.alpha, sch)

            acc = val_accuracy(student, val_loader, device)
            delta = acc - prev_acc
            prev_acc = acc

            if is_ce_round:
                oracle_gain += delta
            else:
                peer_gain += delta

            log["val_acc"].append(acc)
            log["phase"].append("ce" if is_ce_round else "kl")
            log["delta_acc"].append(delta)
            log["oracle_gain_cumulative"].append(oracle_gain)
            log["peer_gain_cumulative"].append(peer_gain)

            if is_ce_round or r % GRAD_LOG_INTERVAL_KL == 0 or r == total_rounds:
                cos, gce_n, gkl_n = gradient_cosine(
                    student, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
                log["ce_kl_cosine"].append({"round": r, "cos": cos, "is_ce_round": is_ce_round})
                log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
                tag = "CE" if is_ce_round else "KL"
                print(f"  {tag} r{r:4d} (ce#{n_ce_seen:3d}): val={acc:.2f}%  Delta={delta:+.2f}  "
                      f"cos={cos:.3f}  oracle_gain={oracle_gain:+.2f}  peer_gain={peer_gain:+.2f}")

    log["final_oracle_gain"] = oracle_gain
    log["final_peer_gain"] = peer_gain

    # first CE-exposure index (not round) where cumulative oracle_gain goes
    # negative and stays negative for the remainder -- the "sustained
    # crossing" analog to CEL-Net's Fig 1 zero-crossing.
    ce_cum = [log["oracle_gain_cumulative"][i] for i, p in enumerate(log["phase"]) if p == "ce"]
    crossing = next((i + 1 for i in range(len(ce_cum)) if all(v < 0 for v in ce_cum[i:])), None)
    log["oracle_gain_sustained_crossing_exposure"] = crossing

    print(f"\nFinal Oracle-gain-style sum = {oracle_gain:+.2f}  "
          f"Final peer-gain-style sum = {peer_gain:+.2f}  "
          f"Sustained crossing at exposure #{crossing}")

    save_json(log, out_dir / "results.json")
    torch.save(student.state_dict(), out_dir / "student_final.pt")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",            type=int,   required=True,
                         help="consecutive KL rounds between each single CE round (= t_eff)")
    parser.add_argument("--n_exposures",  type=int,   default=20,
                         help="fixed number of CE exposures; total_rounds = n_exposures * (k+1)")
    parser.add_argument("--alpha",        type=float, default=0.9)
    parser.add_argument("--temperature",  type=float, default=4.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--data_root",    default="./data")
    parser.add_argument("--teacher_ckpt", required=True,
                         help="path to the CE-pretrained ResNet-50 teacher (reused from Exp 1)")
    run(parser.parse_args())
