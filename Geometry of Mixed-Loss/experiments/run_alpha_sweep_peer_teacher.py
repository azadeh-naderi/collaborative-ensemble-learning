"""
Exp 1 variant -- Mutual Peer Co-Training (no CE-pretrained teacher).

Tests whether Exp 1's null result is an artifact of using a CE-pretrained,
fully-converged ResNet-50 teacher as the KL target. Here the "teacher" is
replaced by a second, randomly-initialized ResNet-18 that co-trains with the
tracked student via CEL-Net's actual peer-session loss (partner's hard
pseudo-label + KL against partner's soft prediction, both directions
simultaneously) -- neither model ever sees a true label during the peer
phase, exactly matching how CEL-Net peers never individually converge to
CE-optimality before being used as KL targets by others.

Protocol: 100 rounds of mutual peer training (student <-> partner, both
random init, zero ground-truth exposure for either), then the partner is
frozen and the student switches to CE-only on true labels for 60 rounds --
otherwise identical to run_alpha_sweep.py (same architecture, durations,
LR schedule, logging), so the existing aggregate_alpha_sweep.py,
analyze_ce_trajectory.py, plot_accuracy_trajectory.py, and
print_accuracy_checkpoints.py all work unchanged -- just point
--results_dir at this experiment's output folder.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/run_alpha_sweep_peer_teacher.py" \
        --alpha 0.9 --kl_rounds 100 --ce_rounds 60 --seed 42 \
        --out "Geometry of Mixed-Loss/results/alpha_sweep_peer_teacher/alpha0_9_seed42/"
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
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_alpha_sweep import _collect_probe, gradient_cosine, val_accuracy, train_ce_round

GRAD_LOG_INTERVAL = 5
PROBE_BATCH_SIZE = 1000


def build_resnet18(seed, device, num_classes=10):
    cls = get_model_class("resnet")
    return initialize_model(cls, 3, False, num_classes, seed=seed, device=device)


def train_mutual_peer_round(student, partner, train_loader, opt_s, opt_p, device,
                             temperature, alpha):
    """One round of CEL-Net-style mutual peer training. Both models train on
    (1-alpha)*CE(f, partner's hard prediction) + alpha*T^2*KL(f || partner's
    soft prediction), using simultaneous updates (each computed against a
    no-grad snapshot of the other taken at the top of the batch, matching
    run_dml_switch.py's convention). Neither model ever sees a true label
    here."""
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    student.train(); partner.train()
    for images, _ in train_loader:
        images = images.to(device)
        with torch.no_grad():
            s_out = student(images)
            p_out = partner(images)

        s_logits = student(images)
        loss_s = (1 - alpha) * criterion_ce(s_logits, p_out.argmax(dim=1)) + \
            alpha * temperature**2 * criterion_kl(
                F.log_softmax(s_logits / temperature, dim=1),
                F.softmax(p_out / temperature, dim=1))
        opt_s.zero_grad(); loss_s.backward(); opt_s.step()

        p_logits = partner(images)
        loss_p = (1 - alpha) * criterion_ce(p_logits, s_out.argmax(dim=1)) + \
            alpha * temperature**2 * criterion_kl(
                F.log_softmax(p_logits / temperature, dim=1),
                F.softmax(s_out / temperature, dim=1))
        opt_p.zero_grad(); loss_p.backward(); opt_p.step()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = cifar_dataset(
        batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = _collect_probe(val_loader, PROBE_BATCH_SIZE, device)

    student = build_resnet18(args.seed, device)
    partner = build_resnet18(args.seed + 999, device)

    s1 = max(1, int(0.55 * args.kl_rounds)); s2 = max(s1 + 1, int(0.80 * args.kl_rounds))
    opt_s = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    opt_p = torch.optim.SGD(partner.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sch_s = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[s1, s2], gamma=0.1)
    sch_p = torch.optim.lr_scheduler.MultiStepLR(opt_p, milestones=[s1, s2], gamma=0.1)

    log = {
        "alpha": args.alpha, "seed": args.seed,
        "kl_rounds": args.kl_rounds, "ce_rounds": args.ce_rounds,
        "teacher_kind": "mutual_peer_no_ce",
        "val_acc": [], "phase": [], "delta_acc": [],
        "ce_kl_cosine": [], "g_ce_norm": [], "g_kl_norm": [],
        "partner_val_acc": [],
    }

    prev_acc = val_accuracy(student, val_loader, device)
    print(f"\n=== Mutual peer phase (student<->partner, no CE for either, alpha={args.alpha}) ===")

    with TimeLogger():
        for r in range(1, args.kl_rounds + 1):
            train_mutual_peer_round(student, partner, train_loader, opt_s, opt_p, device,
                                     args.temperature, args.alpha)
            sch_s.step(); sch_p.step()
            acc = val_accuracy(student, val_loader, device)
            p_acc = val_accuracy(partner, val_loader, device)
            log["val_acc"].append(acc); log["partner_val_acc"].append(p_acc)
            log["phase"].append("kl"); log["delta_acc"].append(acc - prev_acc)
            prev_acc = acc

            if r % GRAD_LOG_INTERVAL == 0 or r == args.kl_rounds:
                cos, gce_n, gkl_n = gradient_cosine(student, partner, probe_imgs, probe_labs,
                                                     args.temperature, args.alpha)
                log["ce_kl_cosine"].append({"round": r, "cos": cos})
                log["g_ce_norm"].append(gce_n); log["g_kl_norm"].append(gkl_n)
                print(f"  peer {r:3d}: student={acc:.2f}%  partner={p_acc:.2f}%  "
                      f"Delta={acc-prev_acc:+.2f}  cos={cos:.3f}")

    # Freeze the partner exactly at its round-kl_rounds state -- mirrors
    # Exp 1's already-frozen CE teacher. Post-switch training never touches
    # the partner; it exists only so gradient_cosine has a fixed KL
    # direction to compare against, same as Exp 1.
    for p in partner.parameters():
        p.requires_grad_(False)
    partner.eval()

    print(f"\n=== CE phase (student only, true labels) ===")
    s1_ce = max(1, int(0.55 * args.ce_rounds)); s2_ce = max(s1_ce + 1, int(0.80 * args.ce_rounds))
    sch_ce = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[s1_ce, s2_ce], gamma=0.1)

    for r in range(1, args.ce_rounds + 1):
        train_ce_round(student, train_loader, opt_s, device, sch_ce)
        acc = val_accuracy(student, val_loader, device)
        log["val_acc"].append(acc); log["partner_val_acc"].append(log["partner_val_acc"][-1])
        log["phase"].append("ce"); log["delta_acc"].append(acc - prev_acc)
        prev_acc = acc

        if r % GRAD_LOG_INTERVAL == 0 or r == args.ce_rounds:
            cos, gce_n, gkl_n = gradient_cosine(student, partner, probe_imgs, probe_labs,
                                                 args.temperature, args.alpha)
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
    run(parser.parse_args())
