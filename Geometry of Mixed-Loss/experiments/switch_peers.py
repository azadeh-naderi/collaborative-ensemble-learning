"""Peer version of the loss-switch experiment.

A group of ResNet-18 peers, all trained from scratch at constant LR 0.1, teach each other with KD
(alpha*T^2*KL + (1-alpha)*CE on the teacher's predicted labels). Every k-th update of each peer is instead a CE
epoch on the true labels, as the oracle rounds in CEL-Net. Before each CE update, a KD epoch from the same weights
and optimizer state is also measured, so each CE update has an exact counterfactual.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from switch_common import build_loaders, ce_optimum_gap, metrics, predict
from switch_run_student import branch_kd_epoch, train_epoch
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import TimeLogger, seed_everything, save_json


def teacher_of(peer: int, rnd: int, n_peers: int) -> int:
    # cycles through every other peer, so each peer learns from all the others in turn
    return (peer + 1 + rnd % (n_peers - 1)) % n_peers


def is_ce(peer: int, rnd: int, ce_every: int) -> bool:
    # staggered across peers, so most rounds contain a CE update, as in CEL-Net
    return (rnd + peer) % ce_every == ce_every - 1


def frozen_copy(model):
    snap = copy.deepcopy(model).eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_peers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=80, help="every peer trains one epoch per round")
    ap.add_argument("--ce_every", type=int, default=5, help="every k-th update of a peer is CE on the true labels")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.1, help="constant LR, no decay")
    ap.add_argument("--branch_at_ce", action="store_true",
                    help="before each CE update, also measure a KD epoch from the same state")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n_peers < 2:
        raise SystemExit("--n_peers must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, probe_loader = build_loaders(
        args.seed, args.split_seed, args.batch_size, args.num_workers, args.data_root)

    n = args.n_peers
    peers = [initialize_model(get_model_class("resnet"), 3, False, 10, seed=args.seed * 100 + i, device=device)
             for i in range(n)]
    opts = [torch.optim.SGD(p.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) for p in peers]
    acc = [metrics(*predict(p, val_loader, device))["acc"] for p in peers]

    updates = []
    log = {
        "seed": args.seed, "split_seed": args.split_seed, "n_peers": n, "rounds": args.rounds,
        "ce_every": args.ce_every, "alpha": args.alpha, "temperature": args.temperature,
        "kd_hard_labels": "teacher_argmax", "lr": args.lr, "branch_at_ce": args.branch_at_ce,
        "val_init": list(acc), "group_mean_val_acc": [sum(acc) / n], "updates": updates,
    }

    with TimeLogger():
        for rnd in range(args.rounds):
            # teachers are the peers as they were at the start of the round
            snapshots = [frozen_copy(p) for p in peers]
            start_acc = list(acc)
            for i in range(n):
                ce = is_ce(i, rnd, args.ce_every)
                j = teacher_of(i, rnd, n)
                twin = None
                if ce and args.branch_at_ce:
                    twin = branch_kd_epoch(peers[i], opts[i], snapshots[j], train_loader, val_loader,
                                           device, args, None)
                train_loss = train_epoch(peers[i], None if ce else snapshots[j], train_loader, opts[i], device,
                                         "ce" if ce else "kd", args.alpha, args.temperature)
                val = metrics(*predict(peers[i], val_loader, device))
                probe = ce_optimum_gap(peers[i], probe_loader, device)
                acc[i] = val["acc"]
                updates.append({
                    "round": rnd + 1, "peer": i, "phase": "ce" if ce else "kd",
                    "teacher": None if ce else j, "teacher_pre_acc": None if ce else start_acc[j],
                    # for a CE update, the peer it would have learned from is what the counterfactual KD epoch uses
                    "counterfactual_teacher": j if twin else None,
                    "pre_acc": start_acc[i], "val_acc": val["acc"], "delta_acc": val["acc"] - start_acc[i],
                    "val_loss": val["loss"], "val_ece": val["ece"], "val_conf": val["conf"],
                    "train_loss": train_loss, "probe_acc": probe["acc"], "probe_ce_loss": probe["ce_loss"],
                    "branch_kd_val_acc": twin["acc"] if twin else None,
                    "branch_kd_val_loss": twin["loss"] if twin else None,
                })
                if twin:
                    print(f"round {rnd + 1:3d} peer {i} [ce] from {start_acc[i]:.2f}: CE -> {val['acc']:.2f} "
                          f"({val['acc'] - start_acc[i]:+.2f}), KD from peer {j} instead -> {twin['acc']:.2f} "
                          f"({twin['acc'] - start_acc[i]:+.2f})")
            del snapshots
            log["group_mean_val_acc"].append(sum(acc) / n)
            if (rnd + 1) % 10 == 0 or rnd + 1 == args.rounds:
                print(f"round {rnd + 1:3d}  peers val acc " + " ".join(f"{a:.2f}" for a in acc)
                      + f"  mean {sum(acc) / n:.2f}")

    test_logits, test_labels = [], None
    log["test_final"] = []
    for p in peers:
        logits, test_labels = predict(p, test_loader, device)
        test_logits.append(logits)
        log["test_final"].append(metrics(logits, test_labels))
    ens = torch.stack([l.softmax(dim=1) for l in test_logits]).mean(dim=0)
    log["test_ensemble_acc"] = 100.0 * ens.argmax(dim=1).eq(test_labels).float().mean().item()
    save_json(log, out_dir / "results.json")
    torch.save([p.state_dict() for p in peers], out_dir / "peers_final.pt")
    print(f"test acc per peer {[round(m['acc'], 2) for m in log['test_final']]}  "
          f"ensemble {log['test_ensemble_acc']:.2f}  saved to {out_dir}")


if __name__ == "__main__":
    main()
