"""Peer loss-switch experiment with CEL-Net-style pairing, in the controlled switch_exp1 setting.

N ResNet-18 learners plus an oracle (true labels). Each round the models are paired by accuracy as in CEL-Net
(the oracle counts as 100%): in each pair the more accurate model teaches and only the less accurate one trains, so
one learner gets a CE epoch on the true labels and the others get KD epochs from a better peer
(alpha*T^2*KL + (1-alpha)*CE on the teacher's predicted labels). Differences from CEL-Net: fixed train/val split,
official CIFAR-10 test set, constant LR (no scheduler).

Before each oracle CE update, a KD epoch from the best peer (most accurate other learner at the start of the round)
is measured from the same weights and optimizer state. With --all_ce every update is CE instead (same pairing and
schedule), as a control.

results.json uses the same schema as celnet_ce_counterfactual.py, so analyze_celnet_counterfactual.py reads it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from switch_common import build_loaders, metrics, predict
from switch_run_student import branch_kd_epoch, train_epoch
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import TimeLogger, save_json, seed_everything

ORACLE_ID = 0


def evaluate(model, loader, device):
    m = metrics(*predict(model, loader, device))
    return m["acc"], m["loss"]


def pair_models(acc: dict, policy: str):
    """CEL-Net pairing on accuracies {id: acc}, oracle included with 100. Returns [(id_a, id_b), ...]."""
    ids = sorted(acc)                       # same node order as CEL-Net: oracle (0) first, then learners by id
    if policy == "accdiff":                 # src/celnet/pairing.py: max_difference_pairing
        order = sorted(ids, key=lambda i: acc[i])
        return [(order[k], order[len(order) - 1 - k]) for k in range(len(order) // 2)]
    if policy == "mwm_accdiff":             # src/celnet/pairing.py: maximum_weight_matching_accuracy_difference
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(len(ids)))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                g.add_edge(a, b, weight=abs(acc[ids[a]] - acc[ids[b]]))
        return [(ids[a], ids[b]) for a, b in nx.max_weight_matching(g, maxcardinality=True)]
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_learners", type=int, default=9, help="plus the oracle; an odd number gives full pairs")
    ap.add_argument("--rounds", type=int, default=220)
    ap.add_argument("--pairing", choices=["mwm_accdiff", "accdiff"], default="mwm_accdiff")
    ap.add_argument("--all_ce", action="store_true", help="control: every update is CE on the true labels")
    ap.add_argument("--no_counterfactual", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--probe_every", type=int, default=5,
                    help="every k rounds, log each learner's accuracy on 5000 fixed training images (0 = never)")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.n_learners < 2:
        raise SystemExit("--n_learners must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, probe_loader = build_loaders(
        args.seed, args.split_seed, args.batch_size, args.num_workers, args.data_root)

    ids = list(range(1, args.n_learners + 1))
    models = {i: initialize_model(get_model_class("resnet"), 3, False, 10, seed=args.seed * 100 + i, device=device)
              for i in ids}
    opts = {i: torch.optim.SGD(models[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) for i in ids}
    counterfactual = not args.no_counterfactual and not args.all_ce
    condition = f"{args.pairing}{'_all_ce' if args.all_ce else ''}"

    updates, round_start_acc, train_label_agreement = [], [], []
    log = {"config": {"pairing_strategy": condition, "pairing": args.pairing, "all_ce": args.all_ce,
                      "run_seed": args.seed, "n_rounds": args.rounds, "num_models": args.n_learners + 1,
                      "alpha": args.alpha, "temperature": args.temperature, "kd_hard_labels": "teacher_argmax",
                      "lr": args.lr, "lr_schedule": "constant", "split_seed": args.split_seed},
           "counterfactual": counterfactual,
           "counterfactual_teacher_rule": "most accurate other learner at the start of the round",
           "updates": updates, "round_start_acc": round_start_acc, "train_label_agreement": train_label_agreement}

    with TimeLogger():
        for rnd in range(1, args.rounds + 1):
            start = {i: evaluate(models[i], val_loader, device) for i in ids}
            round_start_acc.append({str(i): start[i][0] for i in ids})
            if args.probe_every and (rnd - 1) % args.probe_every == 0:
                train_label_agreement.append({"round": rnd, **{str(i): evaluate(models[i], probe_loader, device)[0]
                                                               for i in ids}})
            pairs = pair_models({ORACLE_ID: 100.0, **{i: start[i][0] for i in ids}}, args.pairing)
            print(f"\n=== Round {rnd}/{args.rounds} ({condition}) ===")

            for a, b in pairs:
                if ORACLE_ID in (a, b):
                    s = b if a == ORACLE_ID else a
                    kind, t = "ce", None
                else:
                    # pairs are disjoint, so start-of-round accuracies are still current; ties: first model teaches,
                    # as in CEL-Net (experiment.py: the second teaches only if strictly more accurate)
                    t, s = (b, a) if start[b][0] > start[a][0] else (a, b)
                    kind = "ce_peer" if args.all_ce else "kd"
                pre_acc, pre_loss = start[s]
                rec = {"round": rnd, "student": s, "phase": kind, "teacher": t,
                       "teacher_acc": None if t is None else start[t][0], "lr": opts[s].param_groups[0]["lr"],
                       "pre_acc": pre_acc, "pre_loss": pre_loss}
                if kind == "ce" and counterfactual:
                    cf_id = max((start[i][0], i) for i in ids if i != s)[1]
                    twin = branch_kd_epoch(models[s], opts[s], models[cf_id], train_loader, val_loader, device,
                                           args, None)
                    rec.update({"cf_teacher": cf_id, "cf_teacher_start_acc": start[cf_id][0],
                                "cf_teacher_acc": start[cf_id][0], "cf_acc": twin["acc"], "cf_loss": twin["loss"]})
                loss_type = "kd" if kind == "kd" else "ce"
                train_epoch(models[s], models[t] if kind == "kd" else None, train_loader, opts[s], device,
                            loss_type, args.alpha, args.temperature)
                rec["post_acc"], rec["post_loss"] = evaluate(models[s], val_loader, device)
                updates.append(rec)
                who = "oracle" if kind == "ce" else f"teacher {t} ({start[t][0]:.2f})"
                cf_txt = (f", KD from peer {rec['cf_teacher']} instead -> {rec['cf_acc']:.2f} "
                          f"({rec['cf_acc'] - pre_acc:+.2f})") if "cf_acc" in rec else ""
                print(f"  {kind}: {who} -> student {s}: {pre_acc:.2f} -> {rec['post_acc']:.2f} "
                      f"({rec['post_acc'] - pre_acc:+.2f}){cf_txt}")

    log["final_val_acc"] = {str(i): evaluate(models[i], val_loader, device)[0] for i in ids}
    test_logits, labels = {}, None
    for i in ids:
        test_logits[i], labels = predict(models[i], test_loader, device)
    log["final_test_acc"] = {str(i): metrics(test_logits[i], labels)["acc"] for i in ids}
    avg_logprob = torch.stack([test_logits[i].log_softmax(dim=1) for i in ids]).mean(dim=0)
    log["ensemble_test_acc_logprob"] = 100.0 * avg_logprob.argmax(dim=1).eq(labels).float().mean().item()
    save_json(log, out_dir / "results.json")
    torch.save({i: models[i].state_dict() for i in ids}, out_dir / "learners_final.pt")
    print(f"\nfinal test acc per learner {log['final_test_acc']}  ensemble {log['ensemble_test_acc_logprob']:.2f}")


if __name__ == "__main__":
    main()
