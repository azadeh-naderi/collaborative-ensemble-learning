"""Peer loss-switch experiment with CEL-Net-style pairing, in the controlled switch_exp1 setting.

N ResNet-18 learners plus an oracle (true labels). Each round the models are paired by accuracy as in CEL-Net
(the oracle counts as 100%): in each pair the more accurate model teaches and only the less accurate one trains, so
one learner gets a CE epoch on the true labels and the others get KD epochs from a better peer
(alpha*T^2*KL + (1-alpha)*CE on the teacher's predicted labels). Differences from CEL-Net: fixed train/val split,
official CIFAR-10 test set, constant LR (no scheduler).

Before each oracle CE update, a KD epoch from the best peer (most accurate other learner at the start of the round)
is measured from the same weights and optimizer state. With --all_ce every update is CE instead (same pairing and
schedule), as a control; the counterfactual is then measured on models shaped by CE only (--no_counterfactual to
skip it).

results.json uses the same schema as celnet_ce_counterfactual.py, so analyze_celnet_counterfactual.py reads it.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import random

from peer_mechanism import CE_VARIANTS, describe_updates, kd_twin, update, update_twin
from switch_common import build_loaders, metrics, predict
from switch_run_student import train_epoch
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
    ap.add_argument("--mechanism", action="store_true",
                    help="at each oracle CE update, log what the CE epoch and the KD counterfactual change "
                         "(see peer_mechanism.py); evaluation only")
    ap.add_argument("--oracle_update", choices=CE_VARIANTS, default="ce",
                    help="the CE update applied at oracle rounds: normal, final layer only, or at --low_lr")
    ap.add_argument("--ce_variants", action="store_true",
                    help="at each oracle update, also measure the other CE variants from the same state")
    ap.add_argument("--kd_variants", action="store_true",
                    help="at each oracle update, also measure KD on the final layer only and KD at --low_lr, "
                         "from the same state and with the same teacher as the KD counterfactual")
    ap.add_argument("--low_lr", type=float, default=0.01, help="learning rate of the *_lowlr variants")
    ap.add_argument("--kd_lowlr_per_round", action="store_true",
                    help="control: in every round, one randomly chosen KD student trains at --low_lr")
    ap.add_argument("--gentle_after", type=float, default=None,
                    help="schedule: from this validation accuracy (%%) on, oracle rounds apply --gentle_update "
                         "instead of --oracle_update")
    ap.add_argument("--gentle_update", choices=CE_VARIANTS, default="ce_head")
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
    if (args.mechanism or args.ce_variants or args.kd_variants) and args.no_counterfactual:
        raise SystemExit("--mechanism, --ce_variants and --kd_variants need the counterfactual")
    if args.all_ce and args.kd_lowlr_per_round:
        raise SystemExit("--kd_lowlr_per_round needs KD updates, not --all_ce")

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
    describe = args.mechanism or args.ce_variants or args.kd_variants
    scratch = copy.deepcopy(models[ids[0]]) if describe else None           # spare model for describe_updates
    # with --all_ce the counterfactual asks the control question: would a KD step have beaten CE for a model that was
    # shaped by CE only?
    counterfactual = not args.no_counterfactual
    condition = f"{args.pairing}{'_all_ce' if args.all_ce else ''}{'_cf' if args.all_ce and counterfactual else ''}"
    if args.oracle_update != "ce":
        condition += f"_oracle_{args.oracle_update}"
    if args.gentle_after is not None:
        condition += f"_{args.gentle_update}_from{args.gentle_after:g}"
    if args.kd_lowlr_per_round:
        condition += "_kd_lowlr_per_round"
    kd_branches = ["kd_head", "kd_lowlr"] if args.kd_variants else []
    lowlr_picker = random.Random(args.seed + 10_000)      # separate from torch's RNG, so data order is unaffected

    updates, round_start_acc, train_label_agreement = [], [], []
    log = {"config": {"pairing_strategy": condition, "pairing": args.pairing, "all_ce": args.all_ce,
                      "run_seed": args.seed, "n_rounds": args.rounds, "num_models": args.n_learners + 1,
                      "alpha": args.alpha, "temperature": args.temperature, "kd_hard_labels": "teacher_argmax",
                      "lr": args.lr, "lr_schedule": "constant", "split_seed": args.split_seed,
                      "mechanism": args.mechanism, "oracle_update": args.oracle_update,
                      "ce_variants": ([v for v in CE_VARIANTS if v != args.oracle_update]
                                      if args.ce_variants else []),
                      "kd_variants": kd_branches, "low_lr": args.low_lr,
                      "kd_lowlr_per_round": args.kd_lowlr_per_round,
                      "gentle_after": args.gentle_after,
                      "gentle_update": args.gentle_update if args.gentle_after is not None else None},
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
            lowlr_student = None
            if args.kd_lowlr_per_round:
                # the student of each KD pair, by the same rule as in the loop below
                kd_students = sorted(a if start[b][0] > start[a][0] else b
                                     for a, b in pairs if ORACLE_ID not in (a, b))
                lowlr_student = lowlr_picker.choice(kd_students) if kd_students else None

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
                if kind == "ce":
                    # how well the student fits the true labels on fixed training images, i.e. what CE will push on
                    rec["pre_train_acc"], rec["pre_train_loss"] = evaluate(models[s], probe_loader, device)
                twin = before = None
                branches = {}
                real = None
                if kind == "ce":
                    gentle = args.gentle_after is not None and pre_acc >= args.gentle_after
                    real = args.gentle_update if gentle else args.oracle_update
                if kind == "ce" and counterfactual:
                    cf_id = max((start[i][0], i) for i in ids if i != s)[1]
                    if describe:
                        before = copy.deepcopy(models[s].state_dict())
                    twin = kd_twin(models[s], opts[s], models[cf_id], train_loader, device, args.alpha,
                                   args.temperature)
                    cf_acc, cf_loss = evaluate(twin, val_loader, device)
                    rec.update({"cf_teacher": cf_id, "cf_teacher_start_acc": start[cf_id][0],
                                "cf_teacher_acc": start[cf_id][0], "cf_acc": cf_acc, "cf_loss": cf_loss})
                    ce_branches = [v for v in CE_VARIANTS if v != real] if args.ce_variants else []
                    for v in ce_branches + kd_branches:
                        branches[v] = update_twin(models[s], opts[s], train_loader, device, v, args.low_lr,
                                                  models[cf_id] if v.startswith("kd") else None,
                                                  args.alpha, args.temperature)
                        v_acc, v_loss = evaluate(branches[v], val_loader, device)
                        rec.setdefault("variants", {})[v] = {"acc": v_acc, "loss": v_loss}
                if kind == "ce":
                    rec["oracle_update"] = real
                    update(models[s], opts[s], train_loader, device, real, args.low_lr)
                elif kind == "kd" and s == lowlr_student:
                    rec["kd_lowlr"] = True
                    update(models[s], opts[s], train_loader, device, "kd_lowlr", args.low_lr, models[t],
                           args.alpha, args.temperature)
                else:
                    loss_type = "kd" if kind == "kd" else "ce"
                    train_epoch(models[s], models[t] if kind == "kd" else None, train_loader, opts[s], device,
                                loss_type, args.alpha, args.temperature)
                rec["post_acc"], rec["post_loss"] = evaluate(models[s], val_loader, device)
                if before is not None:
                    arms = {real: models[s], "kd": twin, **branches}
                    loaders = {"val": val_loader, "train": probe_loader} if args.mechanism else {"val": val_loader}
                    rec["mech"] = describe_updates(before, arms, models[cf_id], loaders, scratch, device)
                del twin, before, branches
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
