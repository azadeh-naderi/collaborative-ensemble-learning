"""Analyze the CE-variant runs (switch_peers_celnet_pairing.py --ce_variants, slurm_peers_ce_variants.sh).

Part 1, per update: at each oracle update of KD-shaped peers, every CE variant (normal, final layer only, low LR) and the
KD counterfactual start from the same state. Compares their val accuracy change, and what they break, fix and move.
Part 2, over the run: does applying a variant at oracle rounds change the peers' accuracy? Compared with the earlier
runs in results/peers_celnet_pairing (normal CE = celnet, and the all-CE control), which use the same setting.

Writes variant_events.csv to <root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["kd", "ce", "ce_head", "ce_lowlr"]


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    return float(np.mean([abs(np.dot(s, d) / len(d)) >= obs - 1e-12
                          for s in itertools.product([1, -1], repeat=len(d))]))


def perm_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, obs = np.r_[a, b], abs(a.mean() - b.mean())
    hits = total = 0
    for comb in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(comb)] = True
        total += 1
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
    return hits / total


def load_events(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/seed*/results.json")):
        r = json.loads(path.read_text())
        real = r["config"]["oracle_update"]
        for e in r["updates"]:
            if e["phase"] != "ce" or "cf_acc" not in e:
                continue
            acc = {"kd": e["cf_acc"], real: e["post_acc"], **{k: v["acc"] for k, v in e.get("variants", {}).items()}}
            m = e.get("mech", {})
            for arm, a in acc.items():
                row = {"condition": real, "seed": r["config"]["run_seed"], "round": e["round"], "pre_acc": e["pre_acc"],
                       "arm": arm, "d_acc": a - e["pre_acc"], "minus_kd": a - e["cf_acc"]}
                for k in ["broken", "fixed_peer_agree", "fixed_peer_disagree", "changed", "kl"]:
                    row[k] = m.get(f"{arm}_val_{k}")
                row["weight_change_backbone"] = m.get(f"{arm}_weight_change_backbone")
                row["weight_change_head"] = m.get(f"{arm}_weight_change_head")
                rows.append(row)
    return pd.DataFrame(rows)


def run_summary(root: Path, earlier: bool = False) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/seed*/results.json")):
        r = json.loads(path.read_text())
        cfg = r["config"]
        if earlier:
            name = {"mwm_accdiff": "normal CE (earlier runs)",
                    "mwm_accdiff_all_ce_cf": "all-CE control (earlier runs)"}.get(cfg["pairing_strategy"])
            if name is None:                           # e.g. all_ce seeds 0-2; the same control is in all_ce_cf
                continue
        else:
            name = f"oracle update: {cfg.get('oracle_update', 'ce')}"
        late = [np.mean(list(x.values())) for x in r["round_start_acc"][-40:]]
        rows.append({"condition": name, "seed": cfg["run_seed"],
                     "val_acc_last_40_rounds": float(np.mean(late)),
                     "learner_test": float(np.mean(list(r["final_test_acc"].values()))),
                     "ensemble_test": r["ensemble_test_acc_logprob"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/peers_ce_variants")
    ap.add_argument("--earlier", default="Geometry of Mixed-Loss/results/peers_celnet_pairing",
                    help="earlier runs in the same setting, used as references in part 2 (skipped if missing)")
    args = ap.parse_args()
    root = Path(args.root)
    ev = load_events(root)
    if ev.empty:
        raise SystemExit(f"no CE-variant records under {root}")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "variant_events.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    line = "=" * 110

    for lo, hi, title in [(70, 101, ">= 70%"), (75, 80, "75-80%")]:
        sub = ev[(ev["pre_acc"] >= lo) & (ev["pre_acc"] < hi)]
        print(line)
        print(f"PART 1. PER UPDATE, oracle updates from {title}: every arm starts from the same state.")
        print("  d_acc = val accuracy change; minus_kd = arm minus the KD counterfactual (> 0: better than KD);")
        print("  p = sign-flip test on the per-seed means of minus_kd; broken / fixed / changed in % of val images")
        for cond, c in sub.groupby("condition"):
            print(f"\n  [applied at oracle rounds: {cond}]   updates: {c[c['arm'] == 'kd'].shape[0]}")
            rows = []
            for arm in ARMS:
                a = c[c["arm"] == arm]
                if a.empty:
                    continue
                per_seed = a.groupby("seed")["minus_kd"].mean()
                rows.append({"arm": arm, "d_acc": a["d_acc"].mean(), "minus_kd": a["minus_kd"].mean(),
                             "better_than_kd": (a["minus_kd"] > 0).mean(), "worse_than_kd": (a["minus_kd"] < 0).mean(),
                             "seeds_better": f"{int((per_seed > 0).sum())}/{len(per_seed)}",
                             "p": sign_flip_p(per_seed) if arm != "kd" else np.nan,
                             "broken": a["broken"].mean(), "fixed": (a["fixed_peer_agree"] + a["fixed_peer_disagree"]).mean(),
                             "fixed_peer_agree": a["fixed_peer_agree"].mean(), "changed": a["changed"].mean(),
                             "kl": a["kl"].mean(), "backbone_change": a["weight_change_backbone"].mean(),
                             "head_change": a["weight_change_head"].mean()})
            print("   " + pd.DataFrame(rows).round(3).to_string(index=False).replace("\n", "\n   "))

    print("\n" + line)
    print("PART 2. OVER THE RUN: accuracy of the peers when the oracle rounds apply each update (mean over seeds)")
    runs = run_summary(root)
    earlier = Path(args.earlier)
    if earlier.exists():
        runs = pd.concat([runs, run_summary(earlier, earlier=True)], ignore_index=True)
    tab = runs.groupby("condition").agg(seeds=("seed", "nunique"), val_acc_last_40_rounds=("val_acc_last_40_rounds", "mean"),
                                        learner_test=("learner_test", "mean"), learner_test_sd=("learner_test", "std"),
                                        ensemble_test=("ensemble_test", "mean"))
    print("   " + tab.round(2).to_string().replace("\n", "\n   "))
    ref_name = "normal CE (earlier runs)"
    if ref_name in set(runs["condition"]):
        ref = runs[runs["condition"] == ref_name]
        print(f"\n   vs {ref_name}, per-seed learner test accuracy (exact permutation test):")
        for cond, c in runs.groupby("condition"):
            if cond == ref_name or len(c) + len(ref) > 20:
                continue
            print(f"   {cond:36s} {c['learner_test'].mean() - ref['learner_test'].mean():+.2f}  "
                  f"p {perm_p(c['learner_test'], ref['learner_test']):.4f}")
    print(f"\nwrote {out / 'variant_events.csv'}")


if __name__ == "__main__":
    main()
