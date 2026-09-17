"""Analyze the mechanism runs (switch_peers_celnet_pairing.py --mechanism).

For oracle CE updates in an accuracy range, compares what the real CE epoch and the KD counterfactual (same starting
state) change, for models shaped by KD (celnet) and by CE only (all_ce_cf). See peer_mechanism.py for the metrics.
All prediction metrics are in % of the split; accuracy change = fixed_peer_agree + fixed_peer_disagree - broken.

Writes mechanism_events.csv to <root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"mwm_accdiff": "KD-shaped", "mwm_accdiff_all_ce_cf": "CE-shaped"}
SPLIT_METRICS = ["acc_change", "broken", "fixed_peer_agree", "fixed_peer_disagree", "changed", "changed_to_peer",
                 "agree_peer_change", "kl", "conf_change"]
ARM_METRICS = ["weight_change_backbone", "weight_change_head", "swap_new_backbone_old_head",
               "swap_old_backbone_new_head"]


def load(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/seed*/results.json")):
        r = json.loads(path.read_text())
        cond = LABELS.get(r["config"]["pairing_strategy"], r["config"]["pairing_strategy"])
        for e in r["updates"]:
            if "mech" not in e:
                continue
            m = e["mech"]
            for arm in ("ce", "kd"):
                row = {"cond": cond, "seed": r["config"]["run_seed"], "round": e["round"], "student": e["student"],
                       "pre_acc": e["pre_acc"], "arm": arm}
                for split in ("val", "train"):
                    row[f"{split}_acc_change"] = m[f"{arm}_{split}_acc_after"] - m[f"{split}_acc_before"]
                    row[f"{split}_agree_peer_change"] = m[f"{arm}_{split}_agree_peer_after"] - m[f"{split}_agree_peer_before"]
                    for k in ["broken", "fixed_peer_agree", "fixed_peer_disagree", "changed", "changed_to_peer", "kl",
                              "conf_change"]:
                        row[f"{split}_{k}"] = m[f"{arm}_{split}_{k}"]
                    for k in ["share_wrong_agree_peer", "share_wrong_disagree_peer", "agree_peer_before",
                              "acc_before"]:
                        row[f"{split}_{k}"] = m[f"{split}_{k}"]
                row["weight_change_backbone"] = m[f"{arm}_weight_change_backbone"]
                row["weight_change_head"] = m[f"{arm}_weight_change_head"]
                row["swap_new_backbone_old_head"] = m[f"{arm}_val_acc_new_backbone_old_head"] - m["val_acc_before"]
                row["swap_old_backbone_new_head"] = m[f"{arm}_val_acc_old_backbone_new_head"] - m["val_acc_before"]
                rows.append(row)
    if not rows:
        raise SystemExit(f"no mechanism records under {root}")
    return pd.DataFrame(rows)


def perm_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, n = np.r_[a, b], len(a)
    obs = abs(a.mean() - b.mean())
    hits = total = 0
    for comb in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(comb)] = True
        total += 1
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
    return hits / total


def report(ev: pd.DataFrame, title: str):
    print("\n" + "=" * 110)
    print(title)
    if ev.empty:
        print("  no events")
        return
    counts = ev[ev["arm"] == "ce"].groupby("cond").agg(n=("seed", "size"), seeds=("seed", "nunique"),
                                                       mean_pre_acc=("pre_acc", "mean"))
    print(counts.round(2).to_string())

    before_cols = [f"{s}_{k}" for s in ("val", "train") for k in
                   ("acc_before", "share_wrong_agree_peer", "share_wrong_disagree_peer", "agree_peer_before")]
    print("\n  [state before the update]")
    print("   " + ev[ev["arm"] == "ce"].groupby("cond")[before_cols].mean().T.round(3).to_string()
          .replace("\n", "\n   "))

    metrics = [f"{s}_{k}" for s in ("val", "train") for k in SPLIT_METRICS] + ARM_METRICS
    per_seed = ev.groupby(["cond", "seed", "arm"])[metrics].mean()
    table = []
    for metric in metrics:
        row = {"metric": metric}
        diffs = {}
        for cond in ("KD-shaped", "CE-shaped"):
            if cond not in per_seed.index.get_level_values(0):
                continue
            s = per_seed.loc[cond, metric].unstack("arm")
            row[f"{cond} CE"] = s["ce"].mean()
            row[f"{cond} KD"] = s["kd"].mean()
            diffs[cond] = (s["ce"] - s["kd"]).to_numpy()
            row[f"{cond} CE-KD"] = diffs[cond].mean()
        if len(diffs) == 2:
            row["difference"] = diffs["KD-shaped"].mean() - diffs["CE-shaped"].mean()
            row["p_seeds"] = perm_p(diffs["KD-shaped"], diffs["CE-shaped"])
        table.append(row)
    print("\n  [what the update changes; per-seed means, then averaged over seeds]")
    print("   CE / KD = the real CE epoch / the KD counterfactual; difference = (CE-KD, KD-shaped) - (CE-KD, CE-shaped);")
    print("   p_seeds = exact permutation test on the per-seed CE-KD values")
    print("   " + pd.DataFrame(table).round(3).to_string(index=False).replace("\n", "\n   "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/peers_mechanism")
    args = ap.parse_args()
    root = Path(args.root)
    df = load(root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mechanism_events.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    report(df[df["pre_acc"] >= 70], "ORACLE CE UPDATES FROM >= 70% VAL ACCURACY")
    report(df[(df["pre_acc"] >= 75) & (df["pre_acc"] < 80)],
           "ORACLE CE UPDATES FROM 75-80% (range where both conditions have many updates)")
    print(f"\nwrote {out / 'mechanism_events.csv'}")


if __name__ == "__main__":
    main()
