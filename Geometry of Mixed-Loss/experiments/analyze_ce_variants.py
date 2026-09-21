"""Analyze the label-update variant runs (slurm_peers_ce_variants.sh and slurm_peers_lr_controls.sh).

Part 1, per update: at each oracle update, every measured arm starts from the same state: the KD counterfactual (kd),
the update that was applied, and the other measured variants (ce, ce_head, ce_lowlr, kd_head, kd_lowlr).
Part 2, controls: paired contrasts from the same state that separate label-specific effects from learning-rate and
update-size effects: ce_lowlr - kd_lowlr and ce_head - kd_head.
Part 3, over the run: final accuracy of the peers under each condition, with the earlier runs in the same setting
(results/peers_celnet_pairing) as references.

Writes variant_events.csv to <first root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["kd", "ce", "ce_head", "ce_lowlr", "kd_head", "kd_lowlr"]
UPDATE_NAMES = {"ce": "normal CE", "ce_head": "CE on the final layer only", "ce_lowlr": "CE at LR 0.01"}
CONTRASTS = [("ce_lowlr", "kd_lowlr"), ("ce_head", "kd_head"), ("ce", "kd")]


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    return float(np.mean([abs(np.dot(s, d) / len(d)) >= obs - 1e-12
                          for s in itertools.product([1, -1], repeat=len(d))]))


def perm_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) + len(b) > 22:
        return float("nan")
    pooled, obs = np.r_[a, b], abs(a.mean() - b.mean())
    hits = total = 0
    for comb in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(comb)] = True
        total += 1
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
    return hits / total


def label(cfg: dict, earlier: bool = False) -> str:
    base = "all-CE" if cfg.get("all_ce") else "KD-shaped"
    name = f"{base}: {UPDATE_NAMES[cfg.get('oracle_update', 'ce')]}"
    if cfg.get("gentle_after") is not None:
        name = (f"{base}: normal CE, then {UPDATE_NAMES[cfg['gentle_update']]} from {cfg['gentle_after']:g}%")
    if cfg.get("kd_lowlr_per_round"):
        name += " + one KD update per round at LR 0.01"
    return name + (" (earlier runs)" if earlier else "")


def results(root: Path):
    for path in sorted(root.glob("*/seed*/results.json")):
        yield json.loads(path.read_text())


def load_events(roots) -> pd.DataFrame:
    rows = []
    for root in roots:
        for r in results(root):
            cfg = r["config"]
            for e in r["updates"]:
                if e["phase"] != "ce" or "cf_acc" not in e:
                    continue
                real = e.get("oracle_update", cfg.get("oracle_update", "ce"))
                acc = {"kd": e["cf_acc"], real: e["post_acc"],
                       **{k: v["acc"] for k, v in e.get("variants", {}).items()}}
                m = e.get("mech", {})
                for arm, a in acc.items():
                    row = {"condition": label(cfg), "seed": cfg["run_seed"], "round": e["round"],
                           "student": e["student"], "pre_acc": e["pre_acc"], "arm": arm, "applied": arm == real,
                           "d_acc": a - e["pre_acc"], "minus_kd": a - e["cf_acc"]}
                    for k in ["broken", "fixed_peer_agree", "fixed_peer_disagree", "changed", "kl"]:
                        row[k] = m.get(f"{arm}_val_{k}")
                    row["weight_change_backbone"] = m.get(f"{arm}_weight_change_backbone")
                    row["weight_change_head"] = m.get(f"{arm}_weight_change_head")
                    rows.append(row)
    return pd.DataFrame(rows)


def run_summary(root: Path, earlier: bool = False) -> pd.DataFrame:
    rows = []
    for r in results(root):
        cfg = r["config"]
        if earlier and cfg["pairing_strategy"] not in ("mwm_accdiff", "mwm_accdiff_all_ce_cf"):
            continue                                   # e.g. all_ce seeds 0-2; the same control is in all_ce_cf
        late = [np.mean(list(x.values())) for x in r["round_start_acc"][-40:]]
        rows.append({"condition": label(cfg, earlier), "seed": cfg["run_seed"],
                     "val_acc_last_40_rounds": float(np.mean(late)),
                     "learner_test": float(np.mean(list(r["final_test_acc"].values()))),
                     "ensemble_test": r["ensemble_test_acc_logprob"]})
    return pd.DataFrame(rows)


def per_update_table(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        a = sub[sub["arm"] == arm]
        if a.empty:
            continue
        per_seed = a.groupby("seed")["minus_kd"].mean()
        rows.append({"arm": arm + (" (applied)" if a["applied"].all() else ""), "n": len(a),
                     "d_acc": a["d_acc"].mean(), "minus_kd": a["minus_kd"].mean(),
                     "better_than_kd": (a["minus_kd"] > 0).mean(),
                     "seeds_better": f"{int((per_seed > 0).sum())}/{len(per_seed)}",
                     "p": sign_flip_p(per_seed) if arm != "kd" else np.nan,
                     "broken": a["broken"].mean(), "fixed": (a["fixed_peer_agree"] + a["fixed_peer_disagree"]).mean(),
                     "changed": a["changed"].mean(), "backbone_change": a["weight_change_backbone"].mean(),
                     "head_change": a["weight_change_head"].mean()})
    return pd.DataFrame(rows)


def contrasts(sub: pd.DataFrame) -> pd.DataFrame:
    wide = sub.pivot_table(index=["seed", "round", "student"], columns="arm", values="d_acc")
    rows = []
    for a, b in CONTRASTS:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna()
        per_seed = d.groupby(level="seed").mean()
        rows.append({"contrast": f"{a} - {b}", "n": len(d), "mean": d.mean(), "a_better_share": (d > 0).mean(),
                     "seeds_a_better": f"{int((per_seed > 0).sum())}/{len(per_seed)}", "p": sign_flip_p(per_seed)})
    return pd.DataFrame(rows)


def indent(text: str) -> str:
    return "   " + text.replace("\n", "\n   ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["Geometry of Mixed-Loss/results/peers_ce_variants",
                                                   "Geometry of Mixed-Loss/results/peers_lr_controls"])
    ap.add_argument("--earlier", default="Geometry of Mixed-Loss/results/peers_celnet_pairing",
                    help="earlier runs in the same setting, used as references in part 3 (skipped if missing)")
    args = ap.parse_args()
    roots = [Path(r) for r in args.roots if Path(r).exists()]
    if not roots:
        raise SystemExit(f"none of {args.roots} exists")
    ev = load_events(roots)
    if ev.empty:
        raise SystemExit(f"no variant records under {roots}")
    out = roots[0] / "analysis"
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
        print("  p = sign-flip test on per-seed means; broken / fixed / changed in % of val images")
        for cond, c in sub.groupby("condition"):
            print(f"\n  [{cond}]   oracle updates: {c[c['arm'] == 'kd'].shape[0]}")
            print(indent(per_update_table(c).round(3).to_string(index=False)))
            con = contrasts(c)
            if not con.empty:
                print("   paired contrasts from the same state (label-specific effect at equal learning rate / scope):")
                print(indent(con.round(3).to_string(index=False)))

    sched = ev[ev["condition"].str.contains("then") & (ev["applied"])]
    if not sched.empty:
        print("\n" + line)
        print("SCHEDULE: applied oracle updates before and after the switch to the gentle update")
        print(indent(sched.groupby(["condition", "arm"]).agg(
            n=("d_acc", "size"), mean_pre_acc=("pre_acc", "mean"), d_acc=("d_acc", "mean"),
            minus_kd=("minus_kd", "mean"), better_than_kd=("minus_kd", lambda x: (x > 0).mean())).round(3).to_string()))

    print("\n" + line)
    print("PART 3. OVER THE RUN: accuracy of the peers (mean over seeds)")
    runs = pd.concat([run_summary(r) for r in roots], ignore_index=True)
    earlier = Path(args.earlier)
    if earlier.exists():
        runs = pd.concat([runs, run_summary(earlier, earlier=True)], ignore_index=True)
    tab = runs.groupby("condition").agg(seeds=("seed", "size"), val_acc_last_40_rounds=("val_acc_last_40_rounds", "mean"),
                                        learner_test=("learner_test", "mean"), learner_test_sd=("learner_test", "std"),
                                        ensemble_test=("ensemble_test", "mean"))
    print(indent(tab.round(2).to_string()))
    for ref_name in ["KD-shaped: normal CE (earlier runs)", "all-CE: normal CE (earlier runs)"]:
        if ref_name not in set(runs["condition"]):
            continue
        ref = runs[runs["condition"] == ref_name]
        print(f"\n   vs {ref_name}, per-seed learner test accuracy (exact permutation test):")
        for cond, c in runs.groupby("condition"):
            if cond == ref_name or "(earlier runs)" in cond or cond.split(":")[0] != ref_name.split(":")[0]:
                continue
            print(f"   {cond:75s} {c['learner_test'].mean() - ref['learner_test'].mean():+.2f}  "
                  f"p {perm_p(c['learner_test'], ref['learner_test']):.4f}")
    print(f"\nwrote {out / 'variant_events.csv'}")


if __name__ == "__main__":
    main()
