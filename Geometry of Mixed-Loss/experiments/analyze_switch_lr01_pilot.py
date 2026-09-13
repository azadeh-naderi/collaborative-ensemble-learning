"""Analyze the LR-0.1 pilot of the loss-switch experiment.

Part 1 (single switch): KD epochs 1-50 then CE epochs 51-80, against KD-only and CE-only, all at LR 0.1.
Part 2 (interleaved): KD with every 5th epoch CE. At each CE epoch, compare the real CE epoch with a KD epoch
run from the exact same weights and optimizer state (the counterfactual branch).

Writes ce_events.csv and per_epoch.csv to <root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [0, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 100]


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    means = [abs(np.dot(s, d) / len(d)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def load(students: Path):
    rows = []
    for path in sorted(students.glob("*/*/seed*/results.json")):
        r = json.loads(path.read_text())
        mode, teacher = path.parts[-4], path.parts[-3]
        n = len(r["epoch"])
        branch = r.get("branch_kd_val_acc") or [None] * n
        df = pd.DataFrame({
            "mode": mode, "teacher": teacher, "seed": r["seed"], "epoch": r["epoch"], "phase": r["phase"],
            "lr": r["lr"], "val_acc": r["val_acc"], "delta_acc": r["delta_acc"], "val_loss": r["val_loss"],
            "probe_ce_loss": r["probe_ce_loss"], "branch_kd_val_acc": branch})
        df["pre_acc"] = df["val_acc"] - df["delta_acc"]
        df["phase1_epochs"] = r["phase1_epochs"]
        rows.append(df)
    if not rows:
        raise SystemExit(f"no results.json under {students}")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/switch_lr01_pilot")
    args = ap.parse_args()
    root = Path(args.root)
    df = load(root / "students")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_epoch.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    line = "=" * 100

    print(line)
    print("RUNS FOUND")
    print(df.groupby(["mode", "teacher"])["seed"].nunique().rename("n_seeds").reset_index().to_string(index=False))
    print("\nLR values seen:", sorted(df["lr"].unique()))

    # ------------------------------------------------------------------ part 1
    p1 = int(df["phase1_epochs"].iloc[0])
    print("\n" + line)
    print(f"PART 1: SINGLE SWITCH AT LR 0.1 (KD epochs 1-{p1}, CE after). Val accuracy, percentage points.")
    at = df[df["epoch"] == p1].groupby(["mode", "teacher"])["val_acc"].agg(["mean", "std"]).round(2)
    print(f"\n  val accuracy at epoch {p1} (is the KD student above the CE-only level?)")
    print("   " + at.to_string().replace("\n", "\n   "))
    late = df[(df["epoch"] > p1 - 10) & (df["epoch"] <= p1)].groupby(["mode", "teacher"])["val_acc"].mean().round(2)
    print(f"\n  mean val accuracy over epochs {p1 - 9}-{p1} (smooths the epoch-to-epoch noise at LR 0.1)")
    print("   " + late.to_string().replace("\n", "\n   "))

    single = df[df["mode"].isin(["kd_then_ce", "kd_only", "ce_only"]) & (df["epoch"] > p1)].copy()
    ref = df[df["epoch"] == p1].set_index(["mode", "teacher", "seed"])["val_acc"]
    single["cum"] = single["val_acc"] - single.set_index(["mode", "teacher", "seed"]).index.map(ref)
    single["step"] = [f"{e - 1}->{e}" for e in single["epoch"]]
    single["col"] = single["mode"] + ":" + single["teacher"]
    order = sorted(single["col"].unique())
    steps = [f"{e - 1}->{e}" for e in sorted(single["epoch"].unique())]
    for value, title in [("delta_acc", "mean epoch-to-epoch change"), ("cum", f"mean cumulative change since epoch {p1}")]:
        tab = single.pivot_table(index="step", columns="col", values=value, aggfunc="mean").reindex(steps)[order]
        print(f"\n  [{title}]")
        print("   " + tab.round(2).to_string().replace("\n", "\n   "))
    drops = single.assign(drop=single["delta_acc"] < 0).pivot_table(
        index="step", columns="col", values="drop", aggfunc="sum").reindex(steps)[order]
    print("\n  [number of seeds whose accuracy dropped at that epoch]")
    print("   " + drops.astype(int).to_string().replace("\n", "\n   "))

    last = df[df["epoch"] == df["epoch"].max()]
    wide = last[last["mode"].isin(["kd_then_ce", "kd_only"])].pivot_table(
        index=["teacher", "seed"], columns="mode", values="val_acc")
    if {"kd_then_ce", "kd_only"} <= set(wide.columns):
        print("\n  [final val accuracy: KD->CE minus KD-only, same seed; negative = switch hurt]")
        for teacher, d in wide.groupby("teacher"):
            diff = (d["kd_then_ce"] - d["kd_only"]).dropna()
            print(f"   {teacher:8s} effect {diff.mean():+.2f}  sd {diff.std(ddof=1):.2f}  "
                  f"seeds hurt {int((diff < 0).sum())}/{len(diff)}  p {sign_flip_p(diff):.3f}")

    # ------------------------------------------------------------------ part 2
    ev = df[(df["mode"] == "interleaved") & (df["phase"] == "ce") & df["branch_kd_val_acc"].notna()].copy()
    print("\n" + line)
    print("PART 2: INTERLEAVED (every 5th epoch CE). At each CE epoch, from the SAME weights and optimizer state:")
    print("  d_CE = accuracy change from the real CE epoch; d_KD = change had that epoch been KD instead.")
    print("  CE - KD < 0 means the CE epoch did worse than a KD epoch would have.")
    if ev.empty:
        print("  no interleaved runs with the counterfactual branch found")
        return
    ev["d_CE"] = ev["delta_acc"]
    ev["d_KD"] = ev["branch_kd_val_acc"] - ev["pre_acc"]
    ev["CE_minus_KD"] = ev["d_CE"] - ev["d_KD"]
    ev["pre_bin"] = pd.cut(ev["pre_acc"], BINS)
    ev.drop(columns=["phase1_epochs"]).to_csv(out / "ce_events.csv", index=False)

    def summarize(g):
        return pd.Series({"n": len(g), "d_CE": g["d_CE"].mean(), "d_KD": g["d_KD"].mean(),
                          "CE_minus_KD": g["CE_minus_KD"].mean(),
                          "CE_worse_share": (g["CE_minus_KD"] < 0).mean(),
                          "CE_drop_share": (g["d_CE"] < 0).mean(), "KD_drop_share": (g["d_KD"] < 0).mean()})

    print("\n  [all CE epochs, per teacher; p = sign-flip test on the 5 per-seed means]")
    for teacher, g in ev.groupby("teacher"):
        per_seed = g.groupby("seed")["CE_minus_KD"].mean()
        s = summarize(g)
        print(f"   {teacher:8s} n={int(s.n):3d}  d_CE {s.d_CE:+.2f}  d_KD {s.d_KD:+.2f}  CE-KD {s.CE_minus_KD:+.2f}  "
              f"CE worse in {s.CE_worse_share:.0%}  seeds with CE worse on average "
              f"{int((per_seed < 0).sum())}/{len(per_seed)}  p {sign_flip_p(per_seed):.3f}")

    for key, title in [("pre_bin", "by accuracy before the CE epoch"), ("epoch", "by epoch")]:
        tab = ev.groupby(["teacher", key], observed=True)[["d_CE", "d_KD", "CE_minus_KD"]].apply(summarize)
        print(f"\n  [{title}]")
        print("   " + tab.round(2).to_string().replace("\n", "\n   "))

    # reference: ordinary epochs of the no-switch runs, binned by accuracy before the epoch
    base = df[df["mode"].isin(["ce_only", "kd_only"]) & (df["epoch"] > 1)].copy()
    base["pre_bin"] = pd.cut(base["pre_acc"], BINS)
    ref_tab = base.groupby(["mode", "teacher", "pre_bin"], observed=True)["delta_acc"].agg(
        n="size", mean="mean", drop_share=lambda x: (x < 0).mean())
    print("\n  [reference: ordinary epochs of CE-only and KD-only runs at LR 0.1, by accuracy before the epoch]")
    print("   " + ref_tab.round(2).to_string().replace("\n", "\n   "))

    print("\n" + line)
    print(f"wrote {out / 'per_epoch.csv'} and {out / 'ce_events.csv'}")


if __name__ == "__main__":
    main()
