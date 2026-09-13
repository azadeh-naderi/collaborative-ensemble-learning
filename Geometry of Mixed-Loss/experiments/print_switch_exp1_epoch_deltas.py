"""Print the epoch-to-epoch val accuracy change right after the switch (100->101, 101->102, ...) for switch_exp1.

Reads analysis/per_epoch.csv written by analyze_switch_exp1.py. Accuracies are in percentage points
(the val set has 5000 images, so one image = 0.02).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEACHER_ORDER = ["none", "ce_e20", "ce_e50", "ce_e100", "kd_g1", "kd_g2"]


def table(df, values, aggfunc="mean"):
    tab = df.pivot_table(index="teacher", columns="epoch", values=values, aggfunc=aggfunc, observed=True)
    tab = tab.reindex([t for t in TEACHER_ORDER if t in tab.index])
    tab.columns = [f"{e - 1}->{e}" for e in tab.columns]
    return tab


def show(title, tab, decimals=2):
    print(f"\n{title}")
    print(tab.round(decimals).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/switch_exp1")
    ap.add_argument("--window", type=int, default=20, help="number of epochs after the switch to show")
    ap.add_argument("--pre", type=int, default=2, help="number of epochs before the switch to show for reference")
    ap.add_argument("--per_seed", action="store_true", help="also print every seed for kd_then_ce")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root / "analysis" / "per_epoch.csv")
    df = df[(df["relative_epoch"] > -args.pre) & (df["relative_epoch"] <= args.window)].copy()
    df["drop"] = (df["delta_acc"] < 0).astype(int)
    switch = df.loc[df["relative_epoch"] == 0].set_index(["mode", "teacher", "seed"])["val_acc"]
    df["cum"] = df["val_acc"] - df.set_index(["mode", "teacher", "seed"]).index.map(switch)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    switch_epoch = int(df.loc[df["relative_epoch"] == 0, "epoch"].iloc[0])
    print(f"Switch after epoch {switch_epoch}. Columns up to {switch_epoch - 1}->{switch_epoch} are BEFORE the switch "
          f"(reference); {switch_epoch}->{switch_epoch + 1} is the first CE epoch.")
    print("Values: val accuracy at the later epoch minus val accuracy at the earlier epoch, percentage points.")

    kd_ce, kd_only, ce_only = (df[df["mode"] == m] for m in ("kd_then_ce", "kd_only", "ce_only"))
    show("[1] MEAN epoch-to-epoch change, KD->CE (10 seeds)", table(kd_ce, "delta_acc"))
    ref = pd.concat([table(kd_only, "delta_acc"),
                     table(ce_only, "delta_acc").rename(index={"none": "ce_only"})])
    show("[2] MEAN epoch-to-epoch change, no switch: KD-only rows, and CE-only (CE the whole time)", ref)
    show("[3] NUMBER OF SEEDS (of 10) whose accuracy DROPPED at that epoch, KD->CE", table(kd_ce, "drop", "sum"), 0)
    show("[4] NUMBER OF SEEDS (of 10) whose accuracy DROPPED at that epoch, KD-only", table(kd_only, "drop", "sum"), 0)
    show(f"[5] MEAN cumulative change since epoch {switch_epoch}, KD->CE", table(kd_ce, "cum"))

    if args.per_seed:
        for teacher in [t for t in TEACHER_ORDER if t in set(kd_ce["teacher"])]:
            sub = kd_ce[kd_ce["teacher"] == teacher]
            tab = sub.pivot_table(index="seed", columns="epoch", values="delta_acc")
            tab.columns = [f"{e - 1}->{e}" for e in tab.columns]
            show(f"[per seed] KD->CE, teacher {teacher}: epoch-to-epoch change", tab)


if __name__ == "__main__":
    main()
