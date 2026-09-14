"""Loss view of the loss-switch experiments: does val loss go up right after switching from KD to CE?

Change is measured relative to the state just before the switch, at the first epoch(s) after it, against the
no-switch run from the same seed (KD-only), never as last-KD-epoch vs last-CE-epoch.

  switch_exp1 : KD epochs 1-100 (LR decayed to 0.001), CE epochs 101-160.
  LR-0.1 pilot: KD epochs 1-50, CE epochs 51-80 at constant LR 0.1, plus interleaved runs (every 5th epoch CE).
val_loss is CE on the validation set with the true labels; train_CE_loss is CE on 5000 fixed training images.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    means = [abs(np.dot(s, d) / len(d)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def after_switch(df, switch_epoch, windows, value):
    """Per teacher and window after the switch: mean change of `value` since the switch epoch, KD->CE vs KD-only."""
    rows = []
    base = df[df["epoch"] == switch_epoch].set_index(["mode", "teacher", "seed"])[value]
    for teacher in sorted(df.loc[df["mode"] == "kd_then_ce", "teacher"].unique()):
        for lo, hi in windows:
            w = df[(df["epoch"] >= switch_epoch + lo) & (df["epoch"] <= switch_epoch + hi)]
            m = w.groupby(["mode", "teacher", "seed"])[value].mean() - base
            kc, ko = m.loc["kd_then_ce", teacher], m.loc["kd_only", teacher]
            d = (kc - ko).dropna()
            row = {"teacher": teacher, "epochs_after": f"{lo}" if lo == hi else f"{lo}-{hi}",
                   "KD->CE": kc.mean(), "KD-only": ko.mean(), "difference": d.mean(),
                   "seeds_higher": f"{int((d > 0).sum())}/{len(d)}", "p": sign_flip_p(d)}
            if ("ce_only", "none") in {k[:2] for k in m.index}:
                row["CE-only"] = m.loc["ce_only", "none"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def show(title, tab):
    print(f"\n  [{title}]")
    print("   " + tab.round(3).to_string(index=False).replace("\n", "\n   "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1_csv", default="Geometry of Mixed-Loss/results/switch_exp1/analysis/per_epoch.csv")
    ap.add_argument("--pilot_csv", default="Geometry of Mixed-Loss/results/switch_lr01_pilot/analysis/per_epoch.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    line = "=" * 100

    exp1 = Path(args.exp1_csv)
    if exp1.exists():
        df = pd.read_csv(exp1)
        df["train_CE_loss"] = df["probe_ce_loss"]
        print(line)
        print("SWITCH_EXP1 (switch after epoch 100, LR 0.001). Change since epoch 100; difference = KD->CE minus KD-only,")
        print("same seed; seeds_higher = seeds where KD->CE is higher than KD-only (for loss: higher = worse).")
        windows = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10), (20, 20), (40, 40), (60, 60)]
        show("val loss", after_switch(df, 100, windows, "val_loss"))
        show("val accuracy (for reference)", after_switch(df, 100, windows, "val_acc"))
        show("train CE loss", after_switch(df, 100, windows, "train_CE_loss"))

    pilot = Path(args.pilot_csv)
    if pilot.exists():
        df = pd.read_csv(pilot)
        df["train_CE_loss"] = df["probe_ce_loss"]
        print("\n" + line)
        print("LR-0.1 PILOT, SINGLE SWITCH (after epoch 50). Window means (single epochs are noisy at LR 0.1).")
        windows = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 30)]
        show("val loss", after_switch(df, 50, windows, "val_loss"))
        show("val accuracy (for reference)", after_switch(df, 50, windows, "val_acc"))
        show("train CE loss", after_switch(df, 50, windows, "train_CE_loss"))

        print("\n" + line)
        print("LR-0.1 PILOT, INTERLEAVED (every 5th epoch CE), epochs 21-80: change from the previous epoch by position")
        it = df[df["mode"] == "interleaved"].sort_values(["teacher", "seed", "epoch"]).copy()
        for col in ["val_loss", "train_CE_loss"]:
            it["d_" + col] = it.groupby(["teacher", "seed"])[col].diff()
        it = it[it["epoch"] > 20]
        pos = ((it["epoch"] - 1) % 5) + 1
        it["position"] = pos.map({5: "5: CE epoch", 1: "1: KD, 1st after CE", 2: "2: KD, 2nd", 3: "3: KD, 3rd",
                                  4: "4: KD, 4th"})
        tab = it.groupby(["teacher", "position"]).agg(
            acc_change=("delta_acc", "mean"), val_loss_change=("d_val_loss", "mean"),
            val_loss_up_share=("d_val_loss", lambda x: (x > 0).mean()),
            train_CE_loss_change=("d_train_CE_loss", "mean")).reset_index()
        show("mean change from the previous epoch", tab)
        if "branch_kd_val_loss" in df.columns:
            ev = it[(it["phase"] == "ce") & it["branch_kd_val_loss"].notna()].copy()
            prev = df.sort_values(["teacher", "seed", "epoch"]).groupby(["mode", "teacher", "seed"])["val_loss"].shift(1)
            ev["pre_loss"] = prev.loc[ev.index]
            ev["loss_CE"] = ev["val_loss"] - ev["pre_loss"]
            ev["loss_KD"] = ev["branch_kd_val_loss"] - ev["pre_loss"]
            ev["CE_higher"] = ev["loss_CE"] > ev["loss_KD"]
            show("CE epoch vs KD counterfactual from the same state: val loss change",
                 ev.groupby("teacher").agg(n=("loss_CE", "size"), loss_CE=("loss_CE", "mean"),
                                           loss_KD=("loss_KD", "mean"), CE_higher_share=("CE_higher", "mean"))
                 .reset_index())
        else:
            print("\n  (counterfactual KD val loss is not in this CSV; rerun analyze_switch_lr01_pilot.py on Wulver to add it)")


if __name__ == "__main__":
    main()
