"""Analyze the loss-switch experiment: does switching KD -> CE hurt, and does it depend on the teacher?

Writes per_epoch.csv (every epoch of every run) and summary.csv, and prints the main tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEACHER_ORDER = ["none", "ce_e20", "ce_e50", "ce_e100", "kd_g1", "kd_g2"]
PER_EPOCH_COLS = ["epoch", "phase", "lr", "train_loss", "val_acc", "val_loss", "val_ece",
                  "val_conf", "val_agree", "delta_acc", "probe_ce_loss", "probe_acc", "probe_ce_grad_norm"]


def load(students_dir: Path):
    epochs, runs = [], []
    for res_path in sorted(students_dir.glob("*/*/seed*/results.json")):
        r = json.loads(res_path.read_text())
        mode, teacher, seed_dir = res_path.parts[-4], res_path.parts[-3], res_path.parts[-2]
        seed, p1 = int(seed_dir.replace("seed", "")), r["phase1_epochs"]
        t = r.get("teacher") or {}
        df = pd.DataFrame({c: r[c] for c in PER_EPOCH_COLS})
        df.insert(0, "mode", mode)
        df.insert(1, "teacher", teacher)
        df.insert(2, "seed", seed)
        df["relative_epoch"] = df["epoch"] - p1          # <=0 before the switch, 1 = first epoch after it
        epochs.append(df)

        at_switch = df[df["epoch"] == p1].iloc[0]
        runs.append({
            "mode": mode, "teacher": teacher, "seed": seed,
            "teacher_objective": t.get("objective"),
            "teacher_test_acc": (t.get("test") or {}).get("acc"),
            "teacher_train_ce_loss": (t.get("train_probe") or {}).get("ce_loss"),
            "teacher_train_acc": (t.get("train_probe") or {}).get("acc"),
            "test_at_switch": r["test_at_switch"]["acc"], "test_final": r["test_final"]["acc"],
            "val_at_switch": at_switch["val_acc"], "val_final": df["val_acc"].iloc[-1],
            "val_peak_after": df[df["relative_epoch"] > 0]["val_acc"].max(),
            "probe_ce_loss_at_switch": at_switch["probe_ce_loss"],
            "probe_acc_at_switch": at_switch["probe_acc"],
            "probe_ce_loss_final": df["probe_ce_loss"].iloc[-1],
            "probe_acc_final": df["probe_acc"].iloc[-1],
            "ece_at_switch": at_switch["val_ece"], "ece_final": df["val_ece"].iloc[-1],
            "agree_at_switch": at_switch["val_agree"], "agree_final": df["val_agree"].iloc[-1],
        })
    if not runs:
        raise SystemExit(f"no results.json found under {students_dir}")
    per_epoch = pd.concat(epochs, ignore_index=True)
    per_run = pd.DataFrame(runs)
    for df in (per_epoch, per_run):
        df["teacher"] = pd.Categorical(df["teacher"], [t for t in TEACHER_ORDER if t in set(df["teacher"])],
                                       ordered=True)
    per_run["test_change"] = per_run["test_final"] - per_run["test_at_switch"]
    per_run["val_change"] = per_run["val_final"] - per_run["val_at_switch"]
    return per_epoch, per_run.sort_values(["mode", "teacher", "seed"])


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided paired permutation test (flip the sign of each seed's difference)."""
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    means = [abs(np.dot(signs, d) / len(d)) for signs in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def pearson(x: np.ndarray, y: np.ndarray):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def paired(per_run: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per teacher: kd_then_ce minus kd_only on the same seed. Negative = the switch to CE hurt."""
    pair = per_run[per_run["mode"].isin(["kd_then_ce", "kd_only"])]
    wide = pair.pivot_table(index=["teacher", "seed"], columns="mode", values=col, observed=True)
    rows = []
    for teacher, d in wide.groupby("teacher", observed=True):
        if not {"kd_then_ce", "kd_only"} <= set(d.columns):
            continue
        diff = (d["kd_then_ce"] - d["kd_only"]).dropna().to_numpy()
        rows.append({"teacher": teacher, "n": len(diff), "kd_only": d["kd_only"].mean(),
                     "kd_then_ce": d["kd_then_ce"].mean(), "switch_effect": diff.mean(),
                     "sd": diff.std(ddof=1) if len(diff) > 1 else np.nan,
                     "n_seeds_hurt": int((diff < 0).sum()), "p_perm": sign_flip_p(diff)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/switch_exp1")
    args = ap.parse_args()
    root = Path(args.root)
    per_epoch, per_run = load(root / "students")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_epoch.to_csv(out / "per_epoch.csv", index=False)
    per_run.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)

    def fmt(df):
        return df.round(3).to_string(index=False)

    print("=" * 100)
    print("RUNS FOUND (expect 10 seeds per row)")
    print(fmt(per_run.groupby(["mode", "teacher"], observed=True).agg(
        n_seeds=("seed", "nunique"), n_runs=("test_final", "size")).reset_index()))

    print("\n" + "=" * 100)
    print("TEACHERS (train CE loss is measured on 5000 fixed training images; low = near the CE minimum)")
    t = (per_run.dropna(subset=["teacher_test_acc"])
         .groupby("teacher", observed=True)[["teacher_objective", "teacher_test_acc",
                                             "teacher_train_ce_loss", "teacher_train_acc"]].first().reset_index())
    print(fmt(t))

    print("\n" + "=" * 100)
    print("SANITY: kd_then_ce and kd_only must be identical up to the switch")
    pre = per_epoch[(per_epoch["relative_epoch"] <= 0) & (per_epoch["mode"] != "ce_only")]
    w = pre.pivot_table(index=["teacher", "seed", "epoch"], columns="mode", values="val_acc", observed=True)
    if {"kd_then_ce", "kd_only"} <= set(w.columns):
        print(f"  max |val acc difference| before the switch: {(w['kd_then_ce'] - w['kd_only']).abs().max():.4f}")

    print("\n" + "=" * 100)
    print("TEST ACCURACY (mean over seeds)")
    print(fmt(per_run.groupby(["mode", "teacher"], observed=True).agg(
        at_switch=("test_at_switch", "mean"), final=("test_final", "mean"),
        change=("test_change", "mean"), change_sd=("test_change", "std")).reset_index()))
    ce_only = per_run[per_run["mode"] == "ce_only"]["test_final"]
    print(f"\n  CE-only baseline, final test accuracy: {ce_only.mean():.2f} +/- {ce_only.std(ddof=1):.2f} "
          f"(n={len(ce_only)})")

    print("\n" + "=" * 100)
    print("THE SWITCH EFFECT: kd_then_ce minus kd_only, same seed. NEGATIVE = switching to CE hurt.")
    for col, label in [("test_final", "final test accuracy"), ("val_final", "final val accuracy"),
                       ("val_peak_after", "best val accuracy after the switch")]:
        print(f"\n  [{label}]")
        print("   " + fmt(paired(per_run, col)).replace("\n", "\n   "))

    print("\n" + "=" * 100)
    print("SHAPE OF THE EFFECT: mean val accuracy change since the switch (kd_then_ce, then kd_only)")
    after = per_epoch[per_epoch["relative_epoch"] > 0].copy()
    switch_acc = per_epoch[per_epoch["relative_epoch"] == 0].set_index(["mode", "teacher", "seed"])["val_acc"]
    after["cum"] = after["val_acc"] - after.set_index(["mode", "teacher", "seed"]).index.map(switch_acc)
    marks = [1, 2, 5, 10, 20, 40, 60]
    for mode in ["kd_then_ce", "kd_only"]:
        sub = after[after["mode"] == mode]
        if sub.empty:
            continue
        tab = (sub[sub["relative_epoch"].isin(marks)]
               .pivot_table(index="teacher", columns="relative_epoch", values="cum", observed=True))
        print(f"\n  [{mode}]  epochs after the switch")
        print("   " + tab.round(2).to_string().replace("\n", "\n   "))

    print("\n" + "=" * 100)
    print("DOES THE HARM TRACK DISTANCE FROM THE CE MINIMUM?")
    eff = paired(per_run, "test_final").merge(t, on="teacher")
    for xcol, label in [("teacher_train_ce_loss", "teacher train CE loss"), ("teacher_test_acc", "teacher test acc")]:
        r, n = pearson(eff[xcol].to_numpy(float), eff["switch_effect"].to_numpy(float))
        print(f"  across teachers: {label:24s} vs switch effect   r = {r:+.3f} (n={n} teachers)")
    pair = per_run[per_run["mode"].isin(["kd_then_ce", "kd_only"])]
    wide = pair.pivot_table(index=["teacher", "seed"], columns="mode",
                            values=["test_final", "probe_ce_loss_at_switch"], observed=True).dropna()
    if not wide.empty:
        d = (wide[("test_final", "kd_then_ce")] - wide[("test_final", "kd_only")]).to_numpy(float)
        gap = wide[("probe_ce_loss_at_switch", "kd_then_ce")].to_numpy(float)
        r, n = pearson(gap, d)
        print(f"  across runs:     student's train CE loss at the switch vs switch effect   r = {r:+.3f} (n={n} runs)")

    print("\n" + "=" * 100)
    print(f"wrote {out / 'per_epoch.csv'} and {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
