"""Analyze the peer version of the loss-switch experiment (switch_peers.py).

Main question: in a group of peers teaching each other at LR 0.1, does a CE update on the true labels lower accuracy
compared with the KD update the peer would otherwise have made from the same state (as seen in the CEL-Net logs)?

Writes updates.csv to <root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [0, 50, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 100]


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or len(d) > 20:
        return float("nan")
    obs = abs(d.mean())
    means = [abs(np.dot(s, d) / len(d)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def indent(text: str) -> str:
    return "   " + text.replace("\n", "\n   ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/switch_peers")
    ap.add_argument("--pilot_per_epoch", default="Geometry of Mixed-Loss/results/switch_lr01_pilot/analysis/per_epoch.csv",
                    help="LR-0.1 pilot per-epoch CSV, used for the CE-only reference (skipped if missing)")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    frames, runs = [], []
    for path in sorted(root.glob("seed*/results.json")):
        r = json.loads(path.read_text())
        u = pd.DataFrame(r["updates"])
        u.insert(0, "seed", r["seed"])
        frames.append(u)
        runs.append({"seed": r["seed"], "n_peers": r["n_peers"], "rounds": r["rounds"], "ce_every": r["ce_every"],
                     "peer_test_mean": np.mean([m["acc"] for m in r["test_final"]]),
                     "ensemble_test": r["test_ensemble_acc"]})
    if not frames:
        raise SystemExit(f"no results.json under {root}")
    df = pd.concat(frames, ignore_index=True)
    runs = pd.DataFrame(runs)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "updates.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    line = "=" * 100
    w = args.window

    print(line)
    print("RUNS")
    print(runs.round(2).to_string(index=False))

    # ------------------------------------------------------------------ accuracy over time
    print("\n" + line)
    print(f"ACCURACY OVER TIME (mean val accuracy of all peers, windows of {w} rounds)")
    df["window"] = ((df["round"] - 1) // w) * w + 1
    df["window_label"] = df["window"].astype(str) + "-" + (df["window"] + w - 1).astype(str)
    tab = df.groupby("window").agg(rounds=("window_label", "first"), peers_val_acc=("val_acc", "mean"),
                                   peers_train_label_agreement=("probe_acc", "mean"))
    pilot = Path(args.pilot_per_epoch)
    if pilot.exists():
        p = pd.read_csv(pilot)
        p = p[p["mode"] == "ce_only"].copy()
        p["window"] = ((p["epoch"] - 1) // w) * w + 1
        tab["ce_only_student_val_acc (pilot)"] = p.groupby("window")["val_acc"].mean()
        tab["peers_minus_ce_only"] = tab["peers_val_acc"] - tab["ce_only_student_val_acc (pilot)"]
    print(indent(tab.round(2).to_string(index=False)))
    print("   train label agreement = accuracy on 5000 fixed training images; for comparison the fixed teachers of the"
          " pilot had 96.1 (CE teacher) and 90.8 (KL gen 1)")

    # ------------------------------------------------------------------ CE updates vs counterfactual KD
    ev = df[(df["phase"] == "ce") & df["branch_kd_val_acc"].notna()].copy()
    print("\n" + line)
    print("CE UPDATES. From the SAME weights and optimizer state:")
    print("  d_CE = accuracy change from the real CE update; d_KD = change had it been a KD update instead.")
    print("  CE - KD < 0 means the CE update did worse than a KD update would have.")
    if ev.empty:
        print("  no CE updates with the counterfactual found (was --branch_at_ce set?)")
    else:
        ev["d_CE"] = ev["delta_acc"]
        ev["d_KD"] = ev["branch_kd_val_acc"] - ev["pre_acc"]
        ev["CE_minus_KD"] = ev["d_CE"] - ev["d_KD"]
        ev["pre_bin"] = pd.cut(ev["pre_acc"], BINS)

        def summarize(g):
            return pd.Series({"n": len(g), "d_CE": g["d_CE"].mean(), "d_KD": g["d_KD"].mean(),
                              "CE_minus_KD": g["CE_minus_KD"].mean(), "CE_worse_share": (g["CE_minus_KD"] < 0).mean(),
                              "CE_drop_share": (g["d_CE"] < 0).mean(), "KD_drop_share": (g["d_KD"] < 0).mean()})

        s = summarize(ev)
        per_seed = ev.groupby("seed")["CE_minus_KD"].mean()
        print(f"\n  all CE updates: n={int(s.n)}  d_CE {s.d_CE:+.2f}  d_KD {s.d_KD:+.2f}  CE-KD {s.CE_minus_KD:+.2f}  "
              f"CE worse in {s.CE_worse_share:.0%}  seeds with CE worse on average "
              f"{int((per_seed < 0).sum())}/{len(per_seed)}  p {sign_flip_p(per_seed):.3f}")
        late = ev[ev["pre_acc"] >= 72.5]
        if not late.empty:
            s2, ps2 = summarize(late), late.groupby("seed")["CE_minus_KD"].mean()
            print(f"  CE updates from >= 72.5%: n={int(s2.n)}  d_CE {s2.d_CE:+.2f}  d_KD {s2.d_KD:+.2f}  "
                  f"CE-KD {s2.CE_minus_KD:+.2f}  CE worse in {s2.CE_worse_share:.0%}  seeds with CE worse on average "
                  f"{int((ps2 < 0).sum())}/{len(ps2)}  p {sign_flip_p(ps2):.3f}")
        for key, title in [("pre_bin", "by accuracy before the CE update"), ("window_label", f"by round ({w}-round windows)")]:
            order = ev.drop_duplicates(key).sort_values("round")[key] if key == "window_label" else None
            tab = ev.groupby(key, observed=True)[["d_CE", "d_KD", "CE_minus_KD"]].apply(summarize)
            if order is not None:
                tab = tab.reindex(order)
            print(f"\n  [{title}]")
            print(indent(tab.round(2).to_string()))

    # ------------------------------------------------------------------ KD updates
    kd = df[df["phase"] == "kd"].copy()
    kd["pre_bin"] = pd.cut(kd["pre_acc"], BINS)
    print("\n" + line)
    print("KD UPDATES by accuracy before the update (compare with the CE table above, and with CE-only epochs)")
    print(indent(kd.groupby("pre_bin", observed=True)["delta_acc"].agg(
        n="size", mean="mean", drop_share=lambda x: (x < 0).mean()).round(2).to_string()))

    # position of each update relative to the peer's previous CE update (sawtooth check)
    df = df.sort_values(["seed", "peer", "round"])
    df["since_ce"] = np.nan
    for (_, _), g in df.groupby(["seed", "peer"]):
        last, vals = None, []
        for rnd, phase in zip(g["round"], g["phase"]):
            vals.append(np.nan if last is None else rnd - last)
            if phase == "ce":
                last = rnd
        df.loc[g.index, "since_ce"] = vals
    pos = df[df["round"] > 20].copy()
    pos["position"] = np.where(pos["phase"] == "ce", "CE update", "KD, " + pos["since_ce"].astype("Int64").astype(str)
                               + " after CE")
    print("\n  [rounds 21+: change by position relative to the peer's last CE update]")
    print(indent(pos.groupby("position")["delta_acc"].agg(n="size", mean="mean", drop_share=lambda x: (x < 0).mean())
                 .round(2).to_string()))

    print("\n" + line)
    print(f"wrote {out / 'updates.csv'}")


if __name__ == "__main__":
    main()
