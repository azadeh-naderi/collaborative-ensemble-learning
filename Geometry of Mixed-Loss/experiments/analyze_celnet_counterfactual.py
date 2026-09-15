"""Analyze celnet_ce_counterfactual.py runs: are CEL-Net's oracle CE updates really harmful?

For every oracle CE update, compares the real CE epoch with a KD epoch from the student's best peer, both from the
same weights and optimizer state (d_CE vs d_KD), in val accuracy and val loss. Also reproduces the CEL-Net-log view
(gain of CE and KD updates by accuracy before the update) and follows each CE-updated model over its next updates.

Writes updates.csv to <root>/analysis and prints the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [0, 50, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 90, 100]


def indent(text: str, n: int = 3) -> str:
    return " " * n + text.replace("\n", "\n" + " " * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Geometry of Mixed-Loss/results/celnet_counterfactual")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    root = Path(args.root)

    frames, runs = [], []
    for path in sorted(root.glob("*/seed*/results.json")):
        r = json.loads(path.read_text())
        policy, seed = r["config"]["pairing_strategy"], r["config"]["run_seed"]
        u = pd.DataFrame(r["updates"])
        u.insert(0, "policy", policy)
        u.insert(1, "seed", seed)
        frames.append(u)
        runs.append({"policy": policy, "seed": seed, "rounds": r["config"]["n_rounds"],
                     "learners": r["config"]["num_models"] - 1, "counterfactual": r["counterfactual"],
                     "mean_learner_test": np.mean(list(r["final_test_acc"].values())),
                     "ensemble_test": r["ensemble_test_acc_logprob"]})
    if not frames:
        raise SystemExit(f"no results.json under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["d_acc"] = df["post_acc"] - df["pre_acc"]
    df["d_loss"] = df["post_loss"] - df["pre_loss"]
    df["pre_bin"] = pd.cut(df["pre_acc"], BINS)
    w = args.window
    df["window"] = (((df["round"] - 1) // w) * w + 1).astype(str) + "-" + (((df["round"] - 1) // w + 1) * w).astype(str)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["pre_bin"]).to_csv(out / "updates.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    line = "=" * 110

    print(line)
    print("RUNS")
    print(pd.DataFrame(runs).round(2).to_string(index=False))
    lr_share = df.groupby(["policy", "phase"])["lr"].agg(lambda x: (x == x.max()).mean())
    print("\nshare of updates at the initial LR:")
    print(indent(lr_share.round(3).to_string()))

    for policy, p in df.groupby("policy"):
        print("\n" + line)
        print(f"POLICY {policy}")

        # ---------------------------------------------------------------- CEL-Net-log view
        view = p.groupby(["pre_bin", "phase"], observed=True).agg(
            n=("d_acc", "size"), acc_change=("d_acc", "mean"), drop_share=("d_acc", lambda x: (x < 0).mean()),
            loss_change=("d_loss", "mean")).unstack("phase")
        print("\n  [as in the CEL-Net logs: change per update by accuracy before it; CE vs KD updates are from DIFFERENT"
              " models and states]")
        print(indent(view.round(3).to_string()))

        # ---------------------------------------------------------------- counterfactual
        ev = p[(p["phase"] == "ce") & p.get("cf_acc", pd.Series(index=p.index, dtype=float)).notna()].copy()
        if ev.empty:
            print("\n  no counterfactual found for this policy")
            continue
        ev["d_CE"] = ev["d_acc"]
        ev["d_KD"] = ev["cf_acc"] - ev["pre_acc"]
        ev["CE_minus_KD"] = ev["d_CE"] - ev["d_KD"]
        ev["loss_CE"] = ev["d_loss"]
        ev["loss_KD"] = ev["cf_loss"] - ev["pre_loss"]
        ev["lossCE_minus_lossKD"] = ev["loss_CE"] - ev["loss_KD"]
        ev["teacher_gap"] = ev["cf_teacher_acc"] - ev["pre_acc"]

        def summarize(g):
            return pd.Series({"n": len(g), "d_CE": g["d_CE"].mean(), "d_KD": g["d_KD"].mean(),
                              "CE_minus_KD": g["CE_minus_KD"].mean(), "CE_worse_share": (g["CE_minus_KD"] < 0).mean(),
                              "CE_drop_share": (g["d_CE"] < 0).mean(), "KD_drop_share": (g["d_KD"] < 0).mean(),
                              "loss_CE": g["loss_CE"].mean(), "loss_KD": g["loss_KD"].mean(),
                              "lossCE_minus_lossKD": g["lossCE_minus_lossKD"].mean(),
                              "cf_teacher_gap": g["teacher_gap"].mean()})

        cols = ["d_CE", "d_KD", "CE_minus_KD", "loss_CE", "loss_KD", "lossCE_minus_lossKD", "teacher_gap"]
        print("\n  [COUNTERFACTUAL: each oracle CE update vs a KD epoch from the best peer, from the SAME state]")
        print("   CE_minus_KD < 0 = CE did worse than KD would have; lossCE_minus_lossKD > 0 = CE raised val loss more;")
        print("   cf_teacher_gap = counterfactual teacher accuracy minus student accuracy")
        s = summarize(ev)
        per_seed = ev.groupby("seed")["CE_minus_KD"].mean()
        print(f"\n   all: n={int(s.n)}  d_CE {s.d_CE:+.2f}  d_KD {s.d_KD:+.2f}  CE-KD {s.CE_minus_KD:+.2f}  "
              f"CE worse in {s.CE_worse_share:.0%}  | loss CE {s.loss_CE:+.3f} KD {s.loss_KD:+.3f}  "
              f"| per-seed CE-KD: " + ", ".join(f"{v:+.2f}" for v in per_seed))
        for key, title in [("pre_bin", "by accuracy before the CE update"), ("window", f"by round ({w}-round windows)")]:
            tab = ev.groupby(key, observed=True, sort=False)[cols].apply(summarize)
            if key == "window":
                tab = tab.reindex(ev.drop_duplicates("window").sort_values("round")["window"])
            print(f"\n   [{title}]")
            print(indent(tab.round(3).to_string(), 6))
        by_seed = ev[ev["pre_acc"] >= 70].groupby("seed")[cols].apply(summarize)
        if not by_seed.empty:
            print("\n   [CE updates from >= 70%, per seed]")
            print(indent(by_seed.round(3).to_string(), 6))

        # tug of war: does the harm grow with the number of KD updates since the model's previous CE update?
        seq = p.sort_values(["seed", "student", "round"])
        since = {}
        for (seed, student), g in seq.groupby(["seed", "student"]):
            count = None
            for idx, phase in zip(g.index, g["phase"]):
                if phase == "ce":
                    since[idx] = count
                    count = 0
                elif count is not None and phase == "kd":
                    count += 1
        ev["kd_since_last_ce"] = pd.Series(since, dtype=float).reindex(ev.index)
        tw = ev[(ev["pre_acc"] >= 70) & ev["kd_since_last_ce"].notna()].copy()
        if len(tw) >= 3:
            tw["kd_since_bin"] = pd.cut(tw["kd_since_last_ce"], [-1, 2, 4, 7, 12, 10_000],
                                        labels=["0-2", "3-4", "5-7", "8-12", "13+"])
            tab = tw.groupby("kd_since_bin", observed=True).agg(
                n=("CE_minus_KD", "size"), mean_pre_acc=("pre_acc", "mean"), d_CE=("d_CE", "mean"),
                CE_minus_KD=("CE_minus_KD", "mean"), CE_worse_share=("CE_minus_KD", lambda x: (x < 0).mean()),
                lossCE_minus_lossKD=("lossCE_minus_lossKD", "mean"))
            print("\n   [CE updates from >= 70%, by number of KD updates since the model's previous CE update]")
            print(indent(tab.round(3).to_string(), 6))
            x = np.column_stack([np.ones(len(tw)), tw["kd_since_last_ce"].clip(upper=15), tw["pre_acc"]])
            b = np.linalg.lstsq(x, tw["CE_minus_KD"].to_numpy(float), rcond=None)[0]
            print(f"      linear fit: CE_minus_KD = {b[0]:+.2f} {b[1]:+.3f} * KD_updates_since_last_CE (capped at 15) "
                  f"{b[2]:+.3f} * accuracy_before  (n={len(tw)})")

        # ---------------------------------------------------------------- after the CE update
        seq = p.sort_values(["seed", "student", "round"]).reset_index(drop=True)
        rows = []
        for (seed, student), g in seq.groupby(["seed", "student"]):
            g = g.reset_index(drop=True)
            for i in g.index[g["phase"] == "ce"]:
                pre_acc, pre_loss = g.at[i, "pre_acc"], g.at[i, "pre_loss"]
                for k in range(0, 4):
                    if i + k >= len(g):
                        break
                    rows.append({"from_70": pre_acc >= 70, "k": k, "next_phase": g.at[i + k, "phase"],
                                 "acc_change": g.at[i + k, "post_acc"] - pre_acc,
                                 "loss_change": g.at[i + k, "post_loss"] - pre_loss})
        traj = pd.DataFrame(rows)
        print("\n  [AFTER EACH CE UPDATE: the same model k of its own updates later (k=0 is the CE update), relative to"
              " just before the CE update]")
        for label, sub in [("all", traj), ("CE updates from >= 70%", traj[traj["from_70"]] if len(traj) else traj)]:
            if sub.empty:
                continue
            t = sub.groupby("k").agg(n=("acc_change", "size"), acc_change=("acc_change", "mean"),
                                     below_pre_CE=("acc_change", lambda x: (x < 0).mean()),
                                     loss_change=("loss_change", "mean"),
                                     loss_above_pre_CE=("loss_change", lambda x: (x > 0).mean()),
                                     share_KD=("next_phase", lambda x: (x == "kd").mean()))
            print(f"   {label}")
            print(indent(t.round(3).to_string(), 6))

    print("\n" + line)
    print(f"wrote {out / 'updates.csv'}")


if __name__ == "__main__":
    main()
