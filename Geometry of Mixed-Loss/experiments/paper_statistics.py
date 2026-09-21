"""Statistics reported in the paper that are not printed by the per-experiment analysis scripts.

  1. Original observation: CE and KD updates by accuracy in the 300-round CEL-Net logs (paper Table 1),
     and how often the KD teacher is more accurate than its student.
  2. KD-shaped vs CE-shaped networks, controlled setting, 10 seeds: per-seed CE-KD, exact permutation and sign-flip
     tests, accuracy-matched difference with a seed-bootstrap CI (paper Section 3.3, Table 3).
  3. Accumulation: slope of CE-KD on the number of intervening updates, with seed-bootstrap CIs (Section 3.4, Table 4).
  4. Training-label fit of the student before each oracle update (Section 4.1).
  5. Rotating peers: KD update outcome by teacher-minus-student accuracy (Section 6).

Run from the repo root. Reads the per-update CSVs written by the analysis scripts and the raw CEL-Net logs.
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("Geometry of Mixed-Loss/results")
LOGS = {"ClassDist, 9 learners": "results/300_rounds/ClassDist/slurm-978357.out",
        "MWM_ClassDist, 9 learners": "results/300_rounds/MWM_ClassDist/slurm-976409.out",
        "MWM_AccDiff, 19 learners": "results/300_rounds/MWM_AccDiff/20_models/20_ResNet18_mwm_AccDiff.1010747.out"}
ACC_BINS = [0, 50, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 100]


def sign_flip_p(d) -> float:
    d = np.asarray(d, float)
    obs = abs(d.mean())
    return float(np.mean([abs(np.dot(s, d) / len(d)) >= obs - 1e-12
                          for s in itertools.product([1, -1], repeat=len(d))]))


def perm_p(a, b) -> tuple[float, int, int]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, obs = np.r_[a, b], abs(a.mean() - b.mean())
    hits = total = 0
    for comb in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(comb)] = True
        total += 1
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
    return hits / total, hits, total


def header(title):
    print("\n" + "=" * 110 + "\n" + title)


# ---------------------------------------------------------------------------------------------- 1. CEL-Net logs
def parse_celnet_log(path: Path) -> pd.DataFrame:
    """One row per update: kind (CE/KD), accuracy before, gain, teacher accuracy (KD)."""
    r_round = re.compile(r"=== Round (\d+)/")
    r_or = re.compile(r"oracle-train: oracle -> student (\d+) : ([\d.]+)")
    r_og = re.compile(r"oracle-gain in this round: (-?[\d.]+)")
    r_kd = re.compile(r"kd-train: teacher (\d+) \(([\d.]+)\) -> student (\d+) \(([\d.]+)\)")
    rnd, events, obs = 0, [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := r_round.search(line):
            rnd = int(m.group(1))
        elif m := r_or.search(line):
            s, a = int(m.group(1)), float(m.group(2))
            events.append({"round": rnd, "model": s, "kind": "CE", "pre": a, "gain": np.nan, "teacher_acc": np.nan})
            obs.append((rnd, s, a))
        elif (m := r_og.search(line)) and events:
            events[-1]["gain"] = float(m.group(1))
        elif m := r_kd.search(line):
            t, ta, s, sa = int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
            events.append({"round": rnd, "model": s, "kind": "KD", "pre": sa, "gain": np.nan, "teacher_acc": ta})
            obs += [(rnd, t, ta), (rnd, s, sa)]
    ev, ob = pd.DataFrame(events), pd.DataFrame(obs, columns=["round", "model", "acc"])
    # KD gain = the student's next logged accuracy, if it did not train again in between
    for i, e in ev[ev["kind"] == "KD"].iterrows():
        later = ob[(ob["model"] == e["model"]) & (ob["round"] > e["round"])]
        next_train = ev[(ev["model"] == e["model"]) & (ev["round"] > e["round"])]["round"].min()
        if not later.empty and (np.isnan(next_train) or later.iloc[0]["round"] <= next_train):
            ev.at[i, "gain"] = later.iloc[0]["acc"] - e["pre"]
    return ev


def celnet_logs(repo: Path):
    header("1. ORIGINAL OBSERVATION: CEL-Net 300-round logs, mean gain per update by accuracy before it "
           "(CE and KD updates are given to different networks)")
    tables = {}
    for name, rel in LOGS.items():
        path = repo / rel
        if not path.exists():
            print(f"  missing {path}")
            continue
        ev = parse_celnet_log(path)
        ev["bin"] = pd.cut(ev["pre"], ACC_BINS)
        tables[name] = ev.dropna(subset=["gain"]).groupby(["bin", "kind"], observed=True)["gain"].mean().unstack("kind")
        kd = ev[ev["kind"] == "KD"]
        print(f"  {name}: KD teacher more accurate than student in {np.mean(kd['teacher_acc'] > kd['pre']):.0%} "
              f"of {len(kd)} KD updates")
    if tables:
        print(indent(pd.concat(tables, axis=1).round(2).to_string()))


# ---------------------------------------------------------------------------------------------- 2-4. controlled
def load_peer_events(csv: Path) -> pd.DataFrame:
    u = pd.read_csv(csv)
    u = u[u["policy"].isin(["mwm_accdiff", "mwm_accdiff_all_ce_cf"])].copy()
    u["cond"] = u["policy"].map({"mwm_accdiff": "KD-shaped", "mwm_accdiff_all_ce_cf": "CE-shaped"})
    u = u.sort_values(["cond", "seed", "student", "round"])
    since = {}
    for _, g in u.groupby(["cond", "seed", "student"]):
        count = None
        for idx, phase in zip(g.index, g["phase"]):
            if phase == "ce":
                since[idx] = count
                count = 0
            elif count is not None:
                count += 1
    u["since"] = pd.Series(since, dtype=float)
    ev = u[(u["phase"] == "ce") & u["cf_acc"].notna()].copy()
    ev["gap"] = ev["post_acc"] - ev["cf_acc"]
    ev["loss_gap"] = ev["post_loss"] - ev["cf_loss"]
    return ev


def seed_bootstrap(groups: dict, statistic, n_seeds: int, reps: int, rng) -> np.ndarray:
    conds = sorted({c for c, _ in groups})
    out = []
    for _ in range(reps):
        parts = [groups[(c, s)] for c in conds for s in rng.choice(n_seeds, n_seeds, replace=True)]
        out.append(statistic(pd.concat(parts)))
    return np.asarray(out)


def controlled(ev: pd.DataFrame, reps: int, rng):
    hi = ev[ev["pre_acc"] >= 70].copy()
    seeds = sorted(hi["seed"].unique())
    header("2. KD-SHAPED vs CE-SHAPED, controlled setting: CE minus KD from the same state, oracle updates from >= 70%")
    per_seed = hi.groupby(["cond", "seed"])["gap"].mean().unstack("cond")
    print(indent(per_seed.round(3).T.to_string()))
    a, b = per_seed["KD-shaped"].to_numpy(), per_seed["CE-shaped"].to_numpy()
    p, hits, total = perm_p(a, b)
    print(f"  KD-shaped mean {a.mean():+.3f} (sd {a.std(ddof=1):.3f}, range {a.min():+.2f} to {a.max():+.2f}), "
          f"{int((a < 0).sum())}/{len(a)} seeds negative, sign-flip p {sign_flip_p(a):.4f}")
    print(f"  CE-shaped mean {b.mean():+.3f} (sd {b.std(ddof=1):.3f}, range {b.min():+.2f} to {b.max():+.2f}), "
          f"{int((b < 0).sum())}/{len(b)} seeds negative, sign-flip p {sign_flip_p(b):.4f}")
    print(f"  difference {a.mean() - b.mean():+.3f}; exact permutation test p = {p:.2e} ({hits} of {total})")

    hi["bin"] = pd.cut(hi["pre_acc"], [70, 72.5, 75, 77.5, 80, 82.5, 85])
    tab = hi.groupby(["bin", "cond"], observed=True).agg(
        n=("gap", "size"), CE_minus_KD=("gap", "mean"), CE_worse=("gap", lambda x: (x < 0).mean()),
        loss_gap=("loss_gap", "mean")).unstack("cond")
    print("\n  accuracy-matched:")
    print(indent(tab.round(3).to_string()))

    def matched(d):
        w = d.groupby(["bin", "cond"], observed=True)["gap"].mean().unstack("cond").dropna()
        n = d.groupby("bin", observed=True).size().reindex(w.index)
        return np.average(w["KD-shaped"] - w["CE-shaped"], weights=n)

    groups = {(c, i): g for (c, s), g in hi.groupby(["cond", "seed"]) for i in [seeds.index(s)]}
    boots = seed_bootstrap(groups, matched, len(seeds), reps, rng)
    print(f"  accuracy-matched difference {matched(hi):+.3f}, seed-bootstrap 95% CI "
          f"[{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]  ({reps} resamples)")
    early = ev[(ev["pre_acc"] > 0) & (ev["pre_acc"] <= 50)].groupby("cond")["gap"].mean()   # bin (0, 50]
    print(f"  CE minus KD, accuracy before in (0, 50]: " + ", ".join(f"{c} {v:+.2f}" for c, v in early.items()))

    header("3. ACCUMULATION: CE minus KD vs number of updates since the previous oracle CE update (KD updates for "
           "KD-shaped, peer-slot CE for CE-shaped), oracle updates from >= 70%")
    hi["since_bin"] = pd.cut(hi["since"], [-1, 2, 4, 7, 12, 10_000], labels=["0-2", "3-4", "5-7", "8-12", "13+"])
    tab = hi.dropna(subset=["since"]).groupby(["cond", "since_bin"], observed=True).agg(
        n=("gap", "size"), mean_pre_acc=("pre_acc", "mean"), CE_minus_KD=("gap", "mean"),
        CE_worse=("gap", lambda x: (x < 0).mean()))
    print(indent(tab.round(3).to_string()))

    def slope(d):
        d = d.dropna(subset=["since"])
        x = np.column_stack([np.ones(len(d)), d["since"].clip(upper=15), d["pre_acc"]])
        return np.linalg.lstsq(x, d["gap"].to_numpy(float), rcond=None)[0][1]

    for cond in ("KD-shaped", "CE-shaped"):
        d = hi[hi["cond"] == cond]
        g = {(cond, seeds.index(s)): x for s, x in d.groupby("seed")}
        bs = seed_bootstrap(g, slope, len(seeds), reps, rng)
        print(f"  {cond}: slope {slope(d):+.4f} per update (controlling accuracy), seed-bootstrap 95% CI "
              f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    return hi


def training_fit(hi: pd.DataFrame):
    header("4. TRAINING-LABEL FIT before the oracle update (seeds with pre_train_acc logged), updates from >= 70%")
    m = hi[hi["pre_train_acc"].notna()].copy() if "pre_train_acc" in hi else pd.DataFrame()
    if m.empty:
        print("  pre_train_acc not logged")
        return
    print(f"  seeds: {sorted(m['seed'].unique())}")
    print(indent(m.groupby("cond").agg(n=("gap", "size"), val_acc=("pre_acc", "mean"),
                                       train_acc=("pre_train_acc", "mean"),
                                       train_loss=("pre_train_loss", "mean")).round(3).to_string()))
    m["vbin"] = pd.cut(m["pre_acc"], [70, 75, 77.5, 80, 85])
    print("  training accuracy at matched validation accuracy:")
    print(indent(m.groupby(["vbin", "cond"], observed=True)["pre_train_acc"].mean().unstack().round(2).to_string()))
    for cond, d in m.groupby("cond"):
        d = d.dropna(subset=["since"])
        b_fit = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["since"].clip(upper=15), d["pre_acc"]]),
                                d["pre_train_acc"].to_numpy(float), rcond=None)[0]
        b_old = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["since"].clip(upper=15), d["pre_acc"]]),
                                d["gap"].to_numpy(float), rcond=None)[0]
        b_new = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["since"].clip(upper=15), d["pre_train_acc"],
                                                 d["pre_acc"]]), d["gap"].to_numpy(float), rcond=None)[0]
        print(f"  {cond}: training accuracy {b_fit[1]:+.4f} per intervening update (at fixed val accuracy); "
              f"CE-KD slope {b_old[1]:+.4f} -> {b_new[1]:+.4f} when training accuracy is added "
              f"(coefficient {b_new[2]:+.3f} per training-accuracy point); n={len(d)}")


# ---------------------------------------------------------------------------------------------- 5. rotating peers
def rotating_peers(csv: Path):
    header("5. ROTATING PEERS (switch_peers): KD update outcome by teacher-minus-student accuracy, rounds > 20")
    if not csv.exists():
        print(f"  missing {csv}")
        return
    u = pd.read_csv(csv)
    kd = u[(u["phase"] == "kd") & (u["round"] > 20)].copy()
    kd["gap"] = kd["teacher_pre_acc"] - kd["pre_acc"]
    print(f"  teacher more accurate than student in {np.mean(kd['gap'] > 0):.0%} of {len(kd)} KD updates")
    kd["gap_bin"] = pd.cut(kd["gap"], [-100, -5, -2, 0, 2, 5, 100])
    print(indent(kd.groupby("gap_bin", observed=True)["delta_acc"].agg(
        n="size", mean="mean", drop_share=lambda x: (x < 0).mean()).round(2).to_string()))
    ce = u[(u["phase"] == "ce") & u["branch_kd_val_acc"].notna()]
    if not ce.empty:
        better = (ce["delta_acc"] > ce["branch_kd_val_acc"] - ce["pre_acc"]).mean()
        print(f"  CE update better than the KD counterfactual in {better:.0%} of {len(ce)} CE updates")


def indent(text: str) -> str:
    return "   " + text.replace("\n", "\n   ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(args.seed)
    repo = Path(".")
    celnet_logs(repo)
    ev = load_peer_events(RESULTS / "peers_celnet_pairing/analysis/updates.csv")
    hi = controlled(ev, args.reps, rng)
    training_fit(hi)
    rotating_peers(RESULTS / "switch_peers/analysis/updates.csv")


if __name__ == "__main__":
    main()
