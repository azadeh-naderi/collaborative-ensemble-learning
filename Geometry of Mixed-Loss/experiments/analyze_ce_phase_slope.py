"""
Correct operationalization of "CE fine-tuning becomes harmful once KL
dominance is entrenched": the cumulative CE-phase accuracy-gain curve
should trend DOWNWARD (its slope should worsen) as the dominance variable
grows -- NOT that the cumulative total ever crosses below zero, and not
just a coarse "final sum" or "did any single round go negative" count.
This project already found the "crosses zero" operationalization to be
the wrong one once (see permutation_test_repeated_cycle.py); this script
applies the corrected, slope-based version consistently across all three
experiments that share the same delta_acc/phase log schema.

For each seed: take the CE-phase per-round deltas (already logged every
round in run_alpha_sweep.py, run_repeated_cycle.py, run_kd_finetune.py --
no re-run needed), form their cumulative sum, and fit a simple linear
regression of that cumulative sum against the CE-phase round/exposure
index. The slope is the per-seed statistic: a slope near the per-round
mean delta (extrapolated flat) means steady, undiminished CE benefit; a
slope that drops as the dominance variable grows means the *rate* at
which CE helps is declining, which is the real signature of
incompatibility setting in, and is orthogonal to whether the cumulative
total ever dips below zero.

Tests, via permutation, whether the per-seed slope correlates negatively
with the dominance variable (H1: slope decreases as dominance grows).

Works generically across:
    alpha-sweep:    --glob "alpha*_seed*" --group_field alpha
    repeated-cycle: --glob "k*_seed*"     --group_field k
    kd_finetune:    --glob "k*_seed*"     --group_field kd_epochs

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/analyze_ce_phase_slope.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --glob "alpha*_seed*" --group_field alpha --n_perm 200000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_by_group(results_dir: Path, glob_pattern: str, group_field: str):
    by_group: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob(glob_pattern)):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_group.setdefault(data[group_field], []).append(data)
    return by_group


def ce_deltas(data):
    return [d for d, p in zip(data["delta_acc"], data["phase"]) if p == "ce"]


def cumsum(deltas):
    out, running = [], 0.0
    for d in deltas:
        running += d
        out.append(running)
    return out


def ols_slope(ys):
    n = len(ys)
    xs = list(range(1, n + 1))
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    return cov / vx if vx > 0 else 0.0


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", required=True, dest="glob_pattern")
    ap.add_argument("--group_field", required=True)
    ap.add_argument("--n_perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_group = load_by_group(Path(args.results_dir), args.glob_pattern, args.group_field)
    if not by_group:
        raise SystemExit(f"No results.json files found under {args.results_dir} "
                          f"matching {args.glob_pattern!r}")

    print(f"{args.group_field:>10} | {'seeds':>5} | {'mean slope':>11} | {'mean final sum':>14}")
    print("-" * 52)
    gs, slopes = [], []
    for g in sorted(by_group):
        runs = by_group[g]
        seed_slopes = [ols_slope(cumsum(ce_deltas(r))) for r in runs]
        seed_finals = [cumsum(ce_deltas(r))[-1] for r in runs]
        mean_slope = sum(seed_slopes) / len(seed_slopes)
        mean_final = sum(seed_finals) / len(seed_finals)
        print(f"{g:>10} | {len(runs):>5} | {mean_slope:>11.5f} | {mean_final:>14.3f}")
        for s in seed_slopes:
            gs.append(g)
            slopes.append(s)

    obs_r = pearson(gs, slopes)
    slopes_shuf = list(slopes)
    n_le = 0
    for _ in range(args.n_perm):
        rng.shuffle(slopes_shuf)
        if pearson(gs, slopes_shuf) <= obs_r:
            n_le += 1
    p = n_le / args.n_perm

    print(f"\nPearson r({args.group_field}, per-seed cumulative-slope) = {obs_r:.3f}")
    print(f"Permutation p-value (one-sided, H1: slope decreases as "
          f"{args.group_field} grows) = {p:.5f}")
    print(f"  ({args.n_perm} shuffles across {len(gs)} seeds)")
    print("\nInterpretation: a negative r with small p means the cumulative CE-phase")
    print("gain curve trends progressively less steep (or downward) as the dominance")
    print("variable grows -- the corrected 'becomes harmful' signature. This does NOT")
    print("require the cumulative total (mean final sum column) to go negative.")


if __name__ == "__main__":
    main()
