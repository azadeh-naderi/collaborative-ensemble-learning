"""
Exp 1 (alpha-sweep) analog of trend_test_ce_negative_rate.py: tests
whether the RATE of individual negative-delta CE rounds during the fixed
60-round CE phase increases with alpha, rather than relying only on the
coarser statistics (t_star = first negative round, cumulative Delta_CE)
already reported for Exp 1. This applies the same correction made to the
repeated short-cycle experiment -- per-round harm rate, not a cumulative
sum -- to data that is already fully collected, no re-run needed.

Works on either teacher condition's results dir:
    results/alpha_sweep              (CE-pretrained teacher)
    results/alpha_sweep_kl_teacher   (KL-self-distilled teacher)

For each seed, computes the fraction of the 60 CE-phase rounds with
delta_acc < 0. Tests whether this per-seed rate trends upward with alpha
via a permutation test: shuffle which alpha label is attached to which
seed's rate, recompute the Pearson correlation, and see how often a
random shuffle produces a correlation at least as strong as the true
(alpha, rate) pairing.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/trend_test_alpha_sweep_negative_rate.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --n_perm 200000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_by_alpha(results_dir: Path):
    by_alpha: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("alpha*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_alpha.setdefault(data["alpha"], []).append(data)
    return by_alpha


def ce_deltas(data):
    return [d for d, p in zip(data["delta_acc"], data["phase"]) if p == "ce"]


def negative_rate(data):
    deltas = ce_deltas(data)
    return sum(1 for d in deltas if d < 0) / len(deltas)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--n_perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_alpha = load_by_alpha(Path(args.results_dir))
    if not by_alpha:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print(f"{'alpha':>7} | {'seeds':>5} | {'mean neg-round rate':>20} | {'pooled n_neg/n_total':>20}")
    print("-" * 65)
    xs, rates = [], []
    for a in sorted(by_alpha):
        runs = by_alpha[a]
        n_ce = len(ce_deltas(runs[0]))
        per_seed_rates = [negative_rate(r) for r in runs]
        pooled_neg = sum(round(rate * n_ce) for rate in per_seed_rates)
        pooled_total = n_ce * len(runs)
        mean_rate = sum(per_seed_rates) / len(per_seed_rates)
        print(f"{a:>7} | {len(runs):>5} | {mean_rate:>20.3f} | {pooled_neg:>7}/{pooled_total}")
        for rate in per_seed_rates:
            xs.append(a)
            rates.append(rate)

    obs_r = pearson(xs, rates)
    rates_shuf = list(rates)
    n_ge = 0
    for _ in range(args.n_perm):
        rng.shuffle(rates_shuf)
        if pearson(xs, rates_shuf) >= obs_r:
            n_ge += 1
    p = n_ge / args.n_perm

    print(f"\nPearson r(alpha, per-seed negative-round rate) = {obs_r:.3f}")
    print(f"Permutation p-value (one-sided, H1: rate increases with alpha) = {p:.5f}")
    print(f"  ({args.n_perm} shuffles of the alpha<->rate pairing across "
          f"{len(xs)} seeds)")


if __name__ == "__main__":
    main()
