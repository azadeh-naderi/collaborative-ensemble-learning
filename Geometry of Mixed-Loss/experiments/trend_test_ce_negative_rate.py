"""
Tests whether the RATE of individual negative CE-round accuracy deltas
(not a cumulative sum, not a sustained crossing) increases with K in the
repeated short-cycle experiment -- the correct single-model analog of
CEL-Net's own Oracle-gain-negative event (a single isolated CE round
producing a negative accuracy delta), per the note added to
sec:discovery: Oracle gain is a sum of many isolated single-round CE
deltas, and what matters is whether an individual round goes negative,
not whether a running total ever permanently dips below zero.

For each seed, computes the fraction of its 20 CE-round exposures with
delta_acc < 0. Tests whether this per-seed rate trends upward with K via
a permutation test: shuffle which K label is attached to which seed's
rate (this respects the non-independence of the 20 exposures within a
seed by using the seed-level rate, not the raw exposure, as the unit of
analysis), recompute the Pearson correlation between K and rate, and see
how often a random shuffle produces a correlation at least as strong as
the true (K, rate) pairing.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/trend_test_ce_negative_rate.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --n_perm 200000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_by_k(results_dir: Path):
    by_k: dict[int, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("k*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_k.setdefault(data["k"], []).append(data)
    return by_k


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
    by_k = load_by_k(Path(args.results_dir))
    if not by_k:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print(f"{'K':>4} | {'seeds':>5} | {'mean neg-exposure rate':>22} | {'pooled n_neg/n_total':>20}")
    print("-" * 62)
    ks, rates = [], []
    for k in sorted(by_k):
        runs = by_k[k]
        per_seed_rates = [negative_rate(r) for r in runs]
        pooled_neg = sum(round(rate * 20) for rate in per_seed_rates)
        pooled_total = 20 * len(runs)
        mean_rate = sum(per_seed_rates) / len(per_seed_rates)
        print(f"{k:>4} | {len(runs):>5} | {mean_rate:>22.3f} | {pooled_neg:>7}/{pooled_total}")
        for rate in per_seed_rates:
            ks.append(k)
            rates.append(rate)

    obs_r = pearson(ks, rates)
    rates_shuf = list(rates)
    n_ge = 0
    for _ in range(args.n_perm):
        rng.shuffle(rates_shuf)
        if pearson(ks, rates_shuf) >= obs_r:
            n_ge += 1
    p = n_ge / args.n_perm

    print(f"\nPearson r(K, per-seed negative-exposure rate) = {obs_r:.3f}")
    print(f"Permutation p-value (one-sided, H1: rate increases with K) = {p:.5f}")
    print(f"  ({args.n_perm} shuffles of the K<->rate pairing across "
          f"{len(ks)} seeds)")


if __name__ == "__main__":
    main()
