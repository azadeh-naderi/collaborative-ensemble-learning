"""
Exp 2 (KD Pre-Training Depth) analog of trend_test_ce_negative_rate.py:
tests whether the RATE of individual negative-delta CE epochs during the
fixed 60-epoch CE fine-tune phase increases with KD pre-training depth
(kd_epochs), rather than relying on the coarser statistics
(t_star = first negative epoch, initial_ce_delta) already saved by
run_kd_finetune.py. This mirrors the correction made to the repeated
short-cycle experiment: the per-epoch harm rate, not a single summary
number, is the correct single-model analog of CEL-Net's own
Oracle-gain-negative event.

For each seed, computes the fraction of the 60 CE-phase epochs with
delta_acc < 0. Tests whether this per-seed rate trends upward with
kd_epochs via a permutation test: shuffle which kd_epochs label is
attached to which seed's rate, recompute the Pearson correlation, and see
how often a random shuffle produces a correlation at least as strong as
the true (kd_epochs, rate) pairing.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/trend_test_kd_finetune_negative_rate.py" \
        --results_dir "Geometry of Mixed-Loss/results/kd_finetune" \
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
        by_k.setdefault(data["kd_epochs"], []).append(data)
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

    print(f"{'KD epochs':>10} | {'seeds':>5} | {'mean neg-epoch rate':>20} | {'pooled n_neg/n_total':>20}")
    print("-" * 68)
    ks, rates = [], []
    for k in sorted(by_k):
        runs = by_k[k]
        n_ce = len(ce_deltas(runs[0]))
        per_seed_rates = [negative_rate(r) for r in runs]
        pooled_neg = sum(round(rate * n_ce) for rate in per_seed_rates)
        pooled_total = n_ce * len(runs)
        mean_rate = sum(per_seed_rates) / len(per_seed_rates)
        print(f"{k:>10} | {len(runs):>5} | {mean_rate:>20.3f} | {pooled_neg:>7}/{pooled_total}")
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

    print(f"\nPearson r(kd_epochs, per-seed negative-epoch rate) = {obs_r:.3f}")
    print(f"Permutation p-value (one-sided, H1: rate increases with kd_epochs) = {p:.5f}")
    print(f"  ({args.n_perm} shuffles of the kd_epochs<->rate pairing across "
          f"{len(ks)} seeds)")


if __name__ == "__main__":
    main()
