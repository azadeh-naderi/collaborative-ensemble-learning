"""
Sharper version of trend_test_kd_finetune_negative_rate.py: restricts to
just the first --n_early CE epochs of the 60-epoch CE fine-tune phase,
instead of pooling all 60.

Rationale: in Exp 2's design, only CE epoch #1 is actually preceded by
the KD phase -- CE epochs 2..60 are each preceded by the *previous CE
epoch*, not by K rounds of KL, so they are not structurally analogous to
repeated-cycle's "every CE round follows K KL rounds" design. Pooling all
60 dilutes any real KD-depth-driven signal with ~59 epochs of ordinary
CE-training noise that has no reason to depend on K. Restricting to the
first few epochs closest to the KD->CE switch is a sharper test of the
same dominance mechanism.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/trend_test_kd_finetune_early_ce_negative_rate.py" \
        --results_dir "Geometry of Mixed-Loss/results/kd_finetune" \
        --n_early 1 --n_perm 200000
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


def ce_deltas(data, n_early):
    all_ce = [d for d, p in zip(data["delta_acc"], data["phase"]) if p == "ce"]
    return all_ce[:n_early]


def negative_rate(data, n_early):
    deltas = ce_deltas(data, n_early)
    return sum(1 for d in deltas if d < 0) / len(deltas)


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
    ap.add_argument("--n_early", type=int, default=1,
                     help="number of CE epochs from the start of the CE phase to use")
    ap.add_argument("--n_perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_k = load_by_k(Path(args.results_dir))
    if not by_k:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print(f"First {args.n_early} CE epoch(s) after the KD->CE switch:\n")
    print(f"{'KD epochs':>10} | {'seeds':>5} | {'mean neg rate':>13} | {'pooled n_neg/n_total':>20} | {'mean delta':>10}")
    print("-" * 75)
    ks, rates = [], []
    for k in sorted(by_k):
        runs = by_k[k]
        per_seed_rates = [negative_rate(r, args.n_early) for r in runs]
        all_deltas = [d for r in runs for d in ce_deltas(r, args.n_early)]
        mean_delta = sum(all_deltas) / len(all_deltas)
        n_used = len(ce_deltas(runs[0], args.n_early))
        pooled_neg = sum(round(rate * n_used) for rate in per_seed_rates)
        pooled_total = n_used * len(runs)
        mean_rate = sum(per_seed_rates) / len(per_seed_rates)
        print(f"{k:>10} | {len(runs):>5} | {mean_rate:>13.3f} | {pooled_neg:>7}/{pooled_total:<12} | {mean_delta:>+10.4f}")
        for rate in per_seed_rates:
            ks.append(k)
            rates.append(rate)

    if len(set(rates)) > 1:
        obs_r = pearson(ks, rates)
        rates_shuf = list(rates)
        n_ge = 0
        for _ in range(args.n_perm):
            rng.shuffle(rates_shuf)
            if pearson(ks, rates_shuf) >= obs_r:
                n_ge += 1
        p = n_ge / args.n_perm
        print(f"\nPearson r(kd_epochs, per-seed negative rate) = {obs_r:.3f}")
        print(f"Permutation p-value (one-sided, H1: rate increases with kd_epochs) = {p:.5f}")
    else:
        print("\nAll per-seed rates identical (likely n_early=1, binary outcome) -- "
              "Pearson correlation degenerate; read the mean-delta column instead.")


if __name__ == "__main__":
    main()
