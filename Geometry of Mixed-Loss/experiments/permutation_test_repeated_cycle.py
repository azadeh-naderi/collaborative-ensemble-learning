"""
Permutation test for the repeated short-cycle experiment: is the observed
*timing* of a sustained Oracle-gain crossing (cumulative sum goes negative
and stays negative) unusually early, compared to what pure chance would
produce given the exact same per-exposure step sizes in a random order?

Key subtlety: whether a sequence crosses at all depends only on its final
cumulative sum (the total), which is order-invariant -- so if the total is
negative, EVERY reordering trivially "crosses" eventually, typically right
at the last step. That makes "does it cross" the wrong question. The
informative question is the crossing INDEX: does the true chronological
order cross unusually early (supporting a real, order-dependent/compounding
mechanism), or does it cross about as early as a random shuffle of the same
values would (consistent with pure chance given the step-size
distribution, not evidence of a real temporal effect)?

For each seed whose observed sequence crosses at all (final cumulative sum
< 0), this reshuffles the same CE-round Delta_acc values many times,
finds each permutation's crossing index, and reports the p-value: the
fraction of permutations that cross at least as early as the observed
index. A small p-value means the early crossing is unlikely by chance
(evidence of real order-dependence); a large p-value means it's easily
explained by chance alone.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/permutation_test_repeated_cycle.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --n_perm 2000
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as stats
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


def cumsum(deltas):
    out, running = [], 0.0
    for d in deltas:
        running += d
        out.append(running)
    return out


def crossing_index(cum):
    """0-indexed position of the first exposure after which the cumulative
    sum stays negative for the rest of the sequence. None if the final
    value is >= 0 (never sustains a crossing)."""
    if cum[-1] >= 0:
        return None
    n = len(cum)
    for i in range(n):
        if all(v < 0 for v in cum[i:]):
            return i
    return None  # unreachable given cum[-1] < 0, but kept for safety


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_k = load_by_k(Path(args.results_dir))
    if not by_k:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print("For seeds whose chronological sequence crosses at all (final sum < 0):")
    print("p-value = fraction of random reorderings of the SAME step values that")
    print("cross at least as early as the observed crossing exposure.\n")
    print(f"{'K':>4} | {'seed':>6} | {'final sum':>10} | {'observed crossing exp':>22} | {'p-value':>8}")
    print("-" * 62)

    p_values_by_k: dict[int, list[float]] = {}
    for k in sorted(by_k):
        for r in by_k[k]:
            deltas = ce_deltas(r)
            obs_cum = cumsum(deltas)
            obs_idx = crossing_index(obs_cum)
            if obs_idx is None:
                print(f"{k:>4} | {r['seed']:>6} | {obs_cum[-1]:>10.2f} | "
                      f"{'no crossing':>22} | {'--':>8}")
                continue

            d = list(deltas)
            n_as_early = 0
            for _ in range(args.n_perm):
                rng.shuffle(d)
                idx = crossing_index(cumsum(d))
                if idx is not None and idx <= obs_idx:
                    n_as_early += 1
            p = n_as_early / args.n_perm
            p_values_by_k.setdefault(k, []).append(p)
            print(f"{k:>4} | {r['seed']:>6} | {obs_cum[-1]:>10.2f} | "
                  f"{obs_idx + 1:>22} | {p:>8.3f}")

    print("\nSummary per K (only seeds that crossed at all):")
    print(f"{'K':>4} | {'n crossed':>9} | {'mean p-value':>12} | {'n with p<0.05':>13}")
    for k in sorted(p_values_by_k):
        ps = p_values_by_k[k]
        n_sig = sum(1 for p in ps if p < 0.05)
        print(f"{k:>4} | {len(ps):>9} | {stats.mean(ps):>12.3f} | {n_sig:>13}")

    print("\nInterpretation: a small p-value (e.g. <0.05) means the observed")
    print("crossing happened unusually early relative to what reordering the")
    print("same step sizes would produce -- evidence of a real, order-dependent")
    print("(compounding) effect. A large p-value means the timing is consistent")
    print("with pure chance given the step-size distribution alone.")


if __name__ == "__main__":
    main()
