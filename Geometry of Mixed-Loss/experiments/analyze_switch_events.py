"""
Comprehensive switch-event analysis, operating on the CSVs produced by
export_full_delta_acc_csv.py. Reorganizes the analysis around the
corrected metrics established for this project:

  1. Does NOT use the raw accuracy trajectory (too coarse a scale to show
     a sub-1%-magnitude per-round effect against an 80-90% baseline).
  2. Per-round accuracy delta right at the switch (and immediately after):
     is it negative right at the switch, and does it recover?
  3. Cumulative CE-phase delta slope: does the cumulative curve trend
     DOWNWARD as the dominance variable grows (the correct signature),
     not whether it crosses below zero.
  4. Gradient-direction (cosine) change right at the switch, at full
     per-round resolution (requires cos_ce_kl to be densely populated --
     i.e. a CSV exported after the GRAD_LOG_INTERVAL=1 fix + re-run).
  5. T* (first negative post-switch round) vs the dominance variable.

A "switch event" is detected generically as any CE-phase round whose
immediately preceding round was KL/KD-phase -- this is exactly 1 event
per run for alpha-sweep/kd_finetune (single long block), and exactly
N_EXPOSURES events per run for repeated-cycle (every CE round is
preceded by a fresh KL stretch), so the same code handles both designs.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/analyze_switch_events.py" \
        --csv "Geometry of Mixed-Loss/results/alpha_sweep/analysis/full_delta_acc.csv" \
        --n_perm 200000
"""
from __future__ import annotations

import argparse
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

import csv as csv_mod


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        rows = list(reader)
    return rows


def to_float(x):
    return float(x) if x not in (None, "") else None


def group_runs(rows):
    """Returns {(group, seed): [row, row, ...]} sorted by round_index."""
    runs = defaultdict(list)
    for r in rows:
        key = (r["group"], r["seed"])
        runs[key].append(r)
    for key in runs:
        runs[key].sort(key=lambda r: int(r["round_index"]))
    return runs


def switch_events(run_rows, n_after=5):
    """Yields dicts describing each switch event in this run:
    delta_at_switch, cos_before, cos_after, cos_dip, deltas_after (list of
    up to n_after post-switch deltas for recovery tracking)."""
    events = []
    for i in range(1, len(run_rows)):
        prev_phase = run_rows[i - 1]["phase"]
        cur_phase = run_rows[i]["phase"]
        if cur_phase == "ce" and prev_phase != "ce":
            delta_switch = to_float(run_rows[i]["delta_acc"])
            cos_before = to_float(run_rows[i - 1]["cos_ce_kl"])
            cos_after = to_float(run_rows[i]["cos_ce_kl"])
            deltas_after = [to_float(run_rows[j]["delta_acc"])
                             for j in range(i, min(i + n_after, len(run_rows)))]
            events.append({
                "delta_at_switch": delta_switch,
                "cos_before": cos_before,
                "cos_after": cos_after,
                "cos_dip": (cos_before - cos_after) if (cos_before is not None and cos_after is not None) else None,
                "deltas_after": deltas_after,
            })
    return events


def ce_phase_deltas(run_rows):
    return [to_float(r["delta_acc"]) for r in run_rows if r["phase"] == "ce"]


def ols_slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(1, n + 1))
    mx, my = sum(xs) / n, sum(ys) / n
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


def permutation_test(xs, ys, n_perm, rng, direction):
    """direction: 'increase' tests H1 ys correlate positively with xs;
    'decrease' tests H1 ys correlate negatively with xs."""
    obs_r = pearson(xs, ys)
    ys_shuf = list(ys)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_shuf)
        r = pearson(xs, ys_shuf)
        if direction == "increase":
            if r >= obs_r:
                count += 1
        else:
            if r <= obs_r:
                count += 1
    return obs_r, count / n_perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_after", type=int, default=5,
                     help="how many post-switch rounds to show for recovery tracking")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(Path(args.csv))
    runs = group_runs(rows)

    # organize by group
    by_group_events = defaultdict(list)     # group -> list of event dicts (pooled across seeds)
    by_group_seed_events = defaultdict(list)  # group -> list of (seed, [events])
    by_group_seed_slope = defaultdict(list)   # group -> list of per-seed cumulative slope
    by_group_seed_tstar = defaultdict(list)   # group -> list of per-seed T* (None if never negative)

    has_cosine = False

    for (group, seed), run_rows in runs.items():
        evs = switch_events(run_rows, n_after=args.n_after)
        by_group_events[group].extend(evs)
        by_group_seed_events[group].append((seed, evs))

        ce_deltas = ce_phase_deltas(run_rows)
        cum = []
        running = 0.0
        for d in ce_deltas:
            running += d
            cum.append(running)
        by_group_seed_slope[group].append(ols_slope(cum))

        t_star = next((i + 1 for i, d in enumerate(ce_deltas) if d is not None and d < 0), None)
        by_group_seed_tstar[group].append(t_star)

        if any(e["cos_before"] is not None for e in evs):
            has_cosine = True

    groups_sorted = sorted(by_group_events, key=lambda g: float(g))

    # ---- Point 2: delta right at the switch, and recovery ----
    print("=" * 100)
    print("POINT 2: per-round accuracy delta right at the switch (mean across all switch events)")
    print("=" * 100)
    header = f"{'group':>8} | {'n_events':>8} | {'mean d@switch':>14} | {'frac neg @switch':>16} | " + \
             "".join(f"{'d+'+str(i):>9}" for i in range(1, args.n_after))
    print(header)
    for g in groups_sorted:
        evs = by_group_events[g]
        d0 = [e["delta_at_switch"] for e in evs if e["delta_at_switch"] is not None]
        mean_d0 = stats.mean(d0)
        frac_neg = sum(1 for d in d0 if d < 0) / len(d0)
        recovery_means = []
        for i in range(1, args.n_after):
            vals = [e["deltas_after"][i] for e in evs if len(e["deltas_after"]) > i and e["deltas_after"][i] is not None]
            recovery_means.append(stats.mean(vals) if vals else float("nan"))
        row = f"{g:>8} | {len(evs):>8} | {mean_d0:>+14.4f} | {frac_neg:>16.3f} | " + \
              "".join(f"{v:>+9.4f}" for v in recovery_means)
        print(row)

    # ---- Point 3: cumulative CE-phase slope, trend vs group ----
    print("\n" + "=" * 100)
    print("POINT 3: cumulative CE-phase delta slope (decreasing = trending down, NOT crossing zero)")
    print("=" * 100)
    xs, slopes = [], []
    print(f"{'group':>8} | {'seeds':>5} | {'mean slope':>11}")
    for g in groups_sorted:
        vals = by_group_seed_slope[g]
        print(f"{g:>8} | {len(vals):>5} | {stats.mean(vals):>11.5f}")
        for v in vals:
            xs.append(float(g))
            slopes.append(v)
    r, p = permutation_test(xs, slopes, args.n_perm, rng, direction="decrease")
    print(f"\nPearson r(group, slope) = {r:.3f}")
    print(f"Permutation p (one-sided, H1: slope decreases as group grows) = {p:.5f}")

    # ---- Point 4: gradient-direction (cosine) change at the switch ----
    print("\n" + "=" * 100)
    if has_cosine:
        print("POINT 4: cos(g_CE, g_KL) change right at the switch (cos_before - cos_after)")
    else:
        print("POINT 4: SKIPPED -- cos_ce_kl is empty/sparse in this CSV (pre-fix data)")
    print("=" * 100)
    if has_cosine:
        xs2, dips = [], []
        print(f"{'group':>8} | {'n_events':>8} | {'mean cos_before':>15} | {'mean cos_after':>14} | {'mean dip':>9}")
        for g in groups_sorted:
            evs = by_group_events[g]
            cb = [e["cos_before"] for e in evs if e["cos_before"] is not None]
            ca = [e["cos_after"] for e in evs if e["cos_after"] is not None]
            dip = [e["cos_dip"] for e in evs if e["cos_dip"] is not None]
            print(f"{g:>8} | {len(evs):>8} | {stats.mean(cb):>15.4f} | {stats.mean(ca):>14.4f} | {stats.mean(dip):>9.4f}")
            for seed, seed_evs in by_group_seed_events[g]:
                seed_dips = [e["cos_dip"] for e in seed_evs if e["cos_dip"] is not None]
                if seed_dips:
                    xs2.append(float(g))
                    dips.append(stats.mean(seed_dips))
        r2, p2 = permutation_test(xs2, dips, args.n_perm, rng, direction="increase")
        print(f"\nPearson r(group, mean per-seed cos_dip) = {r2:.3f}")
        print(f"Permutation p (one-sided, H1: dip increases as group grows) = {p2:.5f}")

    # ---- Point 5: T* vs group ----
    print("\n" + "=" * 100)
    print("POINT 5: T* (first negative CE-phase round) vs group")
    print("=" * 100)
    print(f"{'group':>8} | {'seeds':>5} | {'n incompatible (T* found)':>26} | {'mean T*':>8}")
    for g in groups_sorted:
        tstars = by_group_seed_tstar[g]
        found = [t for t in tstars if t is not None]
        mean_t = stats.mean(found) if found else float("nan")
        print(f"{g:>8} | {len(tstars):>5} | {len(found):>26}/{len(tstars):<3} | {mean_t:>8.1f}")


if __name__ == "__main__":
    main()
