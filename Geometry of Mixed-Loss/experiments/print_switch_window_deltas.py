"""
Prints round-by-round CE-phase Delta_acc (accuracy this round minus
accuracy the round before) for the first --n_window rounds after the
KL->CE switch, per seed and per dominance-variable group -- directly
answering "is the accuracy change negative right at the switch, and does
it recover or not," rather than relying on the raw accuracy trajectory
plot (accuracy_trajectory.pdf), which is too coarse in scale to show a
sub-1%-magnitude per-round dip against an 80-90% accuracy baseline.

Also prints the cumulative sum of that window (zeroed at the switch), so
you can see directly whether it climbs steadily, flattens, or declines --
the corrected signature from analyze_ce_phase_slope.py, shown here at
full per-round resolution instead of as a single slope number.

Works generically across:
    alpha-sweep:    --glob "alpha*_seed*" --group_field alpha
    repeated-cycle: --glob "k*_seed*"     --group_field k
    kd_finetune:    --glob "k*_seed*"     --group_field kd_epochs

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/print_switch_window_deltas.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --glob "alpha*_seed*" --group_field alpha --n_window 15
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", required=True, dest="glob_pattern")
    ap.add_argument("--group_field", required=True)
    ap.add_argument("--n_window", type=int, default=15,
                     help="number of CE-phase rounds after the switch to show")
    args = ap.parse_args()

    by_group = load_by_group(Path(args.results_dir), args.glob_pattern, args.group_field)
    if not by_group:
        raise SystemExit(f"No results.json files found under {args.results_dir} "
                          f"matching {args.glob_pattern!r}")

    for g in sorted(by_group):
        runs = by_group[g]
        print(f"\n=== {args.group_field} = {g} ({len(runs)} seeds) ===")
        print("Per-round Delta_acc for the first "
              f"{args.n_window} CE-phase rounds (columns = round 1..{args.n_window} "
              "since switch):")
        header = "seed".rjust(8) + "".join(f"{'r'+str(i):>8}" for i in range(1, args.n_window + 1))
        print(header)
        all_windows = []
        for r in runs:
            window = ce_deltas(r)[:args.n_window]
            all_windows.append(window)
            row = f"{r.get('seed', '?'):>8}" + "".join(f"{d:>8.3f}" for d in window)
            print(row)

        mean_row = []
        for i in range(args.n_window):
            vals = [w[i] for w in all_windows if i < len(w)]
            mean_row.append(stats.mean(vals) if vals else float("nan"))
        print("    mean" + "".join(f"{v:>8.3f}" for v in mean_row))

        cum = []
        running = 0.0
        for v in mean_row:
            running += v
            cum.append(running)
        print("cum(mean)" + "".join(f"{v:>8.3f}" for v in cum))

        n_neg_r1 = sum(1 for w in all_windows if len(w) > 0 and w[0] < 0)
        print(f"Round 1 after switch negative: {n_neg_r1}/{len(all_windows)} seeds")


if __name__ == "__main__":
    main()
