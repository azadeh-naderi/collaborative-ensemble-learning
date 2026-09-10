"""
Plots the FULL per-round Delta_acc trajectory (every round, both the KL
phase and the CE phase -- not just a truncated CE-phase window), one line
per dominance-variable group (alpha, or kd_epochs), mean +/- std across
seeds. This is the delta-scale complement to accuracy_trajectory.pdf:
that plot shows raw accuracy on an 80-90% axis, where any true per-round
dip (a fraction of a percent) is invisible; this plots the round-to-round
CHANGE directly, zoomed to a scale where a dip actually shows up.

X-axis is round position RELATIVE to the KL->CE switch (0 = the last KL
round, +1 = the first CE round, -1 = the second-to-last KL round, etc.),
found programmatically from the phase array (first index where
phase == "ce"), so this works whether every group shares the same switch
round (alpha-sweep: always round 100) or the switch position varies by
group (kd_finetune: switch is at kd_epochs, which differs per group).

Works generically across:
    alpha-sweep: --glob "alpha*_seed*" --group_field alpha
    kd_finetune: --glob "k*_seed*"     --group_field kd_epochs

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_full_delta_acc.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --glob "alpha*_seed*" --group_field alpha \
        --out "Geometry of Mixed-Loss/results/alpha_sweep/analysis/full_delta_acc.pdf"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_by_group(results_dir: Path, glob_pattern: str, group_field: str):
    by_group: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob(glob_pattern)):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_group.setdefault(data[group_field], []).append(data)
    return by_group


def relative_series(data):
    """Returns (relative_round, delta) pairs for every logged round.
    relative_round = 0 at the last KL round, +1 at the first CE round."""
    phase = data["phase"]
    deltas = data["delta_acc"]
    switch_idx = phase.index("ce")  # first CE-phase index
    return [(i - switch_idx + 1, d) for i, d in enumerate(deltas)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", required=True, dest="glob_pattern")
    ap.add_argument("--group_field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=None,
                     help="restrict plot to [-window, +window] relative rounds "
                          "(default: full range)")
    args = ap.parse_args()

    by_group = load_by_group(Path(args.results_dir), args.glob_pattern, args.group_field)
    if not by_group:
        raise SystemExit(f"No results.json files found under {args.results_dir} "
                          f"matching {args.glob_pattern!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = sorted(by_group)
    colors = plt.cm.plasma([i / max(1, len(groups) - 1) for i in range(len(groups))])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, g in zip(colors, groups):
        runs = by_group[g]
        by_rel: dict[int, list[float]] = {}
        for r in runs:
            for rel, d in relative_series(r):
                by_rel.setdefault(rel, []).append(d)
        rels = sorted(by_rel)
        if args.window is not None:
            rels = [r for r in rels if -args.window <= r <= args.window]
        means = [stats.mean(by_rel[r]) for r in rels]
        stds = [stats.pstdev(by_rel[r]) if len(by_rel[r]) > 1 else 0.0 for r in rels]
        ax.plot(rels, means, color=color, lw=1.6,
                label=f"{args.group_field}={g}")
        ax.fill_between(rels, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)], color=color, alpha=0.10)

    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(0.5, color="gray", lw=1.2, ls="--")
    ax.annotate("switch (KL|CE)", xy=(0.5, ax.get_ylim()[1]), xytext=(2, -12),
                textcoords="offset points", fontsize=8, color="gray")
    ax.set_xlabel("Round relative to switch (0 = last KL round, +1 = first CE round)")
    ax.set_ylabel(r"$\Delta$acc (this round $-$ previous round)")
    ax.set_title(f"Full per-round accuracy delta, by {args.group_field} "
                  "(mean ± std across seeds)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
