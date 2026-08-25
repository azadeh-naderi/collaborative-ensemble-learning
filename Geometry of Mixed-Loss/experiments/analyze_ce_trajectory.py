"""
Follow-up analysis for Exp 1 (alpha-sweep): looks past the noisy single-round
T* metric to the full CE-phase trajectory.

The original run_alpha_sweep.py defines T* as the first CE round with any
negative Delta_acc -- a single noisy comparison that round-to-round eval
jitter can trigger regardless of alpha. This script instead computes, per
(alpha, seed):

  1. The cumulative Delta_acc curve over all CE rounds (the same "cumulative
     gain" quantity used for Oracle gain in Figure 1 of the paper) -- a much
     more robust signal of whether the CE phase helped or hurt overall.
  2. A "sustained" T*: the first round after which a W-round rolling mean of
     Delta_acc goes negative and STAYS negative for the rest of the CE phase
     (i.e. a permanent regression, not a transient dip).

It prints, per alpha (mean +/- std across seeds):
  - final cumulative Delta_acc at CE round 60
  - sustained T* (or "never" if the rolling mean never goes and stays negative)
  - the mean cumulative curve at checkpoints every 10 rounds, for eyeballing
    the shape directly in text.

It also saves a plot (mean cumulative curve +/- std band, one line per alpha)
to <results_dir>/analysis/ce_cumulative.pdf if matplotlib is available.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/analyze_ce_trajectory.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --window 5
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load_by_alpha(results_dir: Path):
    by_alpha: dict[float, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("alpha*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        data["_ce_deltas"] = [d for d, p in zip(data["delta_acc"], data["phase"]) if p == "ce"]
        by_alpha.setdefault(data["alpha"], []).append(data)
    return by_alpha


def cumulative(deltas):
    out, running = [], 0.0
    for d in deltas:
        running += d
        out.append(running)
    return out


def rolling_mean(xs, w):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def sustained_t_star(deltas, window):
    """First round (1-indexed) after which the W-round rolling mean of
    Delta_acc is negative for every remaining round. None if that never
    happens (either always positive, or negative dips that later recover)."""
    roll = rolling_mean(deltas, window)
    n = len(roll)
    for r in range(n):
        if all(v < 0 for v in roll[r:]):
            return r + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--window", type=int, default=5,
                     help="rolling window (rounds) for the sustained-onset definition")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_alpha = load_by_alpha(results_dir)

    print(f"{'alpha':>6} | {'seeds':>5} | {'final cum. Delta_acc (60 CE rounds)':>36} | "
          f"{'sustained T* (w=' + str(args.window) + ')':>20}")
    print("-" * 90)

    curves = {}  # alpha -> list of cumulative curves (one per seed)
    for alpha in sorted(by_alpha):
        runs = by_alpha[alpha]
        finals, t_stars, seed_curves = [], [], []
        for r in runs:
            cum = cumulative(r["_ce_deltas"])
            seed_curves.append(cum)
            finals.append(cum[-1] if cum else 0.0)
            t_stars.append(sustained_t_star(r["_ce_deltas"], args.window))
        curves[alpha] = seed_curves

        final_str = f"{stats.mean(finals):+.2f} +/- {stats.pstdev(finals) if len(finals) > 1 else 0.0:.2f}"
        n_sustained = sum(1 for t in t_stars if t is not None)
        if n_sustained == 0:
            t_str = "never (0/{})".format(len(t_stars))
        else:
            vals = [t for t in t_stars if t is not None]
            t_str = f"{stats.mean(vals):.1f} ({n_sustained}/{len(t_stars)} seeds)"
        print(f"{alpha:>6} | {len(runs):>5} | {final_str:>36} | {t_str:>20}")

    print(f"\nMean cumulative Delta_acc at checkpoints (avg across seeds, per alpha):")
    header = "alpha".rjust(6) + "".join(f"{'r'+str(c):>10}" for c in args.checkpoints)
    print(header)
    for alpha in sorted(by_alpha):
        seed_curves = curves[alpha]
        row = f"{alpha:>6}"
        for c in args.checkpoints:
            vals = [sc[min(c, len(sc)) - 1] for sc in seed_curves if sc]
            row += f"{stats.mean(vals):>10.2f}" if vals else f"{'--':>10}"
        print(row)

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = results_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for alpha in sorted(by_alpha):
            seed_curves = curves[alpha]
            max_len = max(len(c) for c in seed_curves)
            means, stds = [], []
            for i in range(max_len):
                vals = [c[i] for c in seed_curves if i < len(c)]
                means.append(stats.mean(vals))
                stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)
            xs = list(range(1, max_len + 1))
            ax.plot(xs, means, label=f"alpha={alpha}")
            ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)], alpha=0.15)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("CE round")
        ax.set_ylabel("Cumulative Delta_acc (%) since CE phase start")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "ce_cumulative.pdf")
        print(f"\nSaved plot to {out_dir / 'ce_cumulative.pdf'}")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
