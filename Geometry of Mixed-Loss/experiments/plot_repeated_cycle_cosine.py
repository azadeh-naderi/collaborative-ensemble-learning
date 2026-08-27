"""
Plots the full cos(g_CE, g_KL) trajectory across all CE-round exposures
(not just first-vs-last) for the repeated short-cycle experiment, one line
per K, mean across seeds -- to check whether the angle dips in the middle
of the run (coinciding with when sustained Oracle-gain crossings start)
even if it happens to recover by the final exposure, which a first-vs-last
comparison alone cannot distinguish from a monotonic increase.

Also marks, per K, the mean exposure index at which the Oracle-gain-style
cumulative sum first sustains a negative crossing (if any seed did), as a
vertical dotted line -- so the cosine dip (if any) can be checked directly
against when incompatibility actually starts.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/plot_repeated_cycle_cosine.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/analysis/ce_cosine_trajectory.pdf"
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_by_k(results_dir: Path):
    by_k: dict[int, list[dict]] = {}
    for run_dir in sorted(results_dir.glob("k*_seed*")):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_k.setdefault(data["k"], []).append(data)
    return by_k


def ce_round_cosines(data):
    """cos values logged specifically at CE rounds, in chronological
    (exposure) order."""
    return [e["cos"] for e in data["ce_kl_cosine"] if e.get("is_ce_round")]


def mean_crossing_exposure(runs):
    crossings = [r.get("oracle_gain_sustained_crossing_exposure") for r in runs]
    crossed = [c for c in crossings if c is not None]
    if not crossed:
        return None, 0, len(runs)
    return stats.mean(crossed), len(crossed), len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_k = load_by_k(results_dir)
    if not by_k:
        raise SystemExit(f"No results.json files found under {results_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma([i / max(1, len(by_k) - 1) for i in range(len(by_k))])

    for color, k in zip(colors, sorted(by_k)):
        runs = by_k[k]
        cos_lists = [ce_round_cosines(r) for r in runs]
        n_exposures = max(len(c) for c in cos_lists)
        means, stds = [], []
        for i in range(n_exposures):
            vals = [c[i] for c in cos_lists if i < len(c)]
            means.append(stats.mean(vals))
            stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)
        xs = list(range(1, n_exposures + 1))
        ax.plot(xs, means, color=color, lw=1.9, label=f"K={k}")
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)], color=color, alpha=0.12)

        mean_cross, n_crossed, n_total = mean_crossing_exposure(runs)
        if mean_cross is not None:
            ax.axvline(mean_cross, color=color, lw=1.2, ls=":", alpha=0.8)
            ax.annotate(f"K={k} crossing\n({n_crossed}/{n_total} seeds)",
                        xy=(mean_cross, ax.get_ylim()[1]), xytext=(mean_cross, 0),
                        rotation=90, fontsize=6.5, color=color, ha="right", va="bottom")

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("CE exposure # (chronological)")
    ax.set_ylabel(r"$\cos(g_{CE}, g_{KL})$ at CE round")
    ax.set_title("Repeated short-cycle: full cosine trajectory across CE exposures")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
