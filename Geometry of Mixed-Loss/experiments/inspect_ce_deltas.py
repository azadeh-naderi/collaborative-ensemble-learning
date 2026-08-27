"""
Reports, per seed, the raw per-exposure CE-round accuracy delta
(Delta_acc) for the repeated short-cycle experiment -- not the
cumulative sum, and not just whether/when it sustains a negative
crossing, but whether ANY individual CE round itself produced a
negative accuracy delta (the direct analog of CEL-Net's single-round
Oracle-gain-negative event).

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/inspect_ce_deltas.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle"
"""
from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    by_k = load_by_k(Path(args.results_dir))
    if not by_k:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print(f"{'K':>4} | {'seed':>6} | {'n neg CE exposures':>18} | "
          f"{'/ 20':>5} | {'min delta':>10} | {'exposure #s (neg)'}")
    print("-" * 90)

    total_neg = 0
    total_exposures = 0
    for k in sorted(by_k):
        for r in by_k[k]:
            deltas = ce_deltas(r)
            neg_idxs = [i + 1 for i, d in enumerate(deltas) if d < 0]
            min_d = min(deltas) if deltas else float("nan")
            total_neg += len(neg_idxs)
            total_exposures += len(deltas)
            print(f"{k:>4} | {r['seed']:>6} | {len(neg_idxs):>18} | "
                  f"{len(deltas):>5} | {min_d:>10.3f} | {neg_idxs}")

    print("-" * 90)
    print(f"Total negative CE-round exposures across all seeds/K: "
          f"{total_neg} / {total_exposures} "
          f"({100 * total_neg / total_exposures:.1f}%)")


if __name__ == "__main__":
    main()
