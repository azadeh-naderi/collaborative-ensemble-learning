"""
Checks the CE-KL gradient cosine right at the switch (round 100, end of KL
phase, vs round 105, 5 rounds into CE) -- a direct test of the hypothesis
that the model is being pushed toward a KL-favorable point and a CE step
pulls it in the opposing direction (Proposition 1's cos(g_CE,g_KL) < 0
condition), even though Exp 1's accuracy showed no sustained drop.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/print_cosine_at_switch.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep"
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
        by_alpha.setdefault(data["alpha"], []).append(data)
    return by_alpha


def cos_at_round(entries, round_num):
    for e in entries:
        if e["round"] == round_num:
            return e["cos"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--kl_rounds", type=int, default=100)
    ap.add_argument("--check_rounds", type=int, nargs="+",
                     default=[70, 80, 90, 95, 100, 105, 110, 120, 140, 160])
    args = ap.parse_args()

    by_alpha = load_by_alpha(Path(args.results_dir))
    if not by_alpha:
        raise SystemExit(f"No results.json files found under {args.results_dir}")

    print("cos(g_CE, g_KL) across the switch (round <= 100 is KL phase, > 100 is CE phase).\n"
          "Negative/near-zero values right at the switch would support the "
          "'opposite-direction step' hypothesis.\n")
    header = "alpha".rjust(6) + "".join(f"{'r'+str(r):>8}" for r in args.check_rounds)
    print(header)
    print("-" * len(header))
    for alpha in sorted(by_alpha):
        runs = by_alpha[alpha]
        row = f"{alpha:>6}"
        for r in args.check_rounds:
            vals = [cos_at_round(run["ce_kl_cosine"], r) for run in runs]
            vals = [v for v in vals if v is not None]
            row += f"{stats.mean(vals):>8.3f}" if vals else f"{'--':>8}"
        print(row)

    print(f"\n(mean cos across seeds; n per alpha: " +
          ", ".join(f"{a}={len(by_alpha[a])}" for a in sorted(by_alpha)) + ")")


if __name__ == "__main__":
    main()
