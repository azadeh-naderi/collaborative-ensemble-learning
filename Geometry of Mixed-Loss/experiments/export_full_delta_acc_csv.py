"""
Exports the FULL per-round data (every round, both KL and CE phases, every
seed, raw -- not aggregated/plotted) to a single CSV file, for later reuse
in any tool (spreadsheet, pandas, etc.).

One row per (group, seed, round). Columns:
    group        -- the dominance-variable value (alpha, kd_epochs, or k)
    seed
    round_index  -- 1-based absolute round/epoch number within this run
    relative_round -- round position relative to the KL->CE switch
                      (0 = last KL round, +1 = first CE round, -1 = second-
                      to-last KL round, etc.), found programmatically from
                      the phase array (first index where phase == "ce")
    phase        -- "kl" or "ce" (or "kd" for kd_finetune)
    val_acc      -- validation accuracy logged that round
    delta_acc    -- val_acc[this round] - val_acc[previous round]

Works generically across:
    alpha-sweep:    --glob "alpha*_seed*" --group_field alpha
    repeated-cycle: --glob "k*_seed*"     --group_field k
    kd_finetune:    --glob "k*_seed*"     --group_field kd_epochs

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/export_full_delta_acc_csv.py" \
        --results_dir "Geometry of Mixed-Loss/results/alpha_sweep" \
        --glob "alpha*_seed*" --group_field alpha \
        --out "Geometry of Mixed-Loss/results/alpha_sweep/analysis/full_delta_acc.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", required=True, dest="glob_pattern")
    ap.add_argument("--group_field", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_group = load_by_group(Path(args.results_dir), args.glob_pattern, args.group_field)
    if not by_group:
        raise SystemExit(f"No results.json files found under {args.results_dir} "
                          f"matching {args.glob_pattern!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "seed", "round_index", "relative_round",
                          "phase", "val_acc", "delta_acc"])
        for g in sorted(by_group):
            for data in by_group[g]:
                phase = data["phase"]
                switch_idx = phase.index("ce")
                val_acc = data.get("val_acc", [None] * len(phase))
                deltas = data["delta_acc"]
                seed = data.get("seed", "?")
                for i, (p, d) in enumerate(zip(phase, deltas)):
                    va = val_acc[i] if i < len(val_acc) else None
                    writer.writerow([g, seed, i + 1, i - switch_idx, p, va, d])
                    n_rows += 1

    print(f"Wrote {n_rows} rows ({len(by_group)} groups) to {out_path}")


if __name__ == "__main__":
    main()
