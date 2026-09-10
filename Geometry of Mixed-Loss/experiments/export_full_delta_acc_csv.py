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
    cos_ce_kl    -- cos(g_CE, g_KL) at this round (gradient DIRECTION),
                    if logged that round -- blank otherwise. Only present
                    every round for runs made after GRAD_LOG_INTERVAL was
                    changed from 5 to 1; older results.json files only
                    have this at every 5th round, so most rows will be
                    blank for those.
    g_ce_norm, g_kl_norm -- gradient magnitudes (||g_CE||, ||g_KL||) at
                    the same rounds cos_ce_kl is present, for checking the
                    angle change isn't a magnitude artifact. Same
                    blank-row caveat as cos_ce_kl.

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
    n_cos_rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "seed", "round_index", "relative_round",
                          "phase", "val_acc", "delta_acc",
                          "cos_ce_kl", "g_ce_norm", "g_kl_norm"])
        for g in sorted(by_group):
            for data in by_group[g]:
                phase = data["phase"]
                switch_idx = phase.index("ce")
                val_acc = data.get("val_acc", [None] * len(phase))
                deltas = data["delta_acc"]
                seed = data.get("seed", "?")

                # build round -> (cos, g_ce_norm, g_kl_norm) lookup;
                # g_ce_norm/g_kl_norm are parallel-positioned to the cosine
                # log, not separately keyed by round. Two schema variants
                # exist across scripts: run_alpha_sweep.py/run_repeated_cycle.py
                # use "ce_kl_cosine" (key "round") + separate g_ce_norm/
                # g_kl_norm lists; run_kd_finetune.py uses "cosine" (key
                # "epoch") and never logs the norms at all.
                if "ce_kl_cosine" in data:
                    cosine_entries = data["ce_kl_cosine"]
                    round_key = "round"
                    g_ce_norms = data.get("g_ce_norm", [])
                    g_kl_norms = data.get("g_kl_norm", [])
                else:
                    cosine_entries = data.get("cosine", [])
                    round_key = "epoch"
                    g_ce_norms = []
                    g_kl_norms = []
                cos_by_round: dict[int, tuple] = {}
                for pos, entry in enumerate(cosine_entries):
                    gce = g_ce_norms[pos] if pos < len(g_ce_norms) else None
                    gkl = g_kl_norms[pos] if pos < len(g_kl_norms) else None
                    cos_by_round[entry[round_key]] = (entry["cos"], gce, gkl)

                for i, (p, d) in enumerate(zip(phase, deltas)):
                    round_index = i + 1
                    va = val_acc[i] if i < len(val_acc) else None
                    cos, gce, gkl = cos_by_round.get(round_index, (None, None, None))
                    if cos is not None:
                        n_cos_rows += 1
                    # relative_round: 0 = last KL round (i == switch_idx-1),
                    # +1 = first CE round (i == switch_idx)
                    relative_round = i - switch_idx + 1
                    writer.writerow([g, seed, round_index, relative_round, p, va, d,
                                      cos, gce, gkl])
                    n_rows += 1

    print(f"Wrote {n_rows} rows ({len(by_group)} groups) to {out_path}")
    print(f"  {n_cos_rows}/{n_rows} rows have cos_ce_kl populated "
          f"({'per-round logging' if n_cos_rows > n_rows * 0.5 else 'sparse -- likely pre-GRAD_LOG_INTERVAL=1 data'})")


if __name__ == "__main__":
    main()
