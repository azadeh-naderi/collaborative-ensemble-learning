"""Aggregate summary_r*.json files produced by parallel multi-seed runs.

Scans every seed_*/summary_r*.json under a results directory, collects key
metrics, and prints a table of mean ± std across seeds.  Optionally saves the
aggregated data to a JSON file.

Usage
-----
python parallel_seeds/aggregate_results.py --results-dir results/parallel_seeds
python parallel_seeds/aggregate_results.py --results-dir results/parallel_seeds --save
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = [
    "avg_ensemble_test_accuracy",
    "avg_ensemble_test_confidence",
    "mean_pairwise_disagreement",
    "distilled_student_test_accuracy",
    "std_dev",
    "variance",
]


def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for json_path in sorted(results_dir.glob("seed_*/summary_r*.json")):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_source"] = str(json_path)
        summaries.append(data)
    return summaries


def aggregate(summaries: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for key in METRICS:
        values = []
        for s in summaries:
            if key in s:
                val = s[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
        if values:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            agg[key] = {"mean": mean, "std": std, "values": values, "n": len(values)}

    # Per-model accuracies (list field) — mean across seeds, then mean/std across models
    all_model_accs: list[list[float]] = []
    for s in summaries:
        if "individual_model_accuracies" in s:
            all_model_accs.append([float(v) for v in s["individual_model_accuracies"]])
    if all_model_accs:
        n_models = len(all_model_accs[0])
        per_model_means = []
        for mi in range(n_models):
            vals = [run[mi] for run in all_model_accs if mi < len(run)]
            per_model_means.append(statistics.mean(vals))
        agg["per_model_accuracy_mean_across_seeds"] = {
            "per_model": per_model_means,
            "overall_mean": statistics.mean(per_model_means),
            "overall_std": statistics.stdev(per_model_means) if len(per_model_means) > 1 else 0.0,
        }

    seeds = []
    for s in summaries:
        seed = s.get("config", {}).get("run_seed")
        if seed is not None:
            seeds.append(seed)
    agg["_meta"] = {"n_seeds": len(summaries), "seeds": seeds}

    return agg


def print_table(agg: dict) -> None:
    print(f"\n{'Metric':<45}  {'Mean':>10}  {'Std':>10}  {'N':>4}")
    print("-" * 75)
    for key in METRICS:
        if key not in agg:
            continue
        d = agg[key]
        print(f"{key:<45}  {d['mean']:>10.4f}  {d['std']:>10.4f}  {d['n']:>4}")

    if "per_model_accuracy_mean_across_seeds" in agg:
        d = agg["per_model_accuracy_mean_across_seeds"]
        print(
            f"\n{'per_model_accuracy (mean across seeds)':<45}"
            f"  {d['overall_mean']:>10.4f}  {d['overall_std']:>10.4f}"
        )

    meta = agg.get("_meta", {})
    print(f"\nAggregated over {meta.get('n_seeds', '?')} seed(s): {meta.get('seeds', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed CELNet results.")
    parser.add_argument(
        "--results-dir", dest="results_dir", type=str, default="results/parallel_seeds",
        help="Directory containing seed_*/ subdirectories.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save aggregated JSON to <results-dir>/aggregated.json.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        raise SystemExit(1)

    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"No summary_r*.json files found under {results_dir}/seed_*/")
        raise SystemExit(1)

    print(f"Found {len(summaries)} summary file(s) in {results_dir}")
    agg = aggregate(summaries)
    print_table(agg)

    if args.save:
        out_path = results_dir / "aggregated.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2)
        print(f"\nAggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
