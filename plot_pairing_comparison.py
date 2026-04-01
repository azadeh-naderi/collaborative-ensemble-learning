#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celnet.plotting import load_tensor, save_policy_comparison_plot


def parse_args():
    p = argparse.ArgumentParser(description="Plot cumulative oracle vs non-oracle gain across pairing policies.")
    p.add_argument("--output", default="results/Cumulative_Accuracy_Gain_Pairing.png", help="Output PNG path")
    p.add_argument("--euclidean-oracle")
    p.add_argument("--euclidean-non-oracle")
    p.add_argument("--max-oracle")
    p.add_argument("--max-non-oracle")
    p.add_argument("--mwm-acc-oracle")
    p.add_argument("--mwm-acc-non-oracle")
    p.add_argument("--mwm-classacc-oracle")
    p.add_argument("--mwm-classacc-non-oracle")
    return p.parse_args()


def main():
    args = parse_args()
    policy_tensors = {}

    if args.euclidean_oracle and args.euclidean_non_oracle:
        policy_tensors["greedy-euclidean"] = (
            load_tensor(args.euclidean_oracle),
            load_tensor(args.euclidean_non_oracle),
        )
    if args.max_oracle and args.max_non_oracle:
        policy_tensors["max"] = (
            load_tensor(args.max_oracle),
            load_tensor(args.max_non_oracle),
        )
    if args.mwm_acc_oracle and args.mwm_acc_non_oracle:
        policy_tensors["mwm_acc"] = (
            load_tensor(args.mwm_acc_oracle),
            load_tensor(args.mwm_acc_non_oracle),
        )
    if args.mwm_classacc_oracle and args.mwm_classacc_non_oracle:
        policy_tensors["mwm_classAcc"] = (
            load_tensor(args.mwm_classacc_oracle),
            load_tensor(args.mwm_classacc_non_oracle),
        )

    if not policy_tensors:
        raise ValueError("No valid policy tensor pairs were provided.")

    out = save_policy_comparison_plot(policy_tensors, args.output)
    print(f"Saved comparison plot to: {out}")


if __name__ == "__main__":
    main()
