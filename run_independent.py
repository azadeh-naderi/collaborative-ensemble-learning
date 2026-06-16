"""Entry point for the independent baseline experiment."""
import argparse
from pathlib import Path

import yaml

from src.celnet.config import ExperimentConfig, load_config_file
from src.celnet.independent_baseline import run_independent_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", dest="results_dir", default=None)
    args = parser.parse_args()

    defaults = ExperimentConfig()
    file_cfg = load_config_file(args.config)
    cfg = ExperimentConfig(**{**defaults.to_dict(), **file_cfg})

    if args.results_dir:
        cfg.results_dir = args.results_dir

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    run_independent_baseline(cfg)
