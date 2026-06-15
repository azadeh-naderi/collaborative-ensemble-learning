from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class ExperimentConfig:
    run_id: int = 1
    run_seed: int = 42
    model: str = "resnet"
    dataset_name: str = "cifar_10"
    num_models: int = 10
    pairing_strategy: str = "mwm_acc"
    n_rounds: int = 100
    degree: int = 5
    friend_group_size: int = 5
    batch_size: int = 64
    input_channels: int = 3
    num_classes: int = 10
    pretrained: bool = False
    validation: bool = False
    optimizer_type: str = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    gamma: float = 0.1
    temperature: float = 4.0
    alpha: float = 0.9
    train_size_percent: float = 0.80
    val_size_percent: float = 0.10
    test_size_percent: float = 0.10
    num_workers: int = 4
    data_root: str = "./data"
    results_dir: str = "./results"
    model_seeds: List[int] = field(default_factory=lambda: [
        104729, 1299709, 15485863, 179424673, 2038074743,
        32452843, 49979687, 67867967, 86028121, 104395303,
        122949829, 141650939, 160481183, 179424691, 198491317,
        217645199, 236887691, 256203161, 275604541, 295075153,
    ])

    def to_dict(self):
        return asdict(self)


def load_config_file(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CELNet experiment runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--run_id", type=int, default=None)
    parser.add_argument("--run-seed", dest="run_seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, choices=["resnet", "lenet"])
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--num-models", dest="num_models", type=int, default=None)
    parser.add_argument("--pairing-strategy", dest="pairing_strategy", type=str, default=None)
    parser.add_argument("--n-rounds", dest="n_rounds", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--input-channels", dest="input_channels", type=int, default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--optimizer-type", dest="optimizer_type", type=str, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--results-dir", dest="results_dir", type=str, default=None)
    return parser


def build_config_from_args() -> ExperimentConfig:
    parser = _parser()
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.config:
        file_cfg = load_config_file(args.config)
        cfg = ExperimentConfig(**{**cfg.to_dict(), **file_cfg})

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            setattr(cfg, key, value)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    return cfg
