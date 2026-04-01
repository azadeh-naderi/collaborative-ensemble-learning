from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeLogger:
    def __init__(self, timezone: str = "US/Eastern"):
        self.timezone = pytz.timezone(timezone)

    def __enter__(self):
        self.start_time = datetime.now(pytz.utc).astimezone(self.timezone)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = datetime.now(pytz.utc).astimezone(self.timezone)
        print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.end_time - self.start_time}")


def calculate_std_dev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
