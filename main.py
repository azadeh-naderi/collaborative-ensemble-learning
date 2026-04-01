import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from celnet.config import build_config_from_args
from celnet.experiment import run_experiment


def main() -> None:
    cfg = build_config_from_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
