# CELNet: Collaborative Ensemble Learning

Research-style PyTorch repository for collaborative ensemble learning with:

- oracle supervision
- peer-to-peer knowledge distillation
- multiple pairing strategies
- random-graph constrained matching
- ensemble evaluation and disagreement analysis
- ensemble-to-student distillation

This repository is a modularized version of the original experiment script you shared. It is organized so you can more easily run experiments, extend pairing policies, and reuse components across projects.

## Repository structure

```text
celnet-research-repo/
├── configs/
│   └── default.yaml
├── results/
├── scripts/
│   └── run_default.sh
├── src/
│   └── celnet/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── experiment.py
│       ├── metrics.py
│       ├── models.py
│       ├── pairing.py
│       ├── plotting.py
│       ├── training.py
│       └── utils.py
├── main.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default run:

```bash
python main.py --run_id 1
```

Override some parameters from the command line:

```bash
python main.py --run_id 1 --num-models 10 --pairing-strategy random_graph_mwm_acc --n-rounds 100 --model resnet
```

Use a YAML config file:

```bash
python main.py --config configs/default.yaml
```

## Notes

- The current defaults are aligned with your uploaded CIFAR-10 collaborative KD experiment.
- The repository keeps the main experimental logic close to your original script, but separates concerns into reusable modules.
- The `results/` directory is kept for generated plots and tensors.

## Main modules

- `models.py`: model definitions and initialization
- `data.py`: dataset loading and deterministic data splitting
- `pairing.py`: pairing strategies including MWM and random-graph pairing
- `training.py`: oracle training, KD training, optimizer/scheduler management, ensemble distillation
- `metrics.py`: accuracy, confidence, ensemble metrics, disagreement analysis
- `experiment.py`: end-to-end experiment loop
- `plotting.py`: gain-curve visualization and saving

## Suggested next cleanup steps

- move hyperparameters into multiple config files for ablation studies
- save all run metadata as JSON alongside outputs
- add checkpointing and resume support
- add unit tests for pairing strategies and metric functions
- optionally convert `main.py` into an entry point under `src/`
