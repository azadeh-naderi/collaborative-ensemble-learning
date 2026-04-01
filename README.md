# Collaborative Ensemble Learning

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
research-repo/
├── configs/
│   └── default.yaml
├── results/
├── scripts/
│   ├── plot_pairing_comparison.py
│   └── run_default.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── models.py
│   ├── pairing.py
│   ├── plotting.py
│   ├── training.py
│   └── utils.py
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


## Plot cumulative policy comparisons

A helper script is included for the cumulative comparison plot from your plotting file. It is based on the tensor-loading plot script you shared. fileciteturn1file0

Example:

```bash
python scripts/plot_pairing_comparison.py   --euclidean-oracle /path/to/10_oracle_cum_tensor_resnet_euclidean_r1   --euclidean-non-oracle /path/to/10_non_oracle_cum_tensor_resnet_euclidean_r1   --max-oracle /path/to/10_oracle_cum_tensor_resnet_max_r1   --max-non-oracle /path/to/10_non_oracle_cum_tensor_resnet_max_r1   --mwm-acc-oracle /path/to/10_oracle_cum_tensor_resnet_mwm_acc_r1   --mwm-acc-non-oracle /path/to/10_non_oracle_cum_tensor_resnet_mwm_acc_r1   --mwm-classacc-oracle /path/to/10_oracle_cum_tensor_resnet_mwm_classAcc_r1   --mwm-classacc-non-oracle /path/to/10_non_oracle_cum_tensor_resnet_mwm_classAcc_r1
```

This saves:

```text
results/Cumulative_Accuracy_Gain_Pairing.png
```

## Upload everything to GitHub in one step

The easiest way is not the normal GitHub upload page. Instead:

1. Create an empty repository on GitHub.
2. On the repo main page, press the `.` key or open the web editor.
3. In your file explorer, open the local repo folder.
4. Drag the whole set of files and folders from inside the repo into the browser editor window.
5. Commit the changes from the Source Control panel.

That method preserves nested folders like `src/`, `configs/`, and `scripts/` much better than trying to create folders one by one in the upload UI.

Another simple option is GitHub Desktop: create the repo there, drag the project folder into it once, then publish to GitHub. For large research repos, that is usually much easier than the browser uploader.

One thing to be aware of: GitHub's web interface does **not** support uploading a `.zip` and automatically extracting it into your repository. So there is no true single-click zip upload on the GitHub website itself.
