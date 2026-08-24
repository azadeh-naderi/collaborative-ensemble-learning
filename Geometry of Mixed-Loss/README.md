# Geometry of Mixed-Loss Training

**Targeting ICLR 2026**

This folder contains everything for the standalone paper on loss basin incompatibility in sequential CE–KL training. The phenomenon is discovered in CEL-Net but applies broadly to KD fine-tuning, DML, and self-distillation pipelines.

## Core claim

After sustained KL-dominated training (high α), switching to CE loss causes validation accuracy to drop — not because of label noise or teacher quality, but because CE and KL gradient directions have diverged to near-orthogonality. The effect is controlled by:
- **α** (KL weight): higher α → earlier onset
- **Training depth** (rounds/epochs of KL exposure): deeper → stronger incompatibility
- **Oracle exposure frequency** (in CEL-Net): 1D greedy policies maintain alignment; MWM/graph-constrained policies do not

## Folder structure

```
Geometry of Mixed-Loss/
├── paper/
│   └── geometry_paper.html       ← draft paper (open in browser)
├── experiments/
│   ├── run_alpha_sweep.py        ← Exp 1: α ∈ {0.1,0.3,0.5,0.7,0.9,1.0}
│   ├── run_kd_finetune.py        ← Exp 2: KD depth K ∈ {5,20,50,100,200}
│   ├── run_dml_switch.py         ← Exp 3: DML regime switch at epoch 150
│   ├── run_self_distill.py       ← Exp 4: self-distillation cascade Gen 0→3
│   ├── log_gradient_angle.py     ← Exp 5: post-hoc cos(g_CE, g_KL) logging
│   ├── run_arch_sweep.py         ← Exp 6: architecture (RN18/RN50/MNv3)
│   ├── run_dataset_sweep.py      ← Exp 7: dataset (C10/C100/TinyIN-10)
│   └── run_temperature_sweep.py  ← Exp 8: temperature T ∈ {1,2,4,8}
├── scripts/
│   ├── slurm_alpha_sweep.sh      ← 30-task array (6α × 5 seeds)
│   ├── slurm_kd_finetune.sh      ← 25-task array (5K × 5 seeds)
│   ├── slurm_dml_switch.sh       ← 10-task array (2α × 5 seeds)
│   ├── slurm_self_distill.sh     ←  3-task array (3 seeds)
│   ├── slurm_arch_sweep.sh       ←  9-task array (3 archs × 3 seeds)
│   ├── slurm_dataset_sweep.sh    ←  9-task array (3 datasets × 3 seeds)
│   └── slurm_temperature_sweep.sh← 20-task array (4T × 5 seeds)
└── results/                      ← output written here by all scripts
```

## Running experiments (Wulver / Slurm)

All commands run from the **repo root** (`~/collaborative-ensemble-learning`):

```bash
git pull

# Exp 1 — α-sweep (primary, run first)
sbatch "Geometry of Mixed-Loss/scripts/slurm_alpha_sweep.sh"

# Exp 2 — KD fine-tuning depth
sbatch "Geometry of Mixed-Loss/scripts/slurm_kd_finetune.sh"

# Exp 3 — DML regime switch
sbatch "Geometry of Mixed-Loss/scripts/slurm_dml_switch.sh"

# Exp 4 — self-distillation cascade
sbatch "Geometry of Mixed-Loss/scripts/slurm_self_distill.sh"

# Exp 6 — architecture sweep (resnet18 / resnet50 / mobilenet_v3_small)
sbatch "Geometry of Mixed-Loss/scripts/slurm_arch_sweep.sh"

# Exp 7 — dataset sweep (cifar10 / cifar100 / tinyimagenet10)
# Note: TinyImageNet must be pre-downloaded to ./data/tiny-imagenet-200/
sbatch "Geometry of Mixed-Loss/scripts/slurm_dataset_sweep.sh"

# Exp 8 — temperature sweep (T ∈ {1,2,4,8}, α=0.9 fixed)
sbatch "Geometry of Mixed-Loss/scripts/slurm_temperature_sweep.sh"
```

After jobs finish, run the gradient angle logger on each result directory:

```bash
python "Geometry of Mixed-Loss/experiments/log_gradient_angle.py" \
    --ckpt_dir "Geometry of Mixed-Loss/results/alpha_sweep/alpha0_9_seed42/" \
    --alpha 0.9 --seed 42
```

## GPU budget estimate

| Experiment        | Tasks | Hours/task | Total      |
|-------------------|-------|-----------|------------|
| α-sweep           | 30    | 2h        | 60 A100-h  |
| KD depth          | 25    | 3h        | 75 A100-h  |
| DML switch        | 10    | 3h        | 30 A100-h  |
| Self-distil       | 3     | 5h        | 15 A100-h  |
| Arch sweep        | 9     | 3h        | 27 A100-h  |
| Dataset sweep     | 9     | 4h        | 36 A100-h  |
| Temperature sweep | 20    | 2h        | 40 A100-h  |
| **Total**         |       |           | **~283 A100-h** |

### Note on TinyImageNet (Exp 7)
Download and extract before submitting:
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./data/
unzip ./data/tiny-imagenet-200.zip -d ./data/
```
The script uses the first 10 wnids sorted alphabetically as the 10-class subset.

## Key predictions

| Condition | Expected T* | Initial Δ_CE |
|-----------|------------|--------------|
| α=0.1     | none       | positive     |
| α=0.5     | ~80 rounds | ≈ 0          |
| α=0.9     | ~25 rounds | −0.8%        |
| α=1.0     | immediate  | −1.5%        |
| KD K=5ep  | none       | positive     |
| KD K=200ep| ~ep 1–5   | −2%          |
| DML α=0.5 | none       | no effect    |
| DML α=0.9 | ep 1–5 post-switch | −0.5% |
| Self-distil Gen-1 | ep 1–3 | −0.2% |
| Self-distil Gen-3 | ep 1–3 | −1.8% |
