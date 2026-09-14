#!/bin/bash
#SBATCH --job-name=switch-peers
#SBATCH --output=logs/switch_peers_%A_%a.out
#SBATCH --error=logs/switch_peers_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4    # one group of peers per seed

# 4 ResNet-18 peers from scratch, constant LR 0.1, 80 rounds (one epoch per peer per round).
# KD updates: 0.9 * T^2 * KL + 0.1 * CE on the teacher peer's predicted labels (T = 4).
# Every 5th update of each peer is CE on the true labels; before it, a KD epoch from the same state is measured.

set -euo pipefail

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="Geometry of Mixed-Loss/results/switch_peers/seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Peers: seed ${SEED} -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_peers.py" \
    --seed "$SEED" \
    --n_peers 4 \
    --rounds 80 \
    --ce_every 5 \
    --alpha 0.9 \
    --lr 0.1 \
    --branch_at_ce \
    --out "$OUT_DIR" \
    --data_root ./data

echo "Done."
