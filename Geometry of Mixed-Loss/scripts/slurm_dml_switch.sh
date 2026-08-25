#!/bin/bash
#SBATCH --job-name=geo-dml-sw
#SBATCH --output=logs/geometry_dml_%A_%a.out
#SBATCH --error=logs/geometry_dml_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-9     # 2 alpha values × 5 seeds = 10 jobs

set -euo pipefail

ALPHAS=(0.5 0.9)
SEEDS=(42 123 456 789 1337)

A_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

ALPHA=${ALPHAS[$A_IDX]}
SEED=${SEEDS[$S_IDX]}
ALPHA_STR=$(echo "$ALPHA" | tr '.' '_')
OUT_DIR="Geometry of Mixed-Loss/results/dml_switch/alpha${ALPHA_STR}_seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " Alpha      : ${ALPHA}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_dml_switch.py" \
    --alpha         "$ALPHA" \
    --switch_epoch  150 \
    --total_epochs  210 \
    --temperature   4.0 \
    --seed          "$SEED" \
    --out           "$OUT_DIR" \
    --data_root     ./data

echo "Done."
