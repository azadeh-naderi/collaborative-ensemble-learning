#!/bin/bash
#SBATCH --job-name=geo-kd-ft
#SBATCH --output=logs/geometry_kd_%A_%a.out
#SBATCH --error=logs/geometry_kd_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-24    # 5 K values × 5 seeds = 25 jobs

set -euo pipefail

KD_EPOCHS_LIST=(5 20 50 100 200)
SEEDS=(42 123 456 789 1337)

K_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

KD_EP=${KD_EPOCHS_LIST[$K_IDX]}
SEED=${SEEDS[$S_IDX]}
OUT_DIR="Geometry of Mixed-Loss/results/kd_finetune/k${KD_EP}_seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " KD epochs  : ${KD_EP}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_kd_finetune.py" \
    --kd_epochs   "$KD_EP" \
    --ce_epochs   60 \
    --alpha       0.9 \
    --temperature 4.0 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data

echo "Done."
