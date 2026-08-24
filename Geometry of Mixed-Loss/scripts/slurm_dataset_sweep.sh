#!/bin/bash
#SBATCH --job-name=geo-data
#SBATCH --output=logs/geometry_dataset_%A_%a.out
#SBATCH --error=logs/geometry_dataset_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-8     # 3 datasets × 3 seeds = 9 jobs

set -euo pipefail

DATASETS=(cifar10 cifar100 tinyimagenet10)
SEEDS=(42 123 456)

D_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

DATASET=${DATASETS[$D_IDX]}
SEED=${SEEDS[$S_IDX]}
OUT_DIR="Geometry of Mixed-Loss/results/dataset_sweep/${DATASET}_seed${SEED}"
TEACHER_CKPT="Geometry of Mixed-Loss/results/teachers/teacher_resnet50_${DATASET}_seed${SEED}.pt"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p "Geometry of Mixed-Loss/results/teachers"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " Dataset    : ${DATASET}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_dataset_sweep.py" \
    --dataset     "$DATASET" \
    --alpha       0.9 \
    --kl_rounds   100 \
    --ce_rounds   60 \
    --temperature 4.0 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data \
    --teacher_ckpt "$TEACHER_CKPT"

echo "Done."
