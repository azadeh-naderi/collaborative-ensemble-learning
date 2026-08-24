#!/bin/bash
#SBATCH --job-name=geo-alpha
#SBATCH --output=logs/geometry_alpha_%A_%a.out
#SBATCH --error=logs/geometry_alpha_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-29    # 6 alpha values × 5 seeds = 30 jobs

set -euo pipefail

ALPHAS=(0.1 0.3 0.5 0.7 0.9 1.0)
SEEDS=(42 123 456 789 1337)

N_ALPHAS=${#ALPHAS[@]}
N_SEEDS=${#SEEDS[@]}

ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

ALPHA=${ALPHAS[$ALPHA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

ALPHA_STR=$(echo "$ALPHA" | tr '.' '_')
OUT_DIR="Geometry of Mixed-Loss/results/alpha_sweep/alpha${ALPHA_STR}_seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

TEACHER_CKPT="Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed${SEED}.pt"
mkdir -p "Geometry of Mixed-Loss/results/teachers"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " Alpha      : ${ALPHA}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_alpha_sweep.py" \
    --alpha       "$ALPHA" \
    --kl_rounds   100 \
    --ce_rounds   60 \
    --temperature 4.0 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data \
    --teacher_ckpt "$TEACHER_CKPT"

echo "Done."
