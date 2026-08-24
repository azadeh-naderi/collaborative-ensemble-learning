#!/bin/bash
#SBATCH --job-name=geo-temp
#SBATCH --output=logs/geometry_temp_%A_%a.out
#SBATCH --error=logs/geometry_temp_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=dept_dms
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-19    # 4 temperatures × 5 seeds = 20 jobs

set -euo pipefail

TEMPS=(1.0 2.0 4.0 8.0)
SEEDS=(42 123 456 789 1337)

T_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

TEMP=${TEMPS[$T_IDX]}
SEED=${SEEDS[$S_IDX]}
TEMP_STR=$(echo "$TEMP" | tr '.' '_')
OUT_DIR="Geometry of Mixed-Loss/results/temperature_sweep/T${TEMP_STR}_seed${SEED}"
TEACHER_CKPT="Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed${SEED}.pt"

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
echo " Temperature: ${TEMP}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_temperature_sweep.py" \
    --temperature "$TEMP" \
    --alpha       0.9 \
    --kl_rounds   100 \
    --ce_rounds   60 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data \
    --teacher_ckpt "$TEACHER_CKPT"

echo "Done."
