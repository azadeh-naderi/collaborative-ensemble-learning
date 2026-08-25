#!/bin/bash
#SBATCH --job-name=geo-self-dist
#SBATCH --output=logs/geometry_selfdist_%A_%a.out
#SBATCH --error=logs/geometry_selfdist_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-2     # 3 seeds

set -euo pipefail

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="Geometry of Mixed-Loss/results/self_distill/seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo "================================================"

python "Geometry of Mixed-Loss/experiments/run_self_distill.py" \
    --n_gens      3 \
    --kd_epochs   150 \
    --ce_epochs   60 \
    --alpha       0.9 \
    --temperature 4.0 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data

echo "Done."
