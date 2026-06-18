#!/bin/bash
#SBATCH --job-name=celnet-div
#SBATCH --output=logs/diversity_%j.out
#SBATCH --error=logs/diversity_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

OUT_DIR="${OUT_DIR:-results/diversity_mwm_acc_${SLURM_JOB_ID}}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "======================================================"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Node        : ${SLURMD_NODENAME:-local}"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Output dir  : ${OUT_DIR}"
echo "======================================================"

python experiments/diversity/run_diversity.py \
    --config configs/diversity_mwm_acc_220r.yaml \
    --out "$OUT_DIR"

python experiments/diversity/plot_diversity.py \
    --data "$OUT_DIR" \
    --out  "$OUT_DIR"

echo "Done. Results: $OUT_DIR"
