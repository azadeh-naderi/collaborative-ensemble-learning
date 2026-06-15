#!/bin/bash
# =============================================================================
# SLURM script: CelNet MWM_AccDiff on ResNet-18 / CIFAR-10 with 40% symmetric
# label noise on the Oracle's training labels (nKDiff E5-style experiment).
#
# Submit with:
#   sbatch scripts/slurm_noise40_mwm_acc.sh
# =============================================================================

#SBATCH --job-name=celnet-noise40
#SBATCH --output=logs/noise40_%j.out
#SBATCH --error=logs/noise40_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/noise40_mwm_acc_${SLURM_JOB_ID}}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "======================================================"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Node        : ${SLURMD_NODENAME:-local}"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Results dir : ${RESULTS_DIR}"
echo "======================================================"

python main.py \
    --config configs/noise40_mwm_acc.yaml \
    --results-dir "$RESULTS_DIR"

echo "Done. Results: $RESULTS_DIR"
