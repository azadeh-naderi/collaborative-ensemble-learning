#!/bin/bash
# =============================================================================
# SLURM script: run the CelNet / CE / KD flatness (sharpness) comparison
#
# Submit with:
#   sbatch scripts/slurm_flatness.sh
#
# Override checkpoint paths at submission time, e.g.:
#   sbatch --export=CELNET_PATH=/path/to/celnet.pt,CE_PATH=/path/to/ce.pt,KD_PATH=/path/to/kd.pt \
#       scripts/slurm_flatness.sh
# =============================================================================

#SBATCH --job-name=celnet-flatness
#SBATCH --output=logs/flatness_%j.out
#SBATCH --error=logs/flatness_%j.err
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# ── user-configurable defaults (override via --export) ──────────────────────
JOBS_DIR="/home/an57/ondemand/data/sys/myjobs/projects/default/83"
CELNET_PATH="${CELNET_PATH:-${JOBS_DIR}/trained_models_10_resnet_MWM_AccDiff_New}"
CE_PATH="${CE_PATH:-${JOBS_DIR}/trained_CE_models_10_resnet_122epochs}"
KD_PATH="${KD_PATH:-${JOBS_DIR}/trained_KD_models_10_resnet_122epochs}"
SIGMAS="${SIGMAS:-0.0 0.01 0.02 0.05 0.1 0.2}"
REPEATS="${REPEATS:-5}"
RESULTS_DIR="${RESULTS_DIR:-results/flatness_${SLURM_JOB_ID}}"
# ─────────────────────────────────────────────────────────────────────────────

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
pip install --user -q pandas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
mkdir -p logs

echo "======================================================"
echo " Job ID      : ${SLURM_JOB_ID}"
echo " Node        : ${SLURMD_NODENAME:-local}"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " CelNet path : ${CELNET_PATH}"
echo " CE path     : ${CE_PATH}"
echo " KD path     : ${KD_PATH}"
echo " Sigmas      : ${SIGMAS}"
echo " Repeats     : ${REPEATS}"
echo " Results dir : ${RESULTS_DIR}"
echo "======================================================"

python scripts/flatness_experiment.py \
    --celnet-path "$CELNET_PATH" \
    --ce-path "$CE_PATH" \
    --kd-path "$KD_PATH" \
    --sigmas $SIGMAS \
    --repeats "$REPEATS" \
    --output "$RESULTS_DIR"

echo "Done. Results: $RESULTS_DIR"
