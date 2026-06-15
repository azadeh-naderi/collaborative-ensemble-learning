#!/bin/bash
# =============================================================================
# SLURM array job: 40% Oracle label-noise experiment (E5-style) across
# multiple pairing policies on ResNet-18 / CIFAR-10, 220 rounds each.
#
# Array index -> pairing strategy:
#   0: mwm_classAcc   (MWM_ClassDist)
#   1: max            (AccDiff)
#   2: euclidean      (ClassDist)
#   3: random_3reg_max     (RRG_AccDiff)
#   4: random_3reg_uniform (RRG_Random)
#
# Submit with:
#   sbatch scripts/slurm_noise40_policies.sh
# =============================================================================

#SBATCH --job-name=celnet-noise40-pol
#SBATCH --output=logs/noise40_%A_%a.out
#SBATCH --error=logs/noise40_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --array=0-4

set -euo pipefail

POLICIES=(mwm_classAcc max euclidean random_3reg_max random_3reg_uniform)
NAMES=(MWM_ClassDist AccDiff ClassDist RRG_AccDiff RRG_Random)

POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
RESULTS_DIR="results/noise40_${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "======================================================"
echo " Job ID      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo " Node        : ${SLURMD_NODENAME:-local}"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Policy      : ${NAME} (${POLICY})"
echo " Results dir : ${RESULTS_DIR}"
echo "======================================================"

python main.py \
    --config configs/noise40_mwm_acc.yaml \
    --pairing-strategy "$POLICY" \
    --results-dir "$RESULTS_DIR"

echo "Done. Results: $RESULTS_DIR"
