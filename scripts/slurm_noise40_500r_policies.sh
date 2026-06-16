#!/bin/bash
#SBATCH --job-name=celnet-n500-pol
#SBATCH --output=logs/noise40_500r_%A_%a.out
#SBATCH --error=logs/noise40_500r_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-4

set -euo pipefail

POLICIES=(MWM_ClassDist AccDiff ClassDist RRG_AccDiff RRg_Random)
NAMES=(MWM_ClassDist AccDiff ClassDist RRG_AccDiff RRg_Random)

POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
RESULTS_DIR="results/noise40_500r_${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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
    --config configs/noise40_500r.yaml \
    --pairing-strategy "$POLICY" \
    --results-dir "$RESULTS_DIR"

echo "Done. Results: $RESULTS_DIR"
