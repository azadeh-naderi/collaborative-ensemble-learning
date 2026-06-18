#!/bin/bash
#SBATCH --job-name=celnet-div-pol
#SBATCH --output=logs/diversity_%A_%a.out
#SBATCH --error=logs/diversity_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

set -euo pipefail

POLICIES=(MWM_ClassDist AccDiff ClassDist RRG_AccDiff RRg_Random)
POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="results/diversity_${POLICY}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /home/an57/ondemand/data/sys/myjobs/projects/default/15/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "======================================================"
echo " Job ID      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo " Policy      : ${POLICY}"
echo " Output dir  : ${OUT_DIR}"
echo "======================================================"

python experiments/diversity/run_diversity.py \
    --config configs/diversity_mwm_acc_220r.yaml \
    --pairing-strategy "$POLICY" \
    --out "$OUT_DIR"

python experiments/diversity/plot_diversity.py \
    --data "$OUT_DIR" \
    --out  "$OUT_DIR"

echo "Done."
