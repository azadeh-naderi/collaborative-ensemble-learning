#!/bin/bash
#SBATCH --job-name=switch-teacher
#SBATCH --output=logs/switch_teacher_%A_%a.out
#SBATCH --error=logs/switch_teacher_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-2    # CE teachers: weak / medium / strong

set -euo pipefail

TEACHER_EPOCHS=(20 50 100)
E=${TEACHER_EPOCHS[$SLURM_ARRAY_TASK_ID]}
OUT="Geometry of Mixed-Loss/results/switch_exp1/teachers/teacher_ce_e${E}.pt"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Teacher: ResNet-50, CE, ${E} epochs -> ${OUT}"
python "Geometry of Mixed-Loss/experiments/switch_train_teacher.py" \
    --objective ce \
    --epochs "$E" \
    --out "$OUT" \
    --data_root ./data

echo "Done."
