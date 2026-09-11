#!/bin/bash
#SBATCH --job-name=switch-kd-teacher
#SBATCH --output=logs/switch_kd_teacher_%j.out
#SBATCH --error=logs/switch_kd_teacher_%j.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# Usage: sbatch slurm_switch_kd_teacher.sh GEN
#   GEN=1: pure-KL teacher distilled from the strong CE teacher (teacher_ce_e100)
#   GEN=2: pure-KL teacher distilled from the GEN=1 teacher

set -euo pipefail

GEN=${1:?usage: sbatch slurm_switch_kd_teacher.sh GEN   (1 or 2)}
TEACHER_DIR="Geometry of Mixed-Loss/results/switch_exp1/teachers"
if [ "$GEN" = "1" ]; then
    SOURCE="${TEACHER_DIR}/teacher_ce_e100.pt"
else
    SOURCE="${TEACHER_DIR}/teacher_kd_g$(( GEN - 1 )).pt"
fi
OUT="${TEACHER_DIR}/teacher_kd_g${GEN}.pt"
SEED=$(( 1234 + 1000 * GEN ))

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: source teacher not found at $SOURCE"
    exit 1
fi

echo "Teacher: ResNet-50, pure KL from ${SOURCE}, 100 epochs, seed ${SEED} -> ${OUT}"
python "Geometry of Mixed-Loss/experiments/switch_train_teacher.py" \
    --objective kd \
    --source_ckpt "$SOURCE" \
    --epochs 100 \
    --seed "$SEED" \
    --out "$OUT" \
    --data_root ./data

echo "Done."
