#!/bin/bash
#SBATCH --job-name=switch-student
#SBATCH --output=logs/switch_student_%A_%a.out
#SBATCH --error=logs/switch_student_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-109    # 11 conditions x 10 seeds

set -euo pipefail

TEACHERS=(ce_e20 ce_e50 ce_e100 kd_g1 kd_g2)
CONDITIONS=("ce_only:none")
for T in "${TEACHERS[@]}"; do CONDITIONS+=("kd_then_ce:${T}"); done
for T in "${TEACHERS[@]}"; do CONDITIONS+=("kd_only:${T}"); done
SEEDS=(0 1 2 3 4 5 6 7 8 9)

N_SEEDS=${#SEEDS[@]}
COND=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
MODE=${COND%%:*}
TEACHER=${COND##*:}
OUT_DIR="Geometry of Mixed-Loss/results/switch_exp1/students/${MODE}/${TEACHER}/seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

TEACHER_ARGS=()
if [ "$TEACHER" != "none" ]; then
    TEACHER_CKPT="Geometry of Mixed-Loss/results/switch_exp1/teachers/teacher_${TEACHER}.pt"
    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "ERROR: teacher checkpoint not found at $TEACHER_CKPT -- train the teachers first"
        exit 1
    fi
    TEACHER_ARGS=(--teacher_ckpt "$TEACHER_CKPT")
fi

echo "Mode: ${MODE}  Teacher: ${TEACHER}  Seed: ${SEED}  ->  ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_run_student.py" \
    --mode "$MODE" \
    --seed "$SEED" \
    --out "$OUT_DIR" \
    --data_root ./data \
    ${TEACHER_ARGS[@]+"${TEACHER_ARGS[@]}"}

echo "Done."
