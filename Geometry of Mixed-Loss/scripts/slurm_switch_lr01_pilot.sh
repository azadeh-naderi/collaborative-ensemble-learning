#!/bin/bash
#SBATCH --job-name=switch-lr01
#SBATCH --output=logs/switch_lr01_%A_%a.out
#SBATCH --error=logs/switch_lr01_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --array=0-34    # 7 conditions x 5 seeds

# Pilot: everything at constant LR 0.1, 80 epochs, reusing the switch_exp1 teachers.
#   kd_then_ce : KD epochs 1-50, CE epochs 51-80
#   interleaved: KD, with every 5th epoch CE; before each CE epoch a KD epoch from the same state is also measured

set -euo pipefail

CONDITIONS=("ce_only:none" "kd_only:ce_e100" "kd_only:kd_g1" "kd_then_ce:ce_e100" "kd_then_ce:kd_g1"
            "interleaved:ce_e100" "interleaved:kd_g1")
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
COND=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
MODE=${COND%%:*}
TEACHER=${COND##*:}
OUT_DIR="Geometry of Mixed-Loss/results/switch_lr01_pilot/students/${MODE}/${TEACHER}/seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [ "$TEACHER" != "none" ]; then
    TEACHER_CKPT="Geometry of Mixed-Loss/results/switch_exp1/teachers/teacher_${TEACHER}.pt"
    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "ERROR: teacher checkpoint not found at $TEACHER_CKPT"
        exit 1
    fi
    EXTRA_ARGS+=(--teacher_ckpt "$TEACHER_CKPT")
fi
if [ "$MODE" = "interleaved" ]; then
    EXTRA_ARGS+=(--ce_every 5 --branch_at_ce)
fi

echo "Mode: ${MODE}  Teacher: ${TEACHER}  Seed: ${SEED}  ->  ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_run_student.py" \
    --mode "$MODE" \
    --seed "$SEED" \
    --phase1_epochs 50 \
    --phase2_epochs 30 \
    --phase1_lr_schedule constant \
    --ft_lr 0.1 \
    --out "$OUT_DIR" \
    --data_root ./data \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Done."
