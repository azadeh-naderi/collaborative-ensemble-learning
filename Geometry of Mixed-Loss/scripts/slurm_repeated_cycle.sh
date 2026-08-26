#!/bin/bash
#SBATCH --job-name=geo-repcycle
#SBATCH --output=logs/geometry_repcycle_%A_%a.out
#SBATCH --error=logs/geometry_repcycle_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-19    # 4 K values x 5 seeds = 20 jobs

set -euo pipefail

KS=(5 10 20 40)
SEEDS=(42 123 456 789 1337)
N_EXPOSURES=20

N_KS=${#KS[@]}
N_SEEDS=${#SEEDS[@]}

K_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

K=${KS[$K_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="Geometry of Mixed-Loss/results/repeated_cycle/k${K}_seed${SEED}"
TEACHER_CKPT="Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed${SEED}.pt"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

TOTAL_ROUNDS=$(( N_EXPOSURES * (K + 1) ))

echo "================================================"
echo " Array task   : ${SLURM_ARRAY_TASK_ID}"
echo " K            : ${K}"
echo " Seed         : ${SEED}"
echo " N exposures  : ${N_EXPOSURES}"
echo " Total rounds : ${TOTAL_ROUNDS}"
echo " Output       : ${OUT_DIR}"
echo " Teacher      : ${TEACHER_CKPT}"
echo "================================================"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
    echo "Exp 1 (slurm_alpha_sweep.sh) must have completed for seed $SEED first."
    exit 1
fi

python "Geometry of Mixed-Loss/experiments/run_repeated_cycle.py" \
    --k            "$K" \
    --n_exposures  "$N_EXPOSURES" \
    --alpha        0.9 \
    --temperature  4.0 \
    --seed         "$SEED" \
    --out          "$OUT_DIR" \
    --data_root    ./data \
    --teacher_ckpt "$TEACHER_CKPT"

echo "Done."
