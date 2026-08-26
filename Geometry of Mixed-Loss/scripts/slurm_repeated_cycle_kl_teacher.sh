#!/bin/bash
#SBATCH --job-name=geo-repcycle-klt
#SBATCH --output=logs/geometry_repcycle_klt_%A_%a.out
#SBATCH --error=logs/geometry_repcycle_klt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-11    # 4 K values x 3 seeds = 12 jobs (only seeds with a cached Gen-1 teacher)

set -euo pipefail

KS=(5 10 20 40)
SEEDS=(42 123 456)   # only the 3 seeds with a cached KL-self-distilled Gen-1 teacher
N_EXPOSURES=20

N_KS=${#KS[@]}
N_SEEDS=${#SEEDS[@]}

K_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

K=${KS[$K_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="Geometry of Mixed-Loss/results/repeated_cycle_kl_teacher/k${K}_seed${SEED}"
TEACHER_CKPT="Geometry of Mixed-Loss/results/teachers_kl/teacher_resnet50_kl_seed${SEED}.pt"

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
echo " Teacher (KL) : ${TEACHER_CKPT}"
echo "================================================"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: KL-self-distilled Gen-1 teacher checkpoint not found at $TEACHER_CKPT"
    echo "This script only runs seeds with an already-cached Gen-1 teacher"
    echo "(from run_alpha_sweep_kl_teacher.py's earlier pilot). Refusing to"
    echo "silently fall back to a CE-pretrained teacher for a missing seed."
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
