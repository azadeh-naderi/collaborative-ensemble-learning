#!/bin/bash
#SBATCH --job-name=geo-alpha-klt
#SBATCH --output=logs/geometry_alpha_klt_%A_%a.out
#SBATCH --error=logs/geometry_alpha_klt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-29    # 6 alpha values x 5 seeds = 30 jobs

set -euo pipefail

ALPHAS=(0.1 0.3 0.5 0.7 0.9 1.0)
SEEDS=(42 123 456 789 1337)

N_ALPHAS=${#ALPHAS[@]}
N_SEEDS=${#SEEDS[@]}

ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

ALPHA=${ALPHAS[$ALPHA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

ALPHA_STR=$(echo "$ALPHA" | tr '.' '_')
OUT_DIR="Geometry of Mixed-Loss/results/alpha_sweep_kl_teacher/alpha${ALPHA_STR}_seed${SEED}"
GEN0_CKPT="Geometry of Mixed-Loss/results/teachers/teacher_resnet50_seed${SEED}.pt"
GEN1_CKPT="Geometry of Mixed-Loss/results/teachers_kl/teacher_resnet50_kl_seed${SEED}.pt"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p "Geometry of Mixed-Loss/results/teachers_kl"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " Array task : ${SLURM_ARRAY_TASK_ID}"
echo " Alpha      : ${ALPHA}"
echo " Seed       : ${SEED}"
echo " Output     : ${OUT_DIR}"
echo " Gen-0 (CE) : ${GEN0_CKPT}"
echo " Gen-1 (KL) : ${GEN1_CKPT}"
echo "================================================"

if [ ! -f "$GEN0_CKPT" ]; then
    echo "ERROR: Gen-0 checkpoint not found at $GEN0_CKPT"
    echo "Exp 1 (slurm_alpha_sweep.sh) must have completed for seed $SEED first."
    exit 1
fi

python "Geometry of Mixed-Loss/experiments/run_alpha_sweep_kl_teacher.py" \
    --alpha       "$ALPHA" \
    --kl_rounds   100 \
    --ce_rounds   60 \
    --temperature 4.0 \
    --seed        "$SEED" \
    --out         "$OUT_DIR" \
    --data_root   ./data \
    --gen0_ckpt   "$GEN0_CKPT" \
    --gen1_ckpt   "$GEN1_CKPT" \
    --gen1_epochs 100

echo "Done."
