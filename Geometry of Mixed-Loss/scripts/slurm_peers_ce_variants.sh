#!/bin/bash
#SBATCH --job-name=peers-cev
#SBATCH --output=logs/peers_cev_%A_%a.out
#SBATCH --error=logs/peers_cev_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --array=0-14   # 3 oracle updates x 5 seeds; task = 5 * condition + seed

# KD-shaped peers (9 learners + oracle, MWM_AccDiff, 220 rounds, constant LR 0.1). At every oracle update, from the
# same weights and optimizer state, measure the KD counterfactual and all CE variants:
#   ce       : normal CE epoch
#   ce_head  : CE on the final layer only (backbone and BatchNorm statistics frozen)
#   ce_lowlr : normal CE epoch at LR 0.01
# The condition decides which of the three is actually applied at oracle rounds (the others are measured only).

set -euo pipefail

CONDITIONS=(ce ce_head ce_lowlr)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
ORACLE_UPDATE=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
OUT_DIR="Geometry of Mixed-Loss/results/peers_ce_variants/${ORACLE_UPDATE}/seed${SEED}"
if [ -f "$OUT_DIR/results.json" ]; then
    echo "Skipping ${ORACLE_UPDATE} seed ${SEED}: $OUT_DIR/results.json already exists"
    exit 0
fi

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -c "import networkx" || { echo "ERROR: networkx is not installed in this environment (needed for MWM pairing)"; exit 1; }

echo "CE variants: oracle update ${ORACLE_UPDATE}, seed ${SEED} -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_peers_celnet_pairing.py" \
    --seed "$SEED" \
    --n_learners 9 \
    --rounds 220 \
    --pairing mwm_accdiff \
    --alpha 0.9 \
    --lr 0.1 \
    --oracle_update "$ORACLE_UPDATE" \
    --ce_variants \
    --low_lr 0.01 \
    --out "$OUT_DIR" \
    --data_root ./data

echo "Done."
