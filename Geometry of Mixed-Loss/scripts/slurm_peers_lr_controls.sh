#!/bin/bash
#SBATCH --job-name=peers-lrc
#SBATCH --output=logs/peers_lrc_%A_%a.out
#SBATCH --error=logs/peers_lrc_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --array=0-19   # 4 conditions x 5 seeds; task = 5 * condition + seed

# Learning-rate controls for the low-LR label-update result (9 learners + oracle, MWM_AccDiff, 220 rounds, LR 0.1):
#   kd_branches      : KD-shaped, normal CE at oracle rounds; per update also measure CE on the final layer only, CE at
#                      LR 0.01, KD on the final layer only and KD at LR 0.01 (all from the same state)
#   kd_lowlr_budget  : KD-shaped, normal CE at oracle rounds, plus one randomly chosen KD update per round at LR 0.01
#                      (the same number of LR-0.01 epochs as low-LR label updates, spent on KD instead)
#   all_ce_lowlr     : all-CE control with LR-0.01 oracle updates; per update also measure the other variants
#   scheduled_head   : KD-shaped, normal CE below 70% validation accuracy, CE on the final layer only from 70% on
#                      (the mechanism's fix, with no learning-rate change)

set -euo pipefail

CONDITIONS=(kd_branches kd_lowlr_budget all_ce_lowlr scheduled_head)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
CONDITION=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
case "$CONDITION" in
    kd_branches)     EXTRA_ARGS=(--ce_variants --kd_variants) ;;
    kd_lowlr_budget) EXTRA_ARGS=(--kd_lowlr_per_round) ;;
    all_ce_lowlr)    EXTRA_ARGS=(--all_ce --oracle_update ce_lowlr --ce_variants --kd_variants) ;;
    scheduled_head)  EXTRA_ARGS=(--gentle_after 70 --gentle_update ce_head) ;;
esac
OUT_DIR="Geometry of Mixed-Loss/results/peers_lr_controls/${CONDITION}/seed${SEED}"
if [ -f "$OUT_DIR/results.json" ]; then
    echo "Skipping ${CONDITION} seed ${SEED}: $OUT_DIR/results.json already exists"
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

echo "LR controls: ${CONDITION}, seed ${SEED} -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_peers_celnet_pairing.py" \
    --seed "$SEED" \
    --n_learners 9 \
    --rounds 220 \
    --pairing mwm_accdiff \
    --alpha 0.9 \
    --lr 0.1 \
    --low_lr 0.01 \
    --out "$OUT_DIR" \
    --data_root ./data \
    "${EXTRA_ARGS[@]}"

echo "Done."
