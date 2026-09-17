#!/bin/bash
#SBATCH --job-name=peers-mech
#SBATCH --output=logs/peers_mech_%A_%a.out
#SBATCH --error=logs/peers_mech_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=0-9    # 2 conditions x 5 seeds; task = 5 * condition + seed

# Mechanism runs: same setup as slurm_switch_peers_celnet_pairing.sh (9 learners + oracle, MWM_AccDiff, 220 rounds,
# constant LR 0.1, counterfactual KD epoch before each oracle CE update), plus --mechanism: at each oracle CE update,
# log which predictions the CE epoch and the KD counterfactual break or fix, where changed predictions go, how much the
# backbone and the final layer move, and a backbone/final-layer swap.
#   celnet    : models shaped by KD from better peers
#   all_ce_cf : same pairing and schedule, every update is CE (models shaped by CE only)

set -euo pipefail

CONDITIONS=(celnet all_ce_cf)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
CONDITION=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
EXTRA_ARGS=()
if [ "$CONDITION" = "all_ce_cf" ]; then
    EXTRA_ARGS+=(--all_ce)
fi
OUT_DIR="Geometry of Mixed-Loss/results/peers_mechanism/${CONDITION}/seed${SEED}"
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

echo "Mechanism run: condition ${CONDITION}, seed ${SEED} -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_peers_celnet_pairing.py" \
    --seed "$SEED" \
    --n_learners 9 \
    --rounds 220 \
    --pairing mwm_accdiff \
    --alpha 0.9 \
    --lr 0.1 \
    --mechanism \
    --out "$OUT_DIR" \
    --data_root ./data \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Done."
