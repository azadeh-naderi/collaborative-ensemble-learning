#!/bin/bash
#SBATCH --job-name=peers-cel
#SBATCH --output=logs/peers_celnet_%A_%a.out
#SBATCH --error=logs/peers_celnet_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5    # 2 conditions x 3 seeds

# 9 ResNet-18 learners + oracle, CEL-Net MWM_AccDiff pairing, 220 rounds, constant LR 0.1, fixed data split.
#   celnet : KD from the better peer (0.9 KL + 0.1 CE on its predicted labels), 1 oracle CE update per round,
#            with a counterfactual KD epoch from the best peer before each CE update
#   all_ce : same pairing and schedule, every update is CE on the true labels (control)

set -euo pipefail

CONDITIONS=(celnet all_ce)
SEEDS=(0 1 2)

N_SEEDS=${#SEEDS[@]}
CONDITION=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
EXTRA_ARGS=()
if [ "$CONDITION" = "all_ce" ]; then
    EXTRA_ARGS+=(--all_ce)
fi
OUT_DIR="Geometry of Mixed-Loss/results/peers_celnet_pairing/${CONDITION}/seed${SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -c "import networkx" || { echo "ERROR: networkx is not installed in this environment (needed for MWM pairing)"; exit 1; }

echo "Peers with CEL-Net pairing: condition ${CONDITION}, seed ${SEED} -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/switch_peers_celnet_pairing.py" \
    --seed "$SEED" \
    --n_learners 9 \
    --rounds 220 \
    --pairing mwm_accdiff \
    --alpha 0.9 \
    --lr 0.1 \
    --out "$OUT_DIR" \
    --data_root ./data \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Done."
