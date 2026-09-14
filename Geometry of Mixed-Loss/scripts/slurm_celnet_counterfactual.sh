#!/bin/bash
#SBATCH --job-name=celnet-cf
#SBATCH --output=logs/celnet_cf_%A_%a.out
#SBATCH --error=logs/celnet_cf_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=dept_dms
#SBATCH --qos=high_dept_dms
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-5    # 2 pairing policies x 3 run seeds

# CEL-Net rounds (configs/diversity_mwm_acc_220r.yaml: 10 models incl. the oracle, 220 rounds) with a counterfactual
# KD epoch from the best peer measured before every oracle CE update.

set -euo pipefail

POLICIES=(MWM_AccDiff MWM_ClassDist)
RUN_SEEDS=(42 43 44)

N_SEEDS=${#RUN_SEEDS[@]}
POLICY=${POLICIES[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
RUN_SEED=${RUN_SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
OUT_DIR="Geometry of Mixed-Loss/results/celnet_counterfactual/${POLICY}/seed${RUN_SEED}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
set +u
conda activate /project/ikoutis/an57/conda_envs/torch-cuda
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python -c "import networkx" || { echo "ERROR: networkx is not installed in this environment (needed by CEL-Net pairing)"; exit 1; }

echo "CEL-Net counterfactual: policy ${POLICY}, run seed ${RUN_SEED}, 220 rounds -> ${OUT_DIR}"
python "Geometry of Mixed-Loss/experiments/celnet_ce_counterfactual.py" \
    --config configs/diversity_mwm_acc_220r.yaml \
    --pairing_strategy "$POLICY" \
    --run_seed "$RUN_SEED" \
    --n_rounds 220 \
    --data_root ./data \
    --out "$OUT_DIR"

echo "Done."
