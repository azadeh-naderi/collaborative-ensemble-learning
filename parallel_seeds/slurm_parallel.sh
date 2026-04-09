#!/bin/bash
# =============================================================================
# SLURM script: run multiple CELNet seeds in parallel on a single A800 GPU
#
# Submit with:
#   sbatch parallel_seeds/slurm_parallel.sh
#
# Override seeds / workers at submission time:
#   sbatch --export=SEEDS="1 2 3 4 5",MAX_WORKERS=4 parallel_seeds/slurm_parallel.sh
# =============================================================================

#SBATCH --job-name=celnet-multiseed
#SBATCH --output=logs/celnet_multiseed_%j.out
#SBATCH --error=logs/celnet_multiseed_%j.err
#SBATCH --gres=gpu:a800:1          # one A800 GPU
#SBATCH --cpus-per-task=16         # shared across all concurrent experiments
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ── user-configurable defaults (override via --export) ──────────────────────
SEEDS="${SEEDS:-42 123 456 789 1337}"
MAX_WORKERS="${MAX_WORKERS:-5}"         # how many experiments run simultaneously
CONFIG="${CONFIG:-configs/default.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/parallel_seeds_${SLURM_JOB_ID}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # e.g. "--n-rounds 50 --pairing-strategy euclidean"
# ─────────────────────────────────────────────────────────────────────────────

# Activate your conda / venv environment here, e.g.:
# module load cuda/12.1
# conda activate celnet
# source ~/venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================"
echo " Job ID     : ${SLURM_JOB_ID}"
echo " Node       : ${SLURMD_NODENAME:-local}"
echo " GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Seeds      : ${SEEDS}"
echo " Max workers: ${MAX_WORKERS}"
echo " Config     : ${CONFIG}"
echo " Results dir: ${RESULTS_DIR}"
echo "======================================================"

mkdir -p logs

cd "$REPO_ROOT"

# Build the --seeds list for the Python launcher
SEEDS_ARGS=""
for s in $SEEDS; do
    SEEDS_ARGS="$SEEDS_ARGS $s"
done

python parallel_seeds/run_parallel.py \
    --seeds $SEEDS_ARGS \
    --config "$CONFIG" \
    --max-workers "$MAX_WORKERS" \
    --results-dir "$RESULTS_DIR" \
    ${EXTRA_ARGS:+-- $EXTRA_ARGS}

echo ""
echo "All seeds finished. Aggregating results..."

python parallel_seeds/aggregate_results.py \
    --results-dir "$RESULTS_DIR" \
    --save

echo "Done. Results: $RESULTS_DIR"
