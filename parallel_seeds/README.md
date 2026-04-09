# Parallel Multi-Seed Experiments

Run multiple CELNet experiments with different random seeds simultaneously on a single GPU (e.g. A800).

Each seed launches as its own subprocess so CUDA initializes cleanly in each one. All subprocesses share the same GPU — on an A800 (80 GB), ResNet18+CIFAR-10 uses ~1-2 GB per run, so 8-10 concurrent experiments fit comfortably.

## Files

| File | Purpose |
|------|---------|
| `run_parallel.py` | Main launcher — spawns one subprocess per seed |
| `aggregate_results.py` | Collects `summary_r*.json` files and reports mean ± std across seeds |
| `slurm_parallel.sh` | SLURM batch script for cluster submission |

## Output layout

Each seed writes its results to its own subdirectory:

```
results/parallel_seeds/
├── seed_42/
│   ├── run.log               # stdout/stderr for this seed
│   ├── summary_r42.json
│   ├── 10_oracle_cum_tensor_resnet_...r42.pt
│   ├── 10_non_oracle_cum_tensor_...r42.pt
│   └── *.png
├── seed_123/
│   └── ...
└── aggregated.json           # written by aggregate_results.py --save
```

## Usage

### Local / interactive

```bash
# Run 5 seeds, all at once
python parallel_seeds/run_parallel.py --seeds 42 123 456 789 1337

# Cap at 3 concurrent experiments
python parallel_seeds/run_parallel.py --seeds 42 123 456 789 1337 --max-workers 3

# Custom config + extra args forwarded to main.py
python parallel_seeds/run_parallel.py \
    --seeds 42 123 456 789 1337 \
    --config configs/default.yaml \
    --max-workers 4 \
    --results-dir results/run_A \
    -- --n-rounds 50 --pairing-strategy euclidean --num-workers 2
```

Arguments after `--` are forwarded verbatim to `main.py`.

### SLURM

```bash
# Default: 5 seeds, 5 concurrent, A800 GPU
sbatch parallel_seeds/slurm_parallel.sh

# Override seeds and concurrency without editing the file
sbatch --export=SEEDS="1 2 3 4 5 6 7 8",MAX_WORKERS=8 parallel_seeds/slurm_parallel.sh

# Also override config and extra experiment args
sbatch --export=SEEDS="10 20 30",MAX_WORKERS=3,CONFIG="configs/default.yaml",EXTRA_ARGS="--n-rounds 50" \
    parallel_seeds/slurm_parallel.sh
```

The SLURM script automatically runs `aggregate_results.py` after all seeds finish.

### Aggregating results manually

```bash
python parallel_seeds/aggregate_results.py --results-dir results/parallel_seeds
python parallel_seeds/aggregate_results.py --results-dir results/parallel_seeds --save
```

Prints a table of mean ± std for ensemble accuracy, confidence, disagreement, distilled student accuracy, and per-model accuracy across all completed seeds.

## Tips

- **How many concurrent experiments?** Start with `--max-workers` equal to the number of seeds and watch GPU memory with `nvidia-smi`. Reduce if you run out.
- **Data loading contention:** Each experiment spawns `--num-workers` DataLoader workers. If CPU or I/O becomes a bottleneck, pass `-- --num-workers 2` to halve the per-experiment worker count.
- **Reproducibility:** Each seed controls both the global RNG (`run_seed`) and the run identifier (`run_id`), so results are fully reproducible per seed and never overwrite each other.
