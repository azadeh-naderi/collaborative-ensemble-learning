"""Run multiple CELNet experiments with different random seeds in parallel on a single GPU.

Each seed spawns a separate Python subprocess so CUDA initializes cleanly in each one.
All subprocesses share the same GPU (works well on high-memory cards like A800).
Results land in --results-dir/seed_<N>/ with a per-seed run.log.

Usage examples
--------------
# 5 seeds, all running at once (default)
python parallel_seeds/run_parallel.py --seeds 42 123 456 789 1337

# Limit to 3 concurrent experiments
python parallel_seeds/run_parallel.py --seeds 42 123 456 789 1337 --max-workers 3

# Custom config + extra args forwarded to main.py
python parallel_seeds/run_parallel.py \\
    --seeds 1 2 3 4 5 \\
    --config configs/default.yaml \\
    --max-workers 4 \\
    --results-dir results/run_A \\
    -- --n-rounds 50 --pairing-strategy euclidean --num-workers 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"


# ---------------------------------------------------------------------------
# Per-seed worker
# ---------------------------------------------------------------------------

def _stream_output(proc: subprocess.Popen, seed: int, log_path: Path) -> None:
    """Read proc.stdout line-by-line, write to stdout and log file with a seed tag."""
    prefix = f"[seed={seed}] "
    with open(log_path, "w", encoding="utf-8") as fh:
        for line in proc.stdout:  # type: ignore[union-attr]
            tagged = prefix + line
            sys.stdout.write(tagged)
            sys.stdout.flush()
            fh.write(tagged)


def run_one_seed(
    seed: int,
    base_cmd: list[str],
    results_dir: Path,
) -> tuple[int, int]:
    """Launch main.py for *seed*, stream its output, return (seed, returncode)."""
    seed_dir = results_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "run.log"

    cmd = base_cmd + [
        "--run-seed", str(seed),
        "--run_id",   str(seed),
        "--results-dir", str(seed_dir),
    ]

    print(f"[seed={seed}] Starting  cmd: {' '.join(str(c) for c in cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )

    t = threading.Thread(target=_stream_output, args=(proc, seed, log_path), daemon=True)
    t.start()
    proc.wait()
    t.join()

    status = "DONE" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[seed={seed}] {status}  log -> {log_path}", flush=True)
    return seed, proc.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CELNet experiments in parallel, one subprocess per seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help="List of integer seeds — one experiment is launched per seed.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (resolved relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max simultaneous experiments. Default: run all seeds at once.",
    )
    parser.add_argument(
        "--results-dir", dest="results_dir", type=str, default="results/parallel_seeds",
        help="Parent output directory. Each seed writes to <results-dir>/seed_<N>/.",
    )
    # Everything after '--' is forwarded verbatim to main.py
    parser.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="Extra args forwarded to main.py, e.g. -- --n-rounds 50 --model lenet",
    )
    args = parser.parse_args()

    extra_args: list[str] = args.extra
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build the part of the command that is the same for every seed
    base_cmd: list[str] = [sys.executable, str(MAIN_PY)]
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        base_cmd += ["--config", str(config_path)]
    base_cmd += extra_args

    n_seeds = len(args.seeds)
    max_workers = args.max_workers or n_seeds

    print(
        f"\nLaunching {n_seeds} experiment(s), up to {max_workers} concurrent.\n"
        f"Seeds       : {args.seeds}\n"
        f"Results dir : {results_dir}\n",
        flush=True,
    )

    exit_codes: dict[int, int] = {}

    # ThreadPoolExecutor manages the blocking subprocess.wait() calls; the
    # actual GPU work happens inside each subprocess, not in these threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_seed, seed, base_cmd, results_dir): seed
            for seed in args.seeds
        }
        for future in concurrent.futures.as_completed(futures):
            seed, rc = future.result()
            exit_codes[seed] = rc

    print("\n=== Final status ===")
    all_ok = True
    for seed in sorted(exit_codes):
        ok = exit_codes[seed] == 0
        tag = "OK  " if ok else f"FAIL (exit {exit_codes[seed]})"
        print(f"  seed={seed:>6}  {tag}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
