"""
At each CE round of the repeated short-cycle experiment, does the CE step
*raise* the CE loss it is minimizing? Discriminates two accounts of the
switch harm that gradient-norm growth alone cannot separate:

  - over-stepping / instability: the CE update overshoots in a steep
    region, so TRAIN CE loss rises after a CE step on the training set.
  - overfitting: the CE update still lowers TRAIN CE loss, but VAL CE loss
    rises and val accuracy falls.

Reads ce_train_loss_before/after and ce_val_loss_before/after from
results.json (written by run_repeated_cycle.py; one entry per CE exposure,
aligned in order with the CE-phase rows of delta_acc). Runs predating
that logging are skipped with a message.

Reports per group, split at --split_exposure (default 11, the first LR
milestone under the default schedule): mean delta_acc, fraction of CE
steps that raised train loss / val loss, and mean loss changes. Also
writes one row per CE exposure to --out for reuse.

Usage (from repo root):
    python "Geometry of Mixed-Loss/experiments/analyze_switch_loss.py" \
        --results_dir "Geometry of Mixed-Loss/results/repeated_cycle" \
        --out "Geometry of Mixed-Loss/results/repeated_cycle/analysis/switch_loss.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path

LOSS_KEYS = ["ce_train_loss_before", "ce_train_loss_after",
             "ce_val_loss_before", "ce_val_loss_after"]


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", default="k*_seed*", dest="glob_pattern")
    ap.add_argument("--group_field", default="k")
    ap.add_argument("--split_exposure", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = []
    skipped = 0
    for run_dir in sorted(Path(args.results_dir).glob(args.glob_pattern)):
        f = run_dir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        if not all(k in data for k in LOSS_KEYS):
            skipped += 1
            continue
        ce_deltas = [d for d, p in zip(data["delta_acc"], data["phase"]) if p == "ce"]
        n = min(len(ce_deltas), *(len(data[k]) for k in LOSS_KEYS))
        for j in range(n):
            tb, ta = data["ce_train_loss_before"][j], data["ce_train_loss_after"][j]
            vb, va = data["ce_val_loss_before"][j], data["ce_val_loss_after"][j]
            events.append({
                "group": data[args.group_field], "seed": data.get("seed"),
                "exposure_idx": j + 1, "delta_acc": ce_deltas[j],
                "train_loss_before": tb, "train_loss_after": ta, "train_loss_change": ta - tb,
                "val_loss_before": vb, "val_loss_after": va, "val_loss_change": va - vb,
            })

    if skipped:
        print(f"Skipped {skipped} run(s) without CE loss logging.")
    if not events:
        raise SystemExit("No runs with CE loss logging found -- nothing to analyze.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)

    s = args.split_exposure
    print(f"{'group':>6} | {'exposures':>10} | {'n':>4} | {'mean dAcc':>9} | "
          f"{'train loss up':>13} | {'mean dTrainLoss':>15} | {'val loss up':>11} | {'mean dValLoss':>13}")
    print("-" * 104)
    for g in sorted({e["group"] for e in events}):
        for label, keep in [(f"1-{s}", lambda e: e["exposure_idx"] <= s),
                            (f"{s + 1}+", lambda e: e["exposure_idx"] > s)]:
            sub = [e for e in events if e["group"] == g and keep(e)]
            if not sub:
                continue
            print(f"{g:>6} | {label:>10} | {len(sub):>4} | "
                  f"{stats.mean(e['delta_acc'] for e in sub):>+9.3f} | "
                  f"{sum(e['train_loss_change'] > 0 for e in sub) / len(sub):>13.2f} | "
                  f"{stats.mean(e['train_loss_change'] for e in sub):>+15.4f} | "
                  f"{sum(e['val_loss_change'] > 0 for e in sub) / len(sub):>11.2f} | "
                  f"{stats.mean(e['val_loss_change'] for e in sub):>+13.4f}")

    d = [e["delta_acc"] for e in events]
    print(f"\nPooled over {len(events)} CE exposures:")
    print(f"  Spearman(delta_acc, train_loss_change) = "
          f"{spearman(d, [e['train_loss_change'] for e in events]):+.3f}")
    print(f"  Spearman(delta_acc, val_loss_change)   = "
          f"{spearman(d, [e['val_loss_change'] for e in events]):+.3f}")
    print(f"\nPer-exposure rows written to {out}")


if __name__ == "__main__":
    main()
