"""
Exp 5 — Gradient Angle Logger.

Given a directory of model checkpoints saved during a CEL-Net or alpha-sweep run,
computes cos(g_CE, g_KL) and gradient norms at each checkpoint, appending the
measurements to a `gradient_angle_log.json` in the same directory.

Can also be imported and called inline during training (see gradient_cosine() in
run_alpha_sweep.py for the training-time version).

Usage (post-hoc on saved checkpoints):
    python experiments/geometry/log_gradient_angle.py \
        --ckpt_dir results/geometry/alpha_sweep/alpha0.9_seed42/ \
        --data_root ./data \
        --temperature 4.0 \
        --alpha 0.9 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as tvm

from src.celnet.data import cifar_dataset
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import seed_everything

PROBE_N = 1000


@torch.no_grad()
def collect_probe(loader, n, device):
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x); labs.append(y)
        if sum(b.shape[0] for b in imgs) >= n:
            break
    return torch.cat(imgs)[:n].to(device), torch.cat(labs)[:n].to(device)


def gradient_cosine(model, teacher, probe_imgs, probe_labs, temperature, alpha):
    """Returns (cosine_similarity, g_ce_norm, g_kl_norm)."""
    model.train()

    model.zero_grad()
    logits = model(probe_imgs)
    F.cross_entropy(logits, probe_labs).backward()
    g_ce = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])

    model.zero_grad()
    logits = model(probe_imgs)
    with torch.no_grad():
        t_soft = F.softmax(teacher(probe_imgs) / temperature, dim=1)
    (temperature**2 * F.kl_div(
        F.log_softmax(logits / temperature, dim=1), t_soft, reduction="batchmean"
    )).backward()
    g_kl = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])

    model.zero_grad()
    cos = F.cosine_similarity(g_ce.unsqueeze(0), g_kl.unsqueeze(0)).item()
    return cos, g_ce.norm().item(), g_kl.norm().item()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    ckpt_dir = Path(args.ckpt_dir)

    _, val_loader, _ = cifar_dataset(batch_size=64, seed=args.seed, num_workers=4, root=args.data_root)
    probe_imgs, probe_labs = collect_probe(val_loader, PROBE_N, device)

    # load teacher
    teacher = tvm.resnet50(weights=None, num_classes=10).to(device)
    teacher_path = ckpt_dir / "teacher.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_path}")
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # find all student checkpoints (named student_roundNNN.pt)
    ckpts = sorted(ckpt_dir.glob("student_round*.pt"), key=lambda p: int(p.stem.split("round")[-1]))
    if not ckpts:
        # fallback: single final checkpoint
        ckpts = list(ckpt_dir.glob("student_final.pt"))

    if not ckpts:
        raise FileNotFoundError("No student checkpoints found in " + str(ckpt_dir))

    model_cls = get_model_class("resnet")
    results = []
    for ckpt in ckpts:
        rnd = int(ckpt.stem.split("round")[-1]) if "round" in ckpt.stem else -1
        model = initialize_model(model_cls, 3, False, 10, seed=args.seed, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        cos, gce_n, gkl_n = gradient_cosine(model, teacher, probe_imgs, probe_labs, args.temperature, args.alpha)
        results.append({"round": rnd, "cos": cos, "g_ce_norm": gce_n, "g_kl_norm": gkl_n})
        print(f"  round={rnd:4d}  cos={cos:.4f}  ‖g_CE‖={gce_n:.3f}  ‖g_KL‖={gkl_n:.3f}")

    out_path = ckpt_dir / "gradient_angle_log.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "measurements": results}, f, indent=2)
    print(f"\nSaved gradient angle log to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha",       type=float, default=0.9)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--data_root",   default="./data")
    run(parser.parse_args())
