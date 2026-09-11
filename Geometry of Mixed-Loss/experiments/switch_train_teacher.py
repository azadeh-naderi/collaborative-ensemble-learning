"""Train one ResNet-50 teacher for the loss-switch experiment, with CE on true labels or pure KD from a source teacher."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from switch_common import build_loaders, build_teacher_arch, ce_optimum_gap, kd_loss, metrics, predict
from src.celnet.utils import seed_everything, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["ce", "kd"], default="ce",
                    help="ce: true labels; kd: pure KL to --source_ckpt, no labels")
    ap.add_argument("--source_ckpt", default=None, help="teacher to distill from (objective kd)")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out", required=True, help="checkpoint path (.pt); metadata is written next to it as .json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()
    if args.objective == "kd" and not args.source_ckpt:
        raise SystemExit("--source_ckpt is required for objective kd")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    train_loader, val_loader, test_loader, probe_loader = build_loaders(
        args.seed, args.split_seed, args.batch_size, args.num_workers, args.data_root)

    teacher = build_teacher_arch().to(device)
    source = test_source_pred = None
    if args.objective == "kd":
        source = build_teacher_arch().to(device)
        source.load_state_dict(torch.load(args.source_ckpt, map_location=device, weights_only=True))
        source.eval()
        for p in source.parameters():
            p.requires_grad_(False)
        test_source_pred = predict(source, test_loader, device)[0].argmax(dim=1)

    opt = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    milestones = [max(1, int(0.5 * args.epochs)), max(2, int(0.75 * args.epochs))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = teacher(x)
            if args.objective == "ce":
                loss = F.cross_entropy(logits, y)
            else:
                with torch.no_grad():
                    source_logits = source(x)
                loss = kd_loss(logits, source_logits, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            val = metrics(*predict(teacher, val_loader, device))
            print(f"epoch {epoch}/{args.epochs}  val acc {val['acc']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), out)
    save_json({
        "objective": args.objective, "source_ckpt": args.source_ckpt,
        "temperature": args.temperature if args.objective == "kd" else None,
        "epochs": args.epochs, "seed": args.seed, "split_seed": args.split_seed, "milestones": milestones,
        "val": metrics(*predict(teacher, val_loader, device)),
        "test": metrics(*predict(teacher, test_loader, device), test_source_pred),
        "train_probe": ce_optimum_gap(teacher, probe_loader, device),
    }, out.with_suffix(".json"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
