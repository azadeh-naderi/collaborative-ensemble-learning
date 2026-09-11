"""Train one ResNet-18 student for the loss-switch experiment (ce_only, kd_then_ce, or kd_only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from switch_common import build_loaders, build_teacher_arch, ce_optimum_gap, kd_loss, metrics, predict
from src.celnet.models import get_model_class, initialize_model
from src.celnet.utils import TimeLogger, seed_everything, save_json

BASE_LR = 0.1
PHASE_LOSSES = {"ce_only": ("ce", "ce"), "kd_then_ce": ("kd", "ce"), "kd_only": ("kd", "kd")}


def train_epoch(student, teacher, loader, opt, device, loss_type, alpha, temperature):
    student.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = student(x)
        if loss_type == "ce":
            loss = F.cross_entropy(logits, y)
        else:
            with torch.no_grad():
                teacher_logits = teacher(x)
            loss = alpha * kd_loss(logits, teacher_logits, temperature)
            if alpha < 1.0:
                loss = loss + (1.0 - alpha) * F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
        count += y.size(0)
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(PHASE_LOSSES), required=True)
    ap.add_argument("--teacher_ckpt", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--phase1_epochs", type=int, default=100)
    ap.add_argument("--phase2_epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1.0, help="KD weight; 1.0 = pure KL, no true labels in phase 1")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--ft_lr", type=float, default=1e-3, help="constant LR for phase 2 (phase 1 ends at 1e-3)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    first, second = PHASE_LOSSES[args.mode]
    p1 = args.phase1_epochs
    losses = [first] * p1 + [second] * args.phase2_epochs
    if "kd" in losses and not args.teacher_ckpt:
        raise SystemExit(f"--teacher_ckpt is required for mode {args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, probe_loader = build_loaders(
        args.seed, args.split_seed, args.batch_size, args.num_workers, args.data_root)

    student = initialize_model(get_model_class("resnet"), 3, False, 10, seed=args.seed, device=device)

    teacher = teacher_meta = val_teacher_pred = test_teacher_pred = None
    if args.teacher_ckpt:
        teacher = build_teacher_arch().to(device)
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        meta_path = Path(args.teacher_ckpt).with_suffix(".json")
        teacher_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        val_teacher_pred = predict(teacher, val_loader, device)[0].argmax(dim=1)
        test_teacher_pred = predict(teacher, test_loader, device)[0].argmax(dim=1)

    opt = torch.optim.SGD(student.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4)
    milestones = [max(1, int(0.5 * p1)), max(2, int(0.75 * p1))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    init = metrics(*predict(student, val_loader, device), val_teacher_pred)
    log = {
        "mode": args.mode, "seed": args.seed, "split_seed": args.split_seed,
        "teacher_ckpt": args.teacher_ckpt, "teacher": teacher_meta,
        "phase1_epochs": p1, "phase2_epochs": args.phase2_epochs,
        "alpha": args.alpha, "temperature": args.temperature,
        "base_lr": BASE_LR, "ft_lr": args.ft_lr, "phase1_milestones": milestones,
        "val_init": init, "train_probe_init": ce_optimum_gap(student, probe_loader, device),
        "epoch": [], "phase": [], "lr": [], "train_loss": [],
        "val_acc": [], "val_loss": [], "val_ece": [], "val_conf": [], "val_agree": [], "delta_acc": [],
        "probe_ce_loss": [], "probe_acc": [], "probe_ce_grad_norm": [],
    }
    prev_acc = init["acc"]

    with TimeLogger():
        for epoch, loss_type in enumerate(losses, start=1):
            if epoch == p1 + 1:
                for group in opt.param_groups:
                    group["lr"] = args.ft_lr
            lr = opt.param_groups[0]["lr"]
            train_loss = train_epoch(student, teacher, train_loader, opt, device,
                                     loss_type, args.alpha, args.temperature)
            if epoch <= p1:
                sched.step()

            val = metrics(*predict(student, val_loader, device), val_teacher_pred)
            probe = ce_optimum_gap(student, probe_loader, device)
            log["epoch"].append(epoch)
            log["phase"].append(loss_type)
            log["lr"].append(lr)
            log["train_loss"].append(train_loss)
            log["val_acc"].append(val["acc"])
            log["val_loss"].append(val["loss"])
            log["val_ece"].append(val["ece"])
            log["val_conf"].append(val["conf"])
            log["val_agree"].append(val.get("agree"))
            log["delta_acc"].append(val["acc"] - prev_acc)
            log["probe_ce_loss"].append(probe["ce_loss"])
            log["probe_acc"].append(probe["acc"])
            log["probe_ce_grad_norm"].append(probe["ce_grad_norm"])
            prev_acc = val["acc"]

            if epoch == p1:
                log["test_at_switch"] = metrics(*predict(student, test_loader, device), test_teacher_pred)
                torch.save(student.state_dict(), out_dir / "student_at_switch.pt")
            if epoch % 10 == 0 or epoch == len(losses):
                agree = f"  agree {val['agree']:.1f}" if "agree" in val else ""
                print(f"ep {epoch:3d} [{loss_type}] lr {lr:.0e}  val acc {val['acc']:.2f}"
                      f"  delta {log['delta_acc'][-1]:+.2f}  ece {val['ece']:.3f}{agree}"
                      f"  train CE {probe['ce_loss']:.3f} |grad| {probe['ce_grad_norm']:.3f}")

    log["test_final"] = metrics(*predict(student, test_loader, device), test_teacher_pred)
    save_json(log, out_dir / "results.json")
    torch.save(student.state_dict(), out_dir / "student_final.pt")
    print(f"test acc at switch {log['test_at_switch']['acc']:.2f} -> final {log['test_final']['acc']:.2f}"
          f"  saved to {out_dir}")


if __name__ == "__main__":
    main()
