"""Shared helpers for the loss-switch experiment."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader, Subset

from src.celnet.data import cifar_dataset, seed_worker

PROBE_SIZE = 5000


def build_loaders(seed: int, split_seed: int, batch_size: int, num_workers: int, root: str):
    # split_seed fixes the train/val split for every run, so a shared teacher never trains on a student's val images
    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size, seed=split_seed, num_workers=num_workers, root=root)
    generator = torch.Generator().manual_seed(seed)
    train_set = train_loader.dataset
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=num_workers > 0, pin_memory=True)

    # fixed, unaugmented subset of the training images, identical for every run; used to measure the CE optimum gap
    perm = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(split_seed))
    probe_idx = [train_set.indices[i] for i in perm[:PROBE_SIZE].tolist()]
    probe_loader = DataLoader(
        Subset(val_loader.dataset.dataset, probe_idx), batch_size=256, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker, persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader, test_loader, probe_loader


def build_teacher_arch(num_classes: int = 10):
    return tvm.resnet50(weights=None, num_classes=num_classes)


def kd_loss(logits, teacher_logits, temperature: float):
    return temperature ** 2 * F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def ce_optimum_gap(model, loader, device):
    """CE loss, accuracy and full-gradient norm of CE on true labels (eval-mode BN); small values = near the CE minimum."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [torch.zeros_like(p) for p in params]
    n = len(loader.dataset)
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        batch_grads = torch.autograd.grad(loss / n, params)
        for g, bg in zip(grads, batch_grads):
            g.add_(bg)
        total_loss += loss.item()
        correct += logits.argmax(dim=1).eq(y).sum().item()
    grad_norm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
    return {"ce_loss": total_loss / n, "acc": 100.0 * correct / n, "ce_grad_norm": grad_norm}


def expected_calibration_error(probs, labels, n_bins: int = 15):
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(conf[in_bin].mean().item() - correct[in_bin].mean().item())
    return ece


def metrics(logits, labels, teacher_pred=None):
    probs = logits.softmax(dim=1)
    pred = probs.argmax(dim=1)
    out = {
        "acc": 100.0 * pred.eq(labels).float().mean().item(),
        "loss": F.cross_entropy(logits, labels).item(),
        "ece": expected_calibration_error(probs, labels),
        "conf": probs.max(dim=1).values.mean().item(),
    }
    if teacher_pred is not None:
        out["agree"] = 100.0 * pred.eq(teacher_pred).float().mean().item()
    return out
