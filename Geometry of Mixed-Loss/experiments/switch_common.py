"""Shared helpers for the loss-switch experiment."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader

from src.celnet.data import cifar_dataset, seed_worker


def build_loaders(seed: int, split_seed: int, batch_size: int, num_workers: int, root: str):
    # split_seed fixes the train/val split for every run, so a shared teacher never trains on a student's val images
    train_loader, val_loader, test_loader = cifar_dataset(
        batch_size, seed=split_seed, num_workers=num_workers, root=root)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_teacher_arch(num_classes: int = 10):
    return tvm.resnet50(weights=None, num_classes=num_classes)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


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
    }
    if teacher_pred is not None:
        out["agree"] = 100.0 * pred.eq(teacher_pred).float().mean().item()
    return out
