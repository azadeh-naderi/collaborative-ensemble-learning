from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_validation_set_kfold(dataset, k: int, seed: int):
    length = len(dataset)
    lengths = [length // k] * k
    for i in range(length % k):
        lengths[i] += 1
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator=g)


class SymmetricLabelNoiseDataset(torch.utils.data.Dataset):
    """Wraps a dataset, flipping each label to a uniformly random wrong class with probability `rate`."""

    def __init__(self, dataset, num_classes: int, rate: float, seed: int):
        self.dataset = dataset
        self.num_classes = num_classes
        self.rate = rate

        rng = np.random.RandomState(seed)
        n = len(dataset)
        flip_mask = rng.random(n) < rate
        self.labels = []
        flipped = 0
        for i in range(n):
            _, label = dataset[i]
            if flip_mask[i]:
                choices = [c for c in range(num_classes) if c != label]
                label = int(rng.choice(choices))
                flipped += 1
            self.labels.append(label)
        self.actual_noise_rate = flipped / n if n > 0 else 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.labels[idx]


def noisy_loader(loader: DataLoader, num_classes: int, rate: float, seed: int) -> Tuple[DataLoader, float]:
    """Build a DataLoader over a label-noised copy of `loader`'s dataset, mirroring its other settings."""
    noisy_dataset = SymmetricLabelNoiseDataset(loader.dataset, num_classes, rate, seed)
    new_loader = DataLoader(
        noisy_dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=True,
    )
    return new_loader, noisy_dataset.actual_noise_rate


def cifar_dataset(batch_size: int, seed: int = 42, num_workers: int = 4, root: str = "./data"):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    full_train_aug = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=root, train=True, download=True, transform=test_transform)

    train_size = int(0.9 * len(full_train_aug))
    val_size = len(full_train_aug) - train_size

    g_split = torch.Generator().manual_seed(seed)
    train_subset_aug, val_subset_indices = random_split(
        range(len(full_train_aug)), [train_size, val_size], generator=g_split
    )

    train_dataset = torch.utils.data.Subset(full_train_aug, train_subset_aug.indices)
    val_dataset = torch.utils.data.Subset(full_train_eval, val_subset_indices.indices)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    g_loader = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, worker_init_fn=seed_worker, persistent_workers=(num_workers > 0), pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g_loader, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
