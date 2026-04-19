# src/data/datasets.py
"""
Dataset loaders for all benchmarks at NATIVE resolution (no 224×224 upscaling).

Datasets (auto-download):
    - CIFAR-10   (50k train, 10k test, 32×32)
    - CIFAR-100  (50k train, 10k test, 32×32)
    - SVHN       (73k train, 26k test, 32×32)

Datasets (manual download required):
    - CINIC-10   (90k train, 90k val/test, 32×32)
    - Tiny-ImageNet (100k train, 10k val, 64×64, 200 classes)

For Kaggle/Colab (lite mode), pass dataset_fraction < 1.0 to subsample
the training split — e.g., fraction=0.2 gives a genuine 10k-sample
small-dataset regime consistent with the paper's thesis.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

from training.augmentations import build_transforms, DATASET_STATS


DATA_ROOT = Path(os.environ.get("SMOLDATA_ROOT", "./data"))


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 / CIFAR-100
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
    fraction: float = 1.0,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader, optionally subsampling the dataset."""
    if fraction < 1.0:
        from .subsets import stratified_subset
        dataset = stratified_subset(dataset, fraction, seed=seed)
    pin = num_workers > 0  # only pin when workers are available
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),  # avoids re-spawning workers each epoch
    )


def get_cifar10(
    batch_size: int = 64,
    num_workers: int = 2,
    fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 loader. fraction < 1.0 enables lite/small-dataset mode."""
    train_tf = build_transforms("cifar10", img_size=32, is_train=True)
    val_tf = build_transforms("cifar10", img_size=32, is_train=False)

    train_ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, transform=train_tf, download=True)
    val_ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, transform=val_tf, download=True)

    return (
        _make_loader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                     drop_last=True, fraction=fraction, seed=seed),
        _make_loader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
    )


def get_cifar100(
    batch_size: int = 64,
    num_workers: int = 2,
    fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-100 loader. fraction < 1.0 enables lite/small-dataset mode."""
    train_tf = build_transforms("cifar100", img_size=32, is_train=True)
    val_tf = build_transforms("cifar100", img_size=32, is_train=False)

    train_ds = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, transform=train_tf, download=True)
    val_ds = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, transform=val_tf, download=True)

    return (
        _make_loader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                     drop_last=True, fraction=fraction, seed=seed),
        _make_loader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SVHN
# ─────────────────────────────────────────────────────────────────────────────

def get_svhn(
    batch_size: int = 64,
    num_workers: int = 2,
    fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = build_transforms("svhn", img_size=32, is_train=True)
    val_tf = build_transforms("svhn", img_size=32, is_train=False)

    train_ds = torchvision.datasets.SVHN(DATA_ROOT / "svhn", split="train", transform=train_tf, download=True)
    val_ds = torchvision.datasets.SVHN(DATA_ROOT / "svhn", split="test", transform=val_tf, download=True)

    return (
        _make_loader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                     drop_last=True, fraction=fraction, seed=seed),
        _make_loader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CINIC-10 (ImageNet + CIFAR-10 hybrid)
# Must be downloaded manually from: https://datashare.ed.ac.uk/handle/10283/3192
# ─────────────────────────────────────────────────────────────────────────────

def get_cinic10(
    root: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(root) if root else DATA_ROOT / "cinic10"
    if not root.exists():
        raise FileNotFoundError(
            f"CINIC-10 not found at {root}. Download from "
            "https://datashare.ed.ac.uk/handle/10283/3192 and extract."
        )

    train_tf = build_transforms("cinic10", img_size=32, is_train=True)
    val_tf = build_transforms("cinic10", img_size=32, is_train=False)

    train_ds = torchvision.datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(root / "test", transform=val_tf)

    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tiny-ImageNet
# Must be downloaded: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# ─────────────────────────────────────────────────────────────────────────────

def get_tiny_imagenet(
    root: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(root) if root else DATA_ROOT / "tiny-imagenet-200"
    if not root.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {root}. "
            "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        )

    train_tf = build_transforms("tiny_imagenet", img_size=64, is_train=True)
    val_tf = build_transforms("tiny_imagenet", img_size=64, is_train=False)

    train_ds = torchvision.datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(root / "val" / "images", transform=val_tf)

    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "cifar10":       {"fn": get_cifar10,       "num_classes": 10,  "img_size": 32},
    "cifar100":      {"fn": get_cifar100,      "num_classes": 100, "img_size": 32},
    "svhn":          {"fn": get_svhn,          "num_classes": 10,  "img_size": 32},
    "cinic10":       {"fn": get_cinic10,       "num_classes": 10,  "img_size": 32},
    "tiny_imagenet": {"fn": get_tiny_imagenet, "num_classes": 200, "img_size": 64},
}

# Datasets available on Kaggle/Colab without manual download
LITE_DATASETS = ["cifar10", "cifar100"]


def get_dataset(
    name: str,
    batch_size: int = 64,
    fraction: float = 1.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Returns (train_loader, val_loader, num_classes, img_size).

    Args:
        fraction: Fraction of training data to use (lite mode: 0.2 → 10k for CIFAR-10).
                  Validation set is always full (for reliable evaluation).
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[name]
    train_loader, val_loader = info["fn"](batch_size=batch_size, fraction=fraction, seed=seed)
    n_train = len(train_loader.dataset)
    print(f"[dataset] {name}: {n_train} train samples (fraction={fraction:.0%}), "
          f"num_classes={info['num_classes']}, img_size={info['img_size']}")
    return train_loader, val_loader, info["num_classes"], info["img_size"]
