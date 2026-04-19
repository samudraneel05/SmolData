# src/data/subsets.py
"""
Deterministic stratified data-size ablation.
Supports {10, 25, 50, 75, 100}% splits of any dataset's training set.

This implements Phase 3's data-size ablation study.
Stratification ensures class balance is preserved at every percentage.
"""

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset


def stratified_subset(
    dataset: Dataset,
    fraction: float,
    seed: int = 42,
) -> Subset:
    """
    Return a stratified subset of `dataset` containing `fraction` (0–1) of each class.

    Args:
        dataset:  Any dataset that returns (image, label) pairs.
        fraction: Fraction of data to keep per class (0.0 < fraction ≤ 1.0).
        seed:     Random seed for reproducibility.
    """
    assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"
    if fraction == 1.0:
        return Subset(dataset, list(range(len(dataset))))

    rng = np.random.default_rng(seed=seed)

    # Collect indices per class
    labels = _get_labels(dataset)
    classes = np.unique(labels)
    selected_indices = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        n_keep = max(1, int(len(cls_indices) * fraction))
        chosen = rng.choice(cls_indices, size=n_keep, replace=False)
        selected_indices.extend(chosen.tolist())

    return Subset(dataset, sorted(selected_indices))


def _get_labels(dataset: Dataset) -> np.ndarray:
    """Extract labels from a dataset efficiently."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    # Fallback: iterate (slow)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def get_subset_loaders(
    dataset: Dataset,
    fractions: List[float],
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> List[DataLoader]:
    """
    Return a DataLoader for each fraction in the list.
    Used for Phase 3 data-size ablation.
    """
    loaders = []
    for frac in fractions:
        subset = stratified_subset(dataset, frac, seed=seed)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        loaders.append(loader)
    return loaders


# Standard ablation fractions from the experimental plan
ABLATION_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
