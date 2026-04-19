# src/evaluation/metrics.py
"""
Evaluation metrics for all phases.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: int = 1,
) -> Dict[str, float]:
    """Compute top-1 and optionally top-5 accuracy."""
    model.eval()
    correct_1, correct_k, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        _, pred = logits.topk(max(1, top_k), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        correct_1 += correct[:1].reshape(-1).float().sum().item()
        if top_k >= 5:
            correct_k += correct[:5].reshape(-1).float().sum().item()
        total += labels.size(0)

    metrics = {"top1_acc": correct_1 / total}
    if top_k >= 5:
        metrics["top5_acc"] = correct_k / total
    return metrics


def accuracy_efficiency_ratio(
    acc_at_fraction: float,
    acc_at_full: float,
) -> float:
    """
    AER = acc(N%) / acc(100%) — Phase 3 data-size efficiency metric.
    A ratio close to 1.0 means the method is robust to data reduction.
    """
    if acc_at_full < 1e-6:
        return 0.0
    return acc_at_fraction / acc_at_full


def compute_aer_table(
    results: Dict[str, Dict[float, float]],
    fractions: List[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
) -> Dict[str, Dict[float, float]]:
    """
    Compute AER for each variant across all fractions.

    Args:
        results: {variant: {fraction: top1_acc}}
    Returns:
        {variant: {fraction: AER}}
    """
    aer_table = {}
    for variant, frac_acc in results.items():
        full_acc = frac_acc.get(1.00, frac_acc.get(max(frac_acc.keys()), 1.0))
        aer_table[variant] = {
            frac: accuracy_efficiency_ratio(acc, full_acc)
            for frac, acc in frac_acc.items()
        }
    return aer_table


def report_metrics(
    metrics: Dict[str, float],
    model: nn.Module,
    dataset_name: str,
    variant: str,
    seed: int,
) -> Dict[str, float]:
    """
    Build a standardized metrics row for logging.
    Includes model size (params + estimated GMACs).
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "dataset": dataset_name,
        "variant": variant,
        "seed": seed,
        "n_params_M": n_params / 1e6,
        **metrics,
    }
