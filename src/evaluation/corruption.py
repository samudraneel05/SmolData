# src/evaluation/corruption.py
"""
Corruption robustness evaluation — Phase 5.

Evaluates on CIFAR-10-C / CIFAR-100-C using the 18 Hendrycks & Dietterich corruptions
at severity levels 1–5. Computes Mean Corruption Error (MCE) normalized against a
canonical clean accuracy baseline.

Reference: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness
           to Common Corruptions and Perturbations", ICLR 2019.
"""

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# All 18 corruptions × 5 severity levels
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

# Corruptions available in the standard CIFAR-10/100-C release (15 out of 18)
STANDARD_CORRUPTIONS = CORRUPTIONS[:15]


def load_cifar_c(
    dataset: str = "cifar10",
    corruption: str = "gaussian_noise",
    severity: int = 1,
    data_root: str = "./data",
    normalize: bool = True,
) -> TensorDataset:
    """
    Load a single corruption × severity split from CIFAR-10-C or CIFAR-100-C.

    Dataset layout (after download):
        data_root/CIFAR-10-C/{corruption}.npy  (50000 × 32 × 32 × 3, uint8)
        data_root/CIFAR-10-C/labels.npy        (50000,)

    Each severity block = 10000 samples. severity ∈ {1, 2, 3, 4, 5}.
    """
    root = Path(data_root) / f"{'CIFAR-10-C' if dataset == 'cifar10' else 'CIFAR-100-C'}"
    if not root.exists():
        raise FileNotFoundError(
            f"CIFAR-C not found at {root}. "
            "Download from https://zenodo.org/record/2535967"
        )

    images = np.load(root / f"{corruption}.npy")
    labels = np.load(root / "labels.npy").astype(np.int64)

    # Select severity block
    start = (severity - 1) * 10000
    end = severity * 10000
    images = images[start:end]  # (10000, 32, 32, 3)
    labels = labels[start:end]

    # Convert to tensors
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

    if normalize:
        from ..training.augmentations import DATASET_STATS
        stats = DATASET_STATS.get(dataset, DATASET_STATS["cifar10"])
        mean = torch.tensor(stats["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(stats["std"]).view(1, 3, 1, 1)
        x = (x - mean) / std

    y = torch.from_numpy(labels)
    return TensorDataset(x, y)


@torch.no_grad()
def compute_corruption_error(
    model: nn.Module,
    dataset_name: str,
    data_root: str = "./data",
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    corruptions: Optional[List[str]] = None,
    severities: List[int] = (1, 2, 3, 4, 5),
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate corruption error for all corruptions × severities.

    Returns:
        {corruption_name: {severity: error_rate}}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    corruptions = corruptions or STANDARD_CORRUPTIONS

    results = {}
    for corruption in tqdm(corruptions, desc="Corruptions"):
        results[corruption] = {}
        for sev in severities:
            try:
                ds = load_cifar_c(dataset_name, corruption, sev, data_root)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                correct, total = 0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
                results[corruption][sev] = 1.0 - correct / total
            except FileNotFoundError:
                results[corruption][sev] = float("nan")

    return results


def mean_corruption_error(
    corruption_results: Dict[str, Dict[int, float]],
    clean_error: float,
    baseline_errors: Optional[Dict[str, float]] = None,
) -> float:
    """
    MCE = mean over corruptions of (mean over severities of corruption_error)
    Normalized by clean_error if no baseline provided.

    If baseline_errors is provided (e.g., AlexNet reference errors per corruption),
    computes normalized MCE as in the original paper.
    """
    mce_sum = 0.0
    count = 0
    for corruption, severity_errors in corruption_results.items():
        valid = [e for e in severity_errors.values() if not np.isnan(e)]
        if valid:
            mean_err = np.mean(valid)
            if baseline_errors and corruption in baseline_errors:
                mce_sum += mean_err / baseline_errors[corruption]
            else:
                mce_sum += mean_err
            count += 1

    return mce_sum / count if count > 0 else float("nan")
