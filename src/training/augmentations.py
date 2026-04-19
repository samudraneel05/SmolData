# src/training/augmentations.py
"""
Unified augmentation pipeline — identical across ALL model variants to ensure fair comparison.

Augmentations applied (per Gani et al. criticism of inconsistent aug pipelines):
    1. AutoAugment  (CIFAR policy)
    2. Random Horizontal Flip
    3. Random Crop with padding
    4. Normalization (dataset-specific mean/std)
    --- at batch level ---
    5. CutMix  (α = 1.0)
    6. MixUp   (α = 0.4)
    7. Random Erasing (p = 0.25, scale = (0.02, 0.33))
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

DATASET_STATS = {
    "cifar10":     {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "cifar100":    {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "cinic10":     {"mean": (0.47889522, 0.47227842, 0.43047404), "std": (0.24205776, 0.23828046, 0.25874835)},
    "svhn":        {"mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
    "tiny_imagenet": {"mean": (0.4802, 0.4481, 0.3975), "std": (0.2302, 0.2265, 0.2262)},
}


def build_transforms(
    dataset_name: str,
    img_size: int = 32,
    is_train: bool = True,
) -> T.Compose:
    """Build deterministic transform pipeline for train or eval."""
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS["cifar10"])
    normalize = T.Normalize(stats["mean"], stats["std"])

    if is_train:
        return T.Compose([
            T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
        ])
    else:
        return T.Compose([
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level augmentation: CutMix + MixUp
# ─────────────────────────────────────────────────────────────────────────────

class Mixup:
    """
    Applies MixUp and/or CutMix at the batch level.
    Returns (mixed_images, (original_labels, shuffled_labels, lambda_mix)).
    """

    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.prob:
            return x, self._one_hot(y)

        use_cutmix = np.random.rand() < self.switch_prob

        if use_cutmix:
            return self._cutmix(x, y)
        return self._mixup(x, y)

    def _one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(y.size(0), self.num_classes, device=y.device).scatter_(
            1, y.unsqueeze(1), 1.0
        )

    def _mixup(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[perm]
        y_a = self._one_hot(y)
        y_b = self._one_hot(y[perm])
        mixed_y = lam * y_a + (1 - lam) * y_b
        return mixed_x, mixed_y

    def _cutmix(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        _, _, H, W = x.shape

        cut_h = int(H * np.sqrt(1 - lam))
        cut_w = int(W * np.sqrt(1 - lam))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam_actual = 1 - (y2 - y1) * (x2 - x1) / (H * W)

        y_a = self._one_hot(y)
        y_b = self._one_hot(y[perm])
        mixed_y = lam_actual * y_a + (1 - lam_actual) * y_b
        return mixed_x, mixed_y
