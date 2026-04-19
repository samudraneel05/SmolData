# src/analysis/cka.py
"""
Centered Kernel Alignment (CKA) — Kornblith et al. (NeurIPS 2019).

Used in Phase 4 to compare internal representations across:
  - Variant A (scratch) vs Variant D (SS-init + LIFE)
  - Both vs Raghu et al.'s large-dataset reference pattern (Paper 1, Fig 6)

The pairwise CKA matrix (N_layers_A × N_layers_B) reveals which layers
in model A correspond to which layers in model B.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Core CKA computation
# ─────────────────────────────────────────────────────────────────────────────

def center_gram(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representations X (n × p) and Y (n × q).
    CKA(X, Y) = ||Y^T X||^2_F / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    dot_product = np.linalg.norm(X.T @ Y) ** 2
    norm_x = np.linalg.norm(X.T @ X)
    norm_y = np.linalg.norm(Y.T @ Y)

    if norm_x < 1e-10 or norm_y < 1e-10:
        return 0.0
    return float(dot_product / (norm_x * norm_y))


def kernel_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Kernel (RBF) CKA for non-linear comparison.
    Uses HSIC with empirical kernel matrices.
    """
    K_X = center_gram(X @ X.T)
    K_Y = center_gram(Y @ Y.T)

    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sqrt(np.sum(K_X * K_X))
    hsic_yy = np.sqrt(np.sum(K_Y * K_Y))

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / (hsic_xx * hsic_yy))


# ─────────────────────────────────────────────────────────────────────────────
# Representation extraction via hooks
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationExtractor:
    """
    Attaches forward hooks to specified layers and collects activations.

    Usage:
        extractor = RepresentationExtractor(model, ["blocks.0", "blocks.3", "norm"])
        with extractor:
            _ = model(X)
        reps = extractor.representations  # dict: layer_name → (n_samples, features)
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.representations: Dict[str, np.ndarray] = {}
        self._hooks = []
        self._buffers: Dict[str, List[np.ndarray]] = {n: [] for n in layer_names}

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Mean-pool if 3D (B, N, C) → (B, C)
                if output.ndim == 3:
                    out = output.mean(dim=1)
                else:
                    out = output
                self._buffers[name].append(out.detach().cpu().float().numpy())
        return hook

    def __enter__(self):
        for name in self.layer_names:
            try:
                module = dict(self.model.named_modules())[name]
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
            except KeyError:
                print(f"Warning: layer '{name}' not found in model")
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Concatenate buffers
        for name in self.layer_names:
            if self._buffers[name]:
                self.representations[name] = np.concatenate(self._buffers[name], axis=0)
                self._buffers[name].clear()


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise CKA matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_cka(
    reps_a: Dict[str, np.ndarray],
    reps_b: Dict[str, np.ndarray],
    method: str = "linear",
) -> np.ndarray:
    """
    Compute N_A × N_B pairwise CKA matrix between layer representations.

    Args:
        reps_a: {layer_name: (n_samples, features)} for model A
        reps_b: {layer_name: (n_samples, features)} for model B
        method: "linear" or "kernel"

    Returns:
        CKA matrix of shape (len(reps_a), len(reps_b))
    """
    cka_fn = linear_cka if method == "linear" else kernel_cka
    layers_a = list(reps_a.keys())
    layers_b = list(reps_b.keys())
    matrix = np.zeros((len(layers_a), len(layers_b)))

    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            matrix[i, j] = cka_fn(reps_a[la], reps_b[lb])

    return matrix


def extract_representations(
    model: nn.Module,
    layer_names: List[str],
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> Dict[str, np.ndarray]:
    """Extract layer representations from a model over a data loader."""
    model.eval()
    extractor = RepresentationExtractor(model, layer_names)

    collected = 0
    with torch.no_grad(), extractor:
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            collected += images.size(0)
            if collected >= max_samples:
                break

    return {k: v[:max_samples] for k, v in extractor.representations.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_cka_matrix(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "CKA Similarity",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(6, len(row_labels))))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        vmin=0, vmax=1,
        cmap="magma",
        annot=len(row_labels) <= 12,
        fmt=".2f",
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Model B layers", fontsize=11)
    ax.set_ylabel("Model A layers", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
