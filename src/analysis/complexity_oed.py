# src/analysis/complexity_oed.py
"""
Orthogonal Experiment Design (OED) for ViT complexity ablation.
Mirrors the methodology from Yuan & Zuo (2022) — Papers 4/5.

OED factors and levels for ViT:
    F1: depth         ∈ {4, 6, 8}         (number of transformer blocks)
    F2: num_heads     ∈ {3, 6, 12}        (attention heads)
    F3: patch_size    ∈ {2, 4, 8}         (patch size in pixels)
    F4: mlp_ratio     ∈ {2, 4, 8}         (MLP hidden dim / embed_dim)

L9(3^4) standard orthogonal array:
    Row → experiment, columns → factor levels (0-indexed).
    Each pair of factors appears at each combination of levels exactly once.

Goal: find optimal ViT complexity for small datasets and determine whether
      LIFE + SS-init shifts the optimal configuration or flattens sensitivity.
"""

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# L9(3^4) orthogonal array
# ─────────────────────────────────────────────────────────────────────────────

# Standard L9 table — rows are experiments, columns are factors
# Values are 0, 1, 2 (level indices)
L9 = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 2, 2, 2],
    [1, 0, 1, 2],
    [1, 1, 2, 0],
    [1, 2, 0, 1],
    [2, 0, 2, 1],
    [2, 1, 0, 2],
    [2, 2, 1, 0],
])

# Factor levels
FACTOR_LEVELS = {
    "depth":      [4, 6, 8],
    "num_heads":  [3, 6, 12],
    "patch_size": [2, 4, 8],
    "mlp_ratio":  [2, 4, 8],
}

FACTOR_NAMES = list(FACTOR_LEVELS.keys())


class OEDExperiment(NamedTuple):
    experiment_id: int
    depth: int
    num_heads: int
    patch_size: int
    mlp_ratio: int


def get_oed_experiments() -> List[OEDExperiment]:
    """Return 9 OED experiment configurations."""
    experiments = []
    for i, row in enumerate(L9):
        config = {name: FACTOR_LEVELS[name][row[j]] for j, name in enumerate(FACTOR_NAMES)}
        experiments.append(OEDExperiment(experiment_id=i + 1, **config))
    return experiments


def print_oed_table() -> None:
    """Print the OED design table."""
    exps = get_oed_experiments()
    df = pd.DataFrame([e._asdict() for e in exps])
    print("L9(3^4) Orthogonal Array — ViT Complexity Ablation")
    print("=" * 60)
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Range analysis (ANOVA-style) for OED results
# ─────────────────────────────────────────────────────────────────────────────

def range_analysis(
    results: Dict[str, List[float]],
) -> pd.DataFrame:
    """
    Perform range analysis (极差分析) on OED results.

    Args:
        results: {variant_name: [val_acc for each of 9 OED experiments]}

    Returns:
        DataFrame with mean per level and range (R) per factor per variant.
        R = max_level_mean - min_level_mean; larger R → more influential factor.
    """
    analysis = {}
    for variant, accs in results.items():
        accs = np.array(accs)
        row = {}
        for j, factor in enumerate(FACTOR_NAMES):
            levels = L9[:, j]
            means = [accs[levels == k].mean() for k in range(3)]
            for k, m in enumerate(means):
                row[f"{factor}_L{k+1}"] = m
            row[f"{factor}_R"] = max(means) - min(means)
        analysis[variant] = row

    return pd.DataFrame(analysis).T


def plot_factor_effects(
    results: Dict[str, List[float]],
    save_path: str = "outputs/oed_factor_effects.png",
) -> plt.Figure:
    """
    Plot mean accuracy per factor level for each variant.
    Steeper slopes → more sensitivity to that factor.
    """
    fig, axes = plt.subplots(1, len(FACTOR_NAMES), figsize=(4 * len(FACTOR_NAMES), 5), sharey=True)
    colors = {"A_scratch": "#e74c3c", "B_ss_init": "#3498db", "C_life": "#2ecc71", "D_combined": "#9b59b6"}

    for ax, (j, factor) in zip(axes, enumerate(FACTOR_NAMES)):
        factor_values = FACTOR_LEVELS[factor]
        for variant, accs in results.items():
            accs = np.array(accs)
            levels = L9[:, j]
            means = [accs[levels == k].mean() for k in range(3)]
            ax.plot(factor_values, means, marker="o", label=variant,
                    color=colors.get(variant, "gray"), linewidth=2)
        ax.set_title(factor, fontsize=12)
        ax.set_xlabel("Level value")
        if j == 0:
            ax.set_ylabel("Top-1 Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("OED Factor Effect Analysis — CIFAR-100", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
