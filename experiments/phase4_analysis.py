#!/usr/bin/env python3
# experiments/phase4_analysis.py
"""
Phase 4 — Mechanistic Analysis.

1. Pairwise CKA maps: Variant D vs Variant A (and vs large-dataset reference)
2. Attention rollout visualization: 10 test images per class
3. Comparison of lower-layer CNN-similarity between variants

Usage:
    python experiments/phase4_analysis.py \\
        --model_a checkpoints/phase2/vA_vit_tiny_cifar100_s42_best.pt \\
        --model_d checkpoints/phase2/vD_vit_tiny_cifar100_s42_best.pt \\
        --dataset cifar100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch

from utils.seed import set_seed
from utils.config import load_config
from utils.logging import get_logger
from models import vit_tiny, life_deit_tiny
from data.datasets import get_dataset
from analysis.cka import (
    extract_representations, compute_pairwise_cka, plot_cka_matrix,
)
from analysis.attention_viz import visualize_attention

logger = get_logger("phase4")


def get_vit_layer_names(depth: int = 12) -> list:
    """Return standard layer names for CKA extraction in a ViT."""
    names = [f"blocks.{i}" for i in range(depth)]
    names.append("norm")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True, help="Variant A checkpoint")
    parser.add_argument("--model_d", required=True, help="Variant D checkpoint")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cka_samples", type=int, default=1000)
    parser.add_argument("--n_attn_images", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/phase4")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, num_classes, img_size = get_dataset(args.dataset, batch_size=64)

    # Load Variant A
    model_a = vit_tiny(num_classes=num_classes, img_size=img_size)
    model_a.load_state_dict(torch.load(args.model_a, map_location="cpu"))
    model_a = model_a.to(device)

    # Load Variant D
    model_d = life_deit_tiny(num_classes=num_classes, img_size=img_size)
    model_d.load_state_dict(torch.load(args.model_d, map_location="cpu"))
    model_d = model_d.to(device)

    # ── CKA Analysis ──────────────────────────────────────────────────────────
    logger.info("Extracting representations for CKA...")
    layer_names_a = get_vit_layer_names(depth=12)
    layer_names_d = get_vit_layer_names(depth=12)

    reps_a = extract_representations(model_a, layer_names_a, val_loader, device, args.n_cka_samples)
    reps_d = extract_representations(model_d, layer_names_d, val_loader, device, args.n_cka_samples)

    # Cross-model CKA: Variant A vs Variant D
    logger.info("Computing pairwise CKA matrix (A vs D)...")
    cka_matrix = compute_pairwise_cka(reps_a, reps_d, method="linear")
    fig = plot_cka_matrix(
        cka_matrix,
        row_labels=layer_names_a,
        col_labels=layer_names_d,
        title=f"CKA: Variant A vs Variant D — {args.dataset}",
        save_path=str(out_dir / f"cka_A_vs_D_{args.dataset}.png"),
    )
    logger.info(f"Saved CKA plot → {out_dir / f'cka_A_vs_D_{args.dataset}.png'}")

    # Self-similarity: each model with itself (sanity check — should be diagonal)
    cka_self_a = compute_pairwise_cka(reps_a, reps_a, method="linear")
    plot_cka_matrix(cka_self_a, layer_names_a, layer_names_a,
                    title=f"CKA: Variant A self — {args.dataset}",
                    save_path=str(out_dir / f"cka_A_self_{args.dataset}.png"))

    # ── Attention Visualization ────────────────────────────────────────────────
    logger.info("Generating attention rollout visualizations...")
    images, labels = next(iter(val_loader))

    visualize_attention(
        model_a, images, labels.tolist(), device,
        img_size=img_size,
        save_dir=str(out_dir / f"attn_A_{args.dataset}"),
        n_images=args.n_attn_images,
    )
    visualize_attention(
        model_d, images, labels.tolist(), device,
        img_size=img_size,
        save_dir=str(out_dir / f"attn_D_{args.dataset}"),
        n_images=args.n_attn_images,
    )
    logger.info(f"Attention maps saved → {out_dir}")


if __name__ == "__main__":
    main()
