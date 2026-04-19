#!/usr/bin/env python3
# experiments/phase1_baselines.py
"""
Phase 1 — Baseline Replication.

Trains all baseline models with identical hyperparameters on all 5 datasets.
Success criterion: reproduce prior reported numbers ±0.5% top-1 accuracy.

Models replicated:
  - ViT-from-scratch (Dosovitskiy / Zhu et al. config)   → target: ~73% CIFAR-10
  - ResNet-18                                             → target: ~95% CIFAR-10
  - ResNet-56                                             → target: ~93% CIFAR-10
  - SL-ViT (SPT+LSA)                                     → target: ~87% CIFAR-10
  - ViT-Gani (2-stage, our Variant B)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
from utils.seed import set_seed
from utils.config import load_config
from utils.logging import get_logger, CSVLogger
from models import vit_base_paper1, resnet18, resnet56, SLViT
from data.datasets import get_dataset
from training.supervised import train

logger = get_logger("phase1")

BASELINE_TARGETS = {
    # (dataset, model) → expected top-1 acc (±0.5% tolerance)
    ("cifar10",  "vit_scratch"):  0.730,
    ("cifar10",  "resnet18"):     0.950,
    ("cifar10",  "resnet56"):     0.930,
    ("cifar10",  "sl_vit"):       0.870,
    ("cifar100", "vit_scratch"):  0.450,
    ("cifar100", "resnet18"):     0.780,
}


def build_baseline(name: str, num_classes: int, img_size: int):
    if name == "vit_scratch":
        return vit_base_paper1(num_classes=num_classes, img_size=img_size)
    elif name == "resnet18":
        return resnet18(num_classes=num_classes)
    elif name == "resnet56":
        return resnet56(num_classes=num_classes)
    elif name == "sl_vit":
        return SLViT(img_size=img_size, num_classes=num_classes)
    raise ValueError(f"Unknown baseline: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        choices=["vit_scratch", "resnet18", "resnet56", "sl_vit"])
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "svhn", "cinic10", "tiny_imagenet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = dict(load_config(args.config))
    set_seed(args.seed)

    models_to_run = [args.model] if args.model else ["vit_scratch", "resnet18", "resnet56", "sl_vit"]
    csv_log = CSVLogger("results/phase1_baselines.csv")

    for model_name in models_to_run:
        logger.info(f"=== {model_name} on {args.dataset} (seed={args.seed}) ===")
        train_loader, val_loader, num_classes, img_size = get_dataset(args.dataset, cfg["batch_size"])
        cfg["num_classes"] = num_classes

        model = build_baseline(model_name, num_classes, img_size)
        run_name = f"p1_{model_name}_{args.dataset}_s{args.seed}"

        metrics = train(model, train_loader, val_loader, cfg,
                        save_dir=f"checkpoints/phase1/{run_name}",
                        run_name=run_name)

        metrics.update({"model": model_name, "dataset": args.dataset, "seed": args.seed})
        csv_log.log(metrics)

        # Check if within tolerance of expected value
        key = (args.dataset, model_name)
        if key in BASELINE_TARGETS:
            target = BASELINE_TARGETS[key]
            achieved = metrics.get("val_acc", 0)
            ok = abs(achieved - target) <= 0.005
            status = "✓ PASS" if ok else "✗ FAIL (check training setup)"
            logger.info(f"  Baseline check: {achieved:.4f} vs {target:.4f} target → {status}")


if __name__ == "__main__":
    main()
