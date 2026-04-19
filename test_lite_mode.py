"""Smoke test for lite mode dataset loading."""
import sys
sys.path.insert(0, "src")

# Patch imports for direct script execution
import src.training.augmentations  # noqa — ensure relative imports resolve

from src.data.datasets import get_dataset
from src.utils.config import load_config

# Test 1: CIFAR-10 at 20% fraction → 10k samples
train_loader, val_loader, nc, isz = get_dataset("cifar10", batch_size=64, fraction=0.2, seed=42)
n_train = len(train_loader.dataset)
n_val = len(val_loader.dataset)
print(f"CIFAR-10 lite: train={n_train} ({n_train//nc}/class), val={n_val}")
assert n_train == 10000, f"Expected 10000, got {n_train}"

# Test 2: CIFAR-100 at 20% fraction → 10k samples
train100, val100, nc100, _ = get_dataset("cifar100", batch_size=64, fraction=0.2, seed=42)
n100 = len(train100.dataset)
print(f"CIFAR-100 lite: train={n100} ({n100//nc100}/class), val={len(val100.dataset)}")
assert n100 == 10000, f"Expected 10000, got {n100}"

# Test 3: lite.yaml config
cfg = load_config("configs/lite.yaml")
print(f"lite.yaml: epochs={cfg.epochs}, batch={cfg.batch_size}, "
      f"fraction={cfg.dataset_fraction}, ssl_epochs={cfg.ssl_epochs}, "
      f"datasets={list(cfg.datasets)}")

print("\nLite mode smoke test PASSED!")
