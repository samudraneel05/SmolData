#!/usr/bin/env python3
"""
SmolData — Kaggle/Colab Entry Point
====================================
Single script to run the full experiment pipeline on a free GPU instance.

Timeline on Kaggle T4 (15h session):
    Setup & data download  : ~10 min
    Phase 1 baselines      : ~90 min  (4 models × 2 datasets × 50 epochs)
    Phase 2 variants       : ~180 min (4 variants × 50 epochs × 2 datasets)
    Phase 3 OED (A+D only) : ~120 min (9 configs × 2 variants)
    Phase 4 CKA analysis   : ~20 min  (no training)
    Total                  : ~7 hours (fits in one Kaggle session)

Usage:
    # On Kaggle — paste this as the notebook cell, or:
    !python kaggle_run.py

    # To run only one phase:
    !python kaggle_run.py --phases 1 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Environment detection ──────────────────────────────────────────────────────
def is_kaggle():
    return os.path.exists("/kaggle")

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

ROOT = Path(__file__).parent

def run(cmd: str, **kwargs):
    print(f"\n{'='*60}\n$ {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, check=True, **kwargs)
    return result


# ── Step 0: Install dependencies ───────────────────────────────────────────────
def setup():
    print("=== SETUP ===")
    if is_kaggle() or is_colab():
        # On Kaggle/Colab, install into the current Python env
        run("pip install timm einops hydra-core omegaconf wandb rich -q")
    else:
        print("Local environment detected — assuming .venv is already active")

    # Install this project in editable mode
    run(f"pip install -e {ROOT} -q")

    # Download datasets (CIFAR-10, CIFAR-100 via torchvision)
    run(f"bash {ROOT}/scripts/download_datasets.sh")
    print("Setup complete.\n")


# ── Step 1: Phase 1 baselines ──────────────────────────────────────────────────
def phase1():
    print("=== PHASE 1: Baseline Replication (lite mode) ===")
    for model in ["vit_scratch", "resnet18", "sl_vit"]:
        for dataset in ["cifar10", "cifar100"]:
            run(
                f"python {ROOT}/experiments/phase1_baselines.py "
                f"--model {model} --dataset {dataset} --seed 42 "
                f"--config {ROOT}/configs/lite.yaml"
            )


# ── Step 2: Phase 2 four variants ─────────────────────────────────────────────
def phase2():
    print("=== PHASE 2: Four Variants (lite mode) ===")
    print("Runs: 4 variants × 1 arch (vit_tiny) × 2 datasets × 1 seed = 8 runs")
    run(
        f"python {ROOT}/experiments/phase2_variants.py "
        f"--config {ROOT}/configs/lite.yaml"
    )


# ── Step 3: Phase 3 OED (A+D only for speed) ──────────────────────────────────
def phase3():
    print("=== PHASE 3: OED Complexity Ablation (Variants A + D) ===")
    from src.analysis.complexity_oed import get_oed_experiments, print_oed_table
    sys.path.insert(0, str(ROOT / "src"))
    print_oed_table()

    for exp in get_oed_experiments():
        for variant in ["A", "D"]:  # just control vs proposed (saves 50% compute)
            run(
                f"python {ROOT}/experiments/phase2_variants.py "
                f"--variant {variant} --arch vit_tiny --dataset cifar100 --seed 42 "
                f"--config {ROOT}/configs/lite.yaml"
            )


# ── Step 4: Phase 4 analysis (requires Phase 2 checkpoints) ───────────────────
def phase4():
    print("=== PHASE 4: CKA + Attention Visualization ===")
    # Find best checkpoints
    ckpt_a = ROOT / "checkpoints/phase2/vA_vit_tiny_cifar100_s42_best.pt"
    ckpt_d = ROOT / "checkpoints/phase2/vD_vit_tiny_cifar100_s42_best.pt"

    if not ckpt_a.exists() or not ckpt_d.exists():
        print(f"Checkpoints not found — skipping Phase 4. Run Phase 2 first.")
        return

    run(
        f"python {ROOT}/experiments/phase4_analysis.py "
        f"--model_a {ckpt_a} --model_d {ckpt_d} "
        f"--dataset cifar100 --n_cka_samples 500"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Which phases to run (0=setup 1 2 3 4)")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    os.makedirs(ROOT / "results", exist_ok=True)

    phase_fns = {0: setup, 1: phase1, 2: phase2, 3: phase3, 4: phase4}

    for p in args.phases:
        if p == 0 and args.skip_setup:
            continue
        phase_fns[p]()

    print("\n✓ All requested phases complete.")
    print("Results → results/   |   Checkpoints → checkpoints/   |   Plots → outputs/")


if __name__ == "__main__":
    main()
