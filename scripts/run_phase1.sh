#!/bin/bash
# scripts/run_phase1.sh
# Run all Phase 1 baseline replications

set -e
source .venv/bin/activate

DATASETS="cifar10 cifar100 svhn"
MODELS="vit_scratch resnet18 resnet56 sl_vit"

echo "=== Phase 1: Baseline Replication ==="
mkdir -p results checkpoints/phase1

for dataset in $DATASETS; do
  for model in $MODELS; do
    echo "Running: $model on $dataset"
    python experiments/phase1_baselines.py \
      --model "$model" \
      --dataset "$dataset" \
      --seed 42 \
      --config configs/base.yaml
  done
done

echo "Phase 1 complete. Results → results/phase1_baselines.csv"
