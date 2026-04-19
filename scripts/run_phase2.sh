#!/bin/bash
# scripts/run_phase2.sh
# Run Phase 2: all 4 variants × 3 archs × 5 datasets × 3 seeds

set -e
source .venv/bin/activate

GPUS=${GPUS:-0}
mkdir -p results checkpoints/phase2

echo "=== Phase 2: Variant Comparison ==="

# Run per-dataset to allow GPU-level parallelism across datasets
for dataset in cifar10 cifar100 svhn; do
  python experiments/phase2_variants.py \
    --dataset "$dataset" \
    --config configs/base.yaml
done

echo "Phase 2 complete. Results → results/phase2_results.csv"
