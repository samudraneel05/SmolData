# SmolData — Locality-Aware Initialization for ViTs on Small Datasets

## Quick Start

```bash
# 1. Create environment (already done)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Download datasets
bash scripts/download_datasets.sh

# 4. Run Phase 1 (baseline replication)
bash scripts/run_phase1.sh

# 5. Run Phase 2 (4 variants × 3 archs × 5 datasets)
python experiments/phase2_variants.py --dataset cifar10 --arch vit_tiny --smoke-test
bash scripts/run_phase2.sh
```

## Project Structure

```
src/
├── models/          # ViT-scratch, ResNet, SL-ViT, LIFE module, LIFE-DeiT
├── training/        # Supervised, SSL (DINO), fine-tuning, augmentations
├── analysis/        # CKA, attention rollout, OED complexity analysis
├── data/            # Dataset loaders + stratified subsets
├── evaluation/      # Accuracy, AER, corruption (MCE)
└── utils/           # Seed, config (OmegaConf), logging (WandB + CSV)

experiments/         # Phase orchestrators (1–5)
configs/             # YAML hyperparameter configs
scripts/             # Shell scripts for each phase
results/             # CSV result logs (auto-generated)
checkpoints/         # Model weights (auto-generated)
```

## Experimental Phases

| Phase | Description | Runs |
|-------|-------------|------|
| 1 | Baseline replication (±0.5% gate) | ~20 |
| 2 | 4 variants × 3 archs × 5 datasets × 3 seeds | 180 |
| 3 | OED complexity + data-size ablations | 56 |
| 4 | CKA maps + attention visualization | analysis |
| 5 | Corruption robustness + fine-grained eval | ~40 |

## Variants

| Variant | Architecture | Initialization |
|---------|-------------|----------------|
| A — Scratch | Standard ViT | Random |
| B — SS-init | Standard ViT | DINO pre-training |
| C — LIFE only | LIFE Q/K/V ViT | Random |
| D — SS+LIFE (**proposed**) | LIFE Q/K/V ViT | DINO pre-training |

## Datasets

All trained at **native resolution** (no 224×224 upscaling):
- CIFAR-10/100 (32×32, auto-download)
- SVHN (32×32, auto-download)
- CINIC-10 (32×32, manual download)
- Tiny-ImageNet (64×64, manual download)

## Key Papers

1. **Zhu et al. (2023)** — CKA analysis of ViT on small datasets
2. **Gani et al.** — Self-supervised pre-training strategy
3. **Akkaya et al. (2024)** — LIFE module (depthwise-sep conv in Q/K/V)
4. **Yuan & Zuo (2022)** — OED study of CNN complexity on small data
