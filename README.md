# SmolData — Locality-Aware ViT for Small Datasets

Research project comparing four ways to train Vision Transformers on small datasets, combining architectural locality (LIFE module) with self-supervised initialization (DINO).

## Quick Start (Kaggle/Colab)
```bash
python kaggle_run.py       # runs everything end-to-end (~7h on T4)
```

## Structure
```
src/models/      ← ViT, ResNet, SL-ViT, LIFE-DeiT
src/training/    ← supervised, DINO SSL pre-train, fine-tune, augmentations
src/analysis/    ← CKA, attention rollout, OED complexity
src/data/        ← dataset loaders with fraction/lite mode
src/evaluation/  ← accuracy, AER, corruption MCE
configs/         ← base.yaml (full) · lite.yaml (Kaggle/Colab)
experiments/     ← phase orchestrators (1–4)
kaggle_run.py    ← single entry point for cloud runs
```

## The Four Variants
| Variant | Architecture | Init | Purpose |
|---------|-------------|------|---------|
| **A** | Standard ViT | Random | Baseline control |
| **B** | Standard ViT | DINO SSL | Training strategy fix |
| **C** | LIFE ViT | Random | Architectural fix |
| **D** | LIFE ViT | DINO SSL | **Both combined (proposed)** |

## See `Instructions.md` for the full step-by-step run guide.
