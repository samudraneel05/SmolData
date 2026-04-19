# experiments/phase2_variants.py
"""
Phase 2 Orchestrator — Four variants × Three architectures × Five datasets.

Variants:
    A — Scratch:        Standard ViT, random init, supervised
    B — SS-init only:   Standard ViT, DINO pre-train → fine-tune
    C — LIFE only:      LIFE ViT, random init, supervised
    D — SS-init + LIFE: LIFE ViT, DINO pre-train → fine-tune

Architectures: vit_tiny, swin_tiny (via timm), cait_s24 (via timm)
Datasets: cifar10, cifar100, svhn, cinic10, tiny_imagenet
Seeds: [42, 1337, 2024]

Usage:
    python experiments/phase2_variants.py --dataset cifar100 --arch vit_tiny --seed 42
    python experiments/phase2_variants.py --smoke-test  # 1 epoch, debug mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from utils.seed import set_seed
from utils.config import load_config
from utils.logging import get_logger, init_wandb, CSVLogger
from models.vit_scratch import vit_tiny, vit_small
from models.life_deit import life_deit_tiny, life_deit_small
from data.datasets import get_dataset
from training.supervised import train
from training.ssl_pretrain import pretrain_dino, MultiCropDataset, dino_multicrop_transforms
from training.finetune import finetune

logger = get_logger("phase2")

VARIANTS = ["A", "B", "C", "D"]
# Default archs/datasets — overridden by config (lite.yaml uses just vit_tiny + cifar10/100)
DEFAULT_ARCHS = ["vit_tiny", "swin_tiny", "cait_s24"]
DEFAULT_DATASETS = ["cifar10", "cifar100", "svhn", "cinic10", "tiny_imagenet"]
DEFAULT_SEEDS = [42, 1337, 2024]


def build_model(arch: str, variant: str, num_classes: int, img_size: int):
    """Return the right model for (arch, variant) pair."""
    use_life = variant in ("C", "D")

    if arch == "vit_tiny":
        if use_life:
            return life_deit_tiny(num_classes=num_classes, img_size=img_size)
        return vit_tiny(num_classes=num_classes, img_size=img_size)

    elif arch == "swin_tiny":
        import timm
        if use_life:
            # LIFE-Swin not yet implemented: use timm baseline with a warning
            logger.warning("LIFE-Swin not yet patched — using standard Swin-T")
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False,
                                 num_classes=num_classes, img_size=img_size)

    elif arch == "cait_s24":
        import timm
        return timm.create_model("cait_s24_224", pretrained=False,
                                 num_classes=num_classes, img_size=img_size)

    raise ValueError(f"Unknown arch: {arch}")


def run_variant(
    variant: str,
    arch: str,
    dataset: str,
    seed: int,
    cfg: dict,
    smoke_test: bool = False,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fraction = cfg.get("dataset_fraction", 1.0)
    use_wandb = cfg.get("use_wandb", False)
    run_name = f"v{variant}_{arch}_{dataset}_s{seed}"

    logger.info(f"Running Variant {variant} | {arch} | {dataset} "
                f"(fraction={fraction:.0%}) | seed={seed} | device={device}")

    # ── Init W&B run ──────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(
            entity=cfg.get("wandb_entity", None),
            project=cfg.get("wandb_project", "smoldata"),
            name=run_name,
            config={
                "variant": variant, "arch": arch, "dataset": dataset,
                "seed": seed, "fraction": fraction,
                **{k: v for k, v in cfg.items()
                   if k not in ("wandb_project", "wandb_entity", "use_wandb")},
            },
            tags=[f"variant-{variant}", arch, dataset, f"seed-{seed}"],
            reinit=True,
        )

    train_loader, val_loader, num_classes, img_size = get_dataset(
        dataset,
        batch_size=cfg["batch_size"],
        fraction=fraction,
        seed=seed,
    )
    cfg["num_classes"] = num_classes

    if smoke_test:
        cfg["epochs"] = 1
        cfg["ssl_epochs"] = 1

    model = build_model(arch, variant, num_classes, img_size)
    save_dir = f"checkpoints/phase2/{run_name}"

    if variant in ("B", "D"):
        # Self-supervised pre-training first
        transforms, n_crops = dino_multicrop_transforms(img_size=img_size, dataset_name=dataset)
        import torchvision
        raw_ds = getattr(torchvision.datasets, {
            "cifar10": "CIFAR10", "cifar100": "CIFAR100"
        }.get(dataset, "CIFAR10"))("./data", train=True, download=True)
        mc_ds = MultiCropDataset(raw_ds, transforms)
        from torch.utils.data import DataLoader as DL
        ssl_loader = DL(mc_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], drop_last=True)

        cfg["embed_dim"] = 192
        model = pretrain_dino(model, ssl_loader, cfg, save_dir=save_dir,
                              run_name=run_name, use_wandb=use_wandb)
        metrics = finetune(model, train_loader, val_loader, cfg, save_dir=save_dir,
                           run_name=run_name, use_wandb=use_wandb)
    else:
        metrics = train(model, train_loader, val_loader, cfg, save_dir=save_dir,
                        run_name=run_name, use_wandb=use_wandb)

    metrics.update({"variant": variant, "arch": arch, "dataset": dataset, "seed": seed})
    logger.info(f"  → best val_acc={metrics.get('val_acc', 0):.4f}")

    if use_wandb:
        import wandb
        wandb.log({"best_val_acc": metrics.get("val_acc", 0)})
        wandb.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Use configs/lite.yaml for Kaggle/Colab")
    args = parser.parse_args()

    cfg = dict(load_config(args.config))

    # Resolve experiment dimensions from config (lite.yaml shrinks these)
    archs    = [args.arch]    if args.arch    else cfg.get("archs",    DEFAULT_ARCHS)
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", DEFAULT_DATASETS)
    seeds    = [args.seed]    if args.seed    else cfg.get("seeds",    DEFAULT_SEEDS)
    variants = [args.variant] if args.variant else VARIANTS

    csv_log = CSVLogger("results/phase2_results.csv")

    total = len(variants) * len(archs) * len(datasets) * len(seeds)
    fraction = cfg.get("dataset_fraction", 1.0)
    logger.info(
        f"Phase 2 [{args.config}]: {total} total runs — "
        f"{len(variants)} variants × {len(archs)} archs × "
        f"{len(datasets)} datasets × {len(seeds)} seed(s) | "
        f"data fraction={fraction:.0%}"
    )

    for dataset in datasets:
        for arch in archs:
            for variant in variants:
                for seed in seeds:
                    try:
                        metrics = run_variant(
                            variant, arch, dataset, seed, cfg.copy(),
                            smoke_test=args.smoke_test,
                        )
                        csv_log.log(metrics)
                    except Exception as e:
                        logger.error(f"FAILED: {variant} {arch} {dataset} s{seed}: {e}")
                        raise


if __name__ == "__main__":
    main()
