# src/training/finetune.py
"""
Supervised fine-tuning after self-supervised pre-training.
Used for Variants B and D.

Key differences from training from scratch:
- Lower learning rate (5× smaller by default)
- Shorter warmup
- Smaller weight decay (features already meaningful)
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .supervised import build_optimizer, build_scheduler, train_one_epoch, evaluate
from .augmentations import Mixup
from utils.logging import CSVLogger, get_logger
from torch.cuda.amp import GradScaler
from pathlib import Path
import time

logger = get_logger("finetune")


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    save_dir: str = "checkpoints",
    run_name: str = "finetune",
    use_wandb: bool = False,
) -> Dict[str, float]:
    """
    Fine-tune a pre-trained model with a lower learning rate.
    Returns best validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fine-tuning uses a lower LR than pre-training
    ft_cfg = {**cfg}
    ft_cfg["lr"] = cfg.get("finetune_lr", cfg["lr"] / 5)
    ft_cfg["warmup_epochs"] = cfg.get("finetune_warmup", 5)
    ft_cfg["weight_decay"] = cfg.get("finetune_wd", 0.01)

    optimizer = build_optimizer(model, ft_cfg["lr"], ft_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, ft_cfg["epochs"], ft_cfg["warmup_epochs"], len(train_loader))
    scaler = GradScaler()

    mixup = Mixup(
        mixup_alpha=ft_cfg.get("mixup_alpha", 0.4),
        cutmix_alpha=ft_cfg.get("cutmix_alpha", 1.0),
        num_classes=ft_cfg["num_classes"],
        prob=ft_cfg.get("mixup_prob", 1.0),
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_log = CSVLogger(save_dir / f"{run_name}_finetune_log.csv")

    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(1, ft_cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, mixup, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "time_s": elapsed, **train_metrics, **val_metrics,
               "lr": scheduler.get_last_lr()[0]}
        csv_log.log(row)

        if use_wandb:
            import wandb
            wandb.log({f"ft_{k}": v for k, v in row.items()})

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            torch.save(model.state_dict(), save_dir / f"{run_name}_ft_best.pt")

        if epoch % 10 == 0:
            logger.info(f"[FT {epoch:03d}/{ft_cfg['epochs']}] val_acc={val_metrics['val_acc']:.4f} "
                        f"best={best_val_acc:.4f} ({elapsed:.1f}s)")

    torch.save(model.state_dict(), save_dir / f"{run_name}_ft_last.pt")
    return best_metrics
