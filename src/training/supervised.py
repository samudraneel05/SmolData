# src/training/supervised.py
"""
Supervised training loop — used for Variant A (scratch) and Variant C (LIFE-only).

Features:
- AdamW optimizer with cosine decay + linear warmup
- Gradient clipping
- Mixed precision (AMP) via torch.cuda.amp
- WandB or CSV logging
- Checkpoint saving (best val acc + last epoch)
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentations import Mixup
from utils.logging import CSVLogger, get_logger

logger = get_logger("supervised")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    # Separate weight decay: don't apply to bias, LayerNorm weights
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "pos_embed" in name or "cls_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def build_scheduler(optimizer, epochs: int, warmup_epochs: int, steps_per_epoch: int):
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs * steps_per_epoch)
    cosine = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs * steps_per_epoch])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    mixup: Optional[Mixup],
    device: torch.device,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False, desc="train"):
        images, labels = images.to(device), labels.to(device)

        if mixup is not None:
            images, soft_labels = mixup(images, labels)
        else:
            soft_labels = torch.zeros(labels.size(0), criterion.ignore_index if hasattr(criterion, "ignore_index") else images.size(0), device=device)

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            if mixup is not None:
                loss = -(soft_labels * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        if mixup is None:
            correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"train_loss": total_loss / total, "train_acc": correct / total if mixup is None else float("nan")}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"val_loss": total_loss / total, "val_acc": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    save_dir: str = "checkpoints",
    use_wandb: bool = False,
    run_name: str = "run",
) -> Dict[str, float]:
    """Full supervised training loop. Returns best validation metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = build_optimizer(model, cfg["lr"], cfg.get("weight_decay", 0.05))
    scheduler = build_scheduler(
        optimizer,
        cfg["epochs"],
        cfg.get("warmup_epochs", 10),
        len(train_loader),
    )
    scaler = GradScaler()

    mixup = Mixup(
        mixup_alpha=cfg.get("mixup_alpha", 0.4),
        cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
        num_classes=cfg["num_classes"],
        prob=cfg.get("mixup_prob", 1.0),
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_log = CSVLogger(save_dir / f"{run_name}_log.csv")

    if use_wandb:
        import wandb
        wandb.watch(model, log="gradients", log_freq=100)

    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, mixup, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "time_s": elapsed, **train_metrics, **val_metrics, "lr": scheduler.get_last_lr()[0]}
        csv_log.log(row)

        if use_wandb:
            import wandb
            wandb.log(row)

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            torch.save(model.state_dict(), save_dir / f"{run_name}_best.pt")

        if epoch % 10 == 0:
            logger.info(
                f"[{epoch:03d}/{cfg['epochs']}] "
                f"val_acc={val_metrics['val_acc']:.4f} "
                f"best={best_val_acc:.4f} "
                f"({elapsed:.1f}s)"
            )

    torch.save(model.state_dict(), save_dir / f"{run_name}_last.pt")
    return best_metrics
