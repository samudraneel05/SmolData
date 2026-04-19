# src/training/ssl_pretrain.py
"""
DINO-style self-supervised pre-training on small datasets.
Used for Variant B (SS-init only) and Variant D (SS-init + LIFE).

Reference: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021.
Adapted for extremely small datasets following Gani et al.'s two-stage approach.

Key components:
- Student-teacher architecture with EMA teacher update
- Multi-crop augmentation (2 global + N_local crops)
- Centering + sharpening in teacher outputs
- No labels required
"""

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import CSVLogger, get_logger

logger = get_logger("ssl_pretrain")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-crop augmentation
# ─────────────────────────────────────────────────────────────────────────────

import torchvision.transforms as T


def dino_multicrop_transforms(
    img_size: int = 32,
    global_crops_scale: Tuple[float, float] = (0.4, 1.0),
    local_crops_scale: Tuple[float, float] = (0.05, 0.4),
    n_local_crops: int = 6,
    dataset_name: str = "cifar10",
) -> Tuple[List, int]:
    """
    Returns a list of transforms for the multi-crop strategy.
    First 2 transforms produce global crops; remaining n_local_crops produce small crops.
    """
    from .augmentations import DATASET_STATS
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS["cifar10"])
    normalize = T.Normalize(stats["mean"], stats["std"])

    global_size = img_size
    local_size = max(img_size // 2, 16)  # Half resolution for local crops

    global_transform = T.Compose([
        T.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize,
    ])

    local_transform = T.Compose([
        T.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize,
    ])

    transforms = [global_transform, global_transform] + [local_transform] * n_local_crops
    return transforms, n_local_crops + 2


class MultiCropDataset(torch.utils.data.Dataset):
    """Wraps any dataset to apply multi-crop transforms."""

    def __init__(self, base_dataset, transforms: List) -> None:
        self.dataset = base_dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        # img is already a PIL image in most torchvision datasets
        if not hasattr(img, "convert"):
            # Convert tensor back to PIL for transforms
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
        return [t(img) for t in self.transforms]


# ─────────────────────────────────────────────────────────────────────────────
# DINO Head
# ─────────────────────────────────────────────────────────────────────────────

class DINOHead(nn.Module):
    """
    Projection head for DINO: MLP + L2 norm → prototypes.

    Lite defaults (hidden_dim=512, bottleneck_dim=128, n_layers=2) reduce
    GPU memory from ~1.2GB to ~0.3GB compared to the original ViT-S DINO setup.
    out_dim=8192 (from lite.yaml) vs 65536 in the original paper is appropriate
    for small datasets with few classes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 8192,       # was 65536; 8192 is fine for ≤200 classes
        hidden_dim: int = 512,     # was 2048
        bottleneck_dim: int = 128, # was 256
        n_layers: int = 2,         # was 3
    ) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# EMA teacher update
# ─────────────────────────────────────────────────────────────────────────────

class EMATeacher:
    def __init__(self, model: nn.Module, momentum: float = 0.996) -> None:
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, student: nn.Module, step: int, total_steps: int) -> None:
        # Cosine schedule for momentum: 0.996 → 1.0
        m = 1 - (1 - self.momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2
        for param_s, param_t in zip(student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)


# ─────────────────────────────────────────────────────────────────────────────
# DINO loss
# ─────────────────────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        n_crops: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        warmup_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.n_crops = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.full(200, teacher_temp),
        ])
        self.center_momentum = center_momentum

    def forward(
        self, student_out: List[torch.Tensor], teacher_out: List[torch.Tensor], epoch: int
    ) -> torch.Tensor:
        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        total_loss = 0.0
        n_loss_terms = 0

        teacher_softmax = [
            F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
            for t in teacher_out
        ]

        for i_s, s in enumerate(student_out):
            student_log = F.log_softmax(s / self.student_temp, dim=-1)
            for i_t, t in enumerate(teacher_softmax):
                if i_s == i_t:
                    continue
                total_loss += -(t * student_log).sum(dim=-1).mean()
                n_loss_terms += 1

        self._update_center(torch.cat(teacher_out))
        return total_loss / n_loss_terms

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_dino(
    student_backbone: nn.Module,
    train_loader: DataLoader,
    cfg: Dict[str, Any],
    save_dir: str = "checkpoints",
    run_name: str = "ssl_pretrain",
    use_wandb: bool = False,
) -> nn.Module:
    """
    DINO self-supervised pre-training.
    Returns the pre-trained backbone (without the DINO head).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_backbone = student_backbone.to(device)

    embed_dim = cfg.get("embed_dim", 192)
    out_dim = cfg.get("dino_out_dim", 65536)

    student_head = DINOHead(embed_dim, out_dim).to(device)
    ema_teacher = EMATeacher(student_backbone, momentum=cfg.get("ema_momentum", 0.996))
    teacher_head = DINOHead(embed_dim, out_dim).to(device)
    # Copy student head to teacher head initially
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_head.parameters():
        p.requires_grad = False

    n_crops = cfg.get("n_crops", 8)
    dino_loss = DINOLoss(out_dim=out_dim, n_crops=n_crops).to(device)

    all_params = list(student_backbone.parameters()) + list(student_head.parameters())
    optimizer = AdamW(all_params, lr=cfg.get("ssl_lr", 5e-4), weight_decay=cfg.get("weight_decay", 0.04))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("ssl_epochs", 200), eta_min=1e-6)
    scaler = GradScaler()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_log = CSVLogger(save_dir / f"{run_name}_ssl_log.csv")

    total_steps = cfg.get("ssl_epochs", 200) * len(train_loader)
    global_step = 0

    for epoch in range(1, cfg.get("ssl_epochs", 200) + 1):
        t0 = time.time()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, leave=False, desc=f"SSL e{epoch}"):
            # batch: list of n_crops tensors, each (B, C, H, W)
            crops = [c.to(device) for c in batch]

            with autocast():
                # Student processes all crops
                student_outs = [student_head(student_backbone.forward_features(c)) for c in crops]
                # Teacher processes only the 2 global crops
                with torch.no_grad():
                    ema_teacher.update(student_backbone, global_step, total_steps)
                    teacher_outs = [teacher_head(ema_teacher.teacher.forward_features(c)) for c in crops[:2]]

                loss = dino_loss(student_outs, teacher_outs, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            epoch_loss += loss.item()

        scheduler.step()
        row = {"epoch": epoch, "ssl_loss": epoch_loss / len(train_loader), "time_s": time.time() - t0}
        csv_log.log(row)
        if use_wandb:
            import wandb
            wandb.log(row)

        if epoch % 50 == 0:
            torch.save(student_backbone.state_dict(), save_dir / f"{run_name}_backbone_e{epoch}.pt")
            logger.info(f"[SSL {epoch:03d}] loss={row['ssl_loss']:.4f} ({row['time_s']:.1f}s)")

    torch.save(student_backbone.state_dict(), save_dir / f"{run_name}_backbone_final.pt")
    return student_backbone
