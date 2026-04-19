# src/utils/logging.py
"""WandB + CSV experiment logging."""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import wandb


def get_logger(name: str = "smoldata") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger(name)


def init_wandb(
    cfg: Dict[str, Any],
    project: str = "smoldata",
    run_name: Optional[str] = None,
) -> wandb.run:
    """Initialize a W&B run from a config dict."""
    return wandb.init(
        project=project,
        name=run_name,
        config=cfg,
        reinit=True,
    )


class CSVLogger:
    """Lightweight CSV logger as a WandB fallback."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._headers_written = False

    def log(self, row: Dict[str, Any]) -> None:
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
