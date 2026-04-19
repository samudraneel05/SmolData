# src/utils/config.py
"""YAML / Hydra config loading."""

from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    """Load a YAML config file via OmegaConf."""
    return OmegaConf.load(Path(path))


def merge_configs(*cfgs: DictConfig) -> DictConfig:
    """Merge multiple OmegaConf configs (later ones override earlier)."""
    base = OmegaConf.create({})
    for cfg in cfgs:
        base = OmegaConf.merge(base, cfg)
    return base


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)
