"""Shared helpers: config loading, dotted overrides, seeding, device selection."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML configuration (defaults to ``<repo>/config.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # YAML keys of the action mapping may be parsed as int already; normalise.
    mapping = cfg.get("actions", {}).get("mapping", {})
    cfg.setdefault("actions", {})["mapping"] = {int(k): v for k, v in mapping.items()}
    return cfg


def resolve_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Return an absolute path for ``cfg['paths'][key]``."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT_DIR / p


def apply_overrides(cfg: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``cfg`` with dotted-key overrides applied.

    Example: ``{"features.T": 16, "augmentation.enabled": False}``.
    """
    out = copy.deepcopy(cfg)
    for dotted, value in (overrides or {}).items():
        node = out
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return out


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch optional for inference-only installs
        pass


def get_device(preference: str = "auto"):
    """Pick a torch device according to ``preference``."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_names(cfg: Mapping[str, Any]) -> list[str]:
    """Active class names (drops Background when the ablation excludes it)."""
    names = list(cfg["classes"]["names"])
    if cfg["classes"].get("exclude_background", False):
        names = [n for i, n in enumerate(names) if i != cfg["classes"]["background_id"]]
    return names
