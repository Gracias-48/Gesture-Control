"""Model zoo + factory shared by training, evaluation and ONNX export."""
from __future__ import annotations

import torch.nn as nn

from src.data.preprocessing import feature_dim, input_kind
from src.models.bilstm import BiLSTMClassifier
from src.models.stgcn import STGCN
from src.models.tcn import CausalTCN

MODEL_NAMES = ("bilstm", "stgcn", "tcn")
DISPLAY_NAMES = {"bilstm": "BiLSTM", "stgcn": "ST-GCN", "tcn": "TCN"}


def build_model(name: str, num_classes: int, dropout: float, use_global: bool = True,
                model_cfg: dict | None = None) -> nn.Module:
    """Instantiate a model by name with the architecture section of ``config.yaml``."""
    name = name.lower()
    mc = dict(model_cfg or {})
    in_dim = feature_dim(input_kind(name), use_global)
    if name == "bilstm":
        return BiLSTMClassifier(in_dim, num_classes, dropout, **mc)
    if name == "stgcn":
        mc["channels"] = tuple(mc.get("channels", (64, 64, 128, 128)))
        mc["strides"] = tuple(mc.get("strides", (1, 1, 2, 1)))
        return STGCN(in_dim, num_classes, dropout, **mc)
    if name == "tcn":
        mc["dilations"] = tuple(mc.get("dilations", (1, 2, 4, 8)))
        return CausalTCN(in_dim, num_classes, dropout, **mc)
    raise ValueError(f"Unknown model '{name}' (choose from {MODEL_NAMES})")


def dummy_input_shape(name: str, T: int, use_global: bool = True, batch: int = 1) -> tuple[int, ...]:
    """Input tensor shape expected by ``name``."""
    kind = input_kind(name)
    d = feature_dim(kind, use_global)
    return (batch, T, d) if kind == "sequence" else (batch, d, T, 21)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
