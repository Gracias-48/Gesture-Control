"""Model 3 — Causal Dilated Temporal Convolutional Network (Bai et al., 2018).

Input  : ``(B, T, F)`` with F = 131
Output : ``(B, num_classes)`` logits

4 residual blocks, kernel k=3, dilations (1, 2, 4, 8), 64 hidden channels.
Receptive field = 1 + 2·(k-1)·Σd = 61 frames ≥ T=32, so the last time step
"sees" the whole window while remaining strictly causal (streaming friendly).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """1D convolution with left-only padding ``(k-1)·d`` (no look-ahead)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """[CausalConv -> BN -> ReLU -> Dropout] × 2  + residual (1×1 conv if channels differ)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation), nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation), nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class CausalTCN(nn.Module):
    def __init__(self, in_features: int = 131, num_classes: int = 8, dropout: float = 0.2,
                 channels: int = 64, kernel_size: int = 3, dilations: Sequence[int] = (1, 2, 4, 8),
                 pooling: str = "last"):
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")
        layers, c_in = [], in_features
        for d in dilations:
            layers.append(TemporalBlock(c_in, channels, kernel_size, d, dropout))
            c_in = channels
        self.tcn = nn.Sequential(*layers)
        self.pooling = pooling
        self.fc = nn.Linear(channels, num_classes)

    @property
    def receptive_field(self) -> int:
        k = self.tcn[0].net[0].kernel_size[0]
        return 1 + 2 * (k - 1) * sum(b.net[0].dilation[0] for b in self.tcn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))  # (B, C, T)
        z = h[:, :, -1] if self.pooling == "last" else h.mean(dim=2)
        return self.fc(z)
