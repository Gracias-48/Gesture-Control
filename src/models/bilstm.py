"""Model 1 — BiLSTM baseline with temporal attention pooling (TRD §3.1, ≈0.6 M params).

Input  : ``(B, T, F)`` with F = 131 (126 without global stream)
Output : ``(B, num_classes)`` logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionPooling(nn.Module):
    """α_t = softmax_t( wᵀ tanh(W_a h_t) );  c = Σ_t α_t h_t."""

    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_a = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(torch.tanh(self.W_a(h))).squeeze(-1)  # (B, T)
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, dim)
        return context, alpha


class BiLSTMClassifier(nn.Module):
    """Linear(F->128)+LayerNorm+ReLU -> 2-layer BiLSTM(128/dir) -> attention -> MLP head."""

    def __init__(self, in_features: int = 131, num_classes: int = 8, dropout: float = 0.3,
                 proj_dim: int = 128, hidden_size: int = 128, num_layers: int = 2,
                 attn_dim: int = 128, fc_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttentionPooling(2 * hidden_size, attn_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, fc_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h, _ = self.lstm(self.input_proj(x))  # (B, T, 2H)
        context, alpha = self.attention(h)
        logits = self.head(context)
        return (logits, alpha) if return_attention else logits
