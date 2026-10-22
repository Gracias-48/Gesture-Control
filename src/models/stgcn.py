"""Model 2 — Spatio-Temporal Graph Convolutional Network on the 21-joint hand (Yan et al., 2018).

Input  : ``(B, C=10, T, V=21)``
Output : ``(B, num_classes)`` logits

Graph: MediaPipe hand skeleton with the *spatial configuration* partition
(root / centripetal / centrifugal) relative to the wrist, giving
``A ∈ R^{3×21×21}``. Each of the 4 ST-GCN blocks has a learnable edge
importance mask ``M`` (A ⊙ M) and a residual connection.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# MediaPipe Hands topology (21 landmarks)
HAND_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]
NUM_NODES = 21
CENTER = 0  # wrist


def _bfs_distances(num_nodes: int, edges: Sequence[tuple[int, int]], src: int) -> np.ndarray:
    adj = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full(num_nodes, np.inf)
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_hand_adjacency(num_nodes: int = NUM_NODES, edges=HAND_EDGES, center: int = CENTER) -> np.ndarray:
    """Spatial-configuration partitioned, degree-normalised adjacency ``(3, V, V)``.

    ``A[k, j, i]`` is the weight with which node ``i`` aggregates neighbour ``j``:
      k=0 root        : j at the same distance to the wrist as i (incl. self loop)
      k=1 centripetal : j closer to the wrist than i
      k=2 centrifugal : j farther from the wrist than i
    """
    A = np.eye(num_nodes)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=0)
    A_norm = A / deg[None, :]  # column-normalised: A D^{-1}
    dist = _bfs_distances(num_nodes, edges, center)
    parts = np.zeros((3, num_nodes, num_nodes), np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[j, i] == 0:
                continue
            if dist[j] == dist[i]:
                parts[0, j, i] = A_norm[j, i]
            elif dist[j] < dist[i]:
                parts[1, j, i] = A_norm[j, i]
            else:
                parts[2, j, i] = A_norm[j, i]
    return parts


class SpatialGraphConv(nn.Module):
    """Σ_k  (X W_k) (A_k ⊙ M_k)  implemented as a 1×1 conv + einsum."""

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int):
        super().__init__()
        self.K = num_partitions
        self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.K, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A).contiguous()


class STGCNBlock(nn.Module):
    """GCN -> BN -> ReLU -> temporal conv (Kt×1) -> BN -> Dropout, + residual, ReLU."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, kt: int = 9,
                 stride: int = 1, dropout: float = 0.0, edge_importance: bool = True):
        super().__init__()
        K = A.shape[0]
        self.gcn = SpatialGraphConv(in_channels, out_channels, K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)), nn.BatchNorm2d(out_channels),
            )
        self.edge_importance = nn.Parameter(torch.ones_like(A)) if edge_importance else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_eff = A * self.edge_importance if self.edge_importance is not None else A
        return self.relu(self.tcn(self.gcn(x, A_eff)) + self.residual(x))


class STGCN(nn.Module):
    """Input BN -> 4 ST-GCN blocks (64, 64, 128, 128) -> global avg pool over (T, V) -> FC."""

    def __init__(self, in_channels: int = 10, num_classes: int = 8, dropout: float = 0.3,
                 channels: Sequence[int] = (64, 64, 128, 128), strides: Sequence[int] = (1, 1, 2, 1),
                 temporal_kernel: int = 9, edge_importance: bool = True, num_nodes: int = NUM_NODES):
        super().__init__()
        self.register_buffer("A", torch.tensor(build_hand_adjacency(num_nodes), dtype=torch.float32))
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        blocks, c_in = [], in_channels
        for c_out, s in zip(channels, strides):
            blocks.append(STGCNBlock(c_in, c_out, self.A, temporal_kernel, s, dropout, edge_importance))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)          # (N, V*C, T)
        x = self.data_bn(x).view(n, v, c, t).permute(0, 2, 3, 1)  # (N, C, T, V)
        for blk in self.blocks:
            x = blk(x, self.A)
        x = x.mean(dim=(2, 3))
        return self.fc(self.dropout(x))
