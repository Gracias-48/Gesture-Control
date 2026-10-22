"""Temporal resampling + Dual-Stream Normalization (TRD §2.2–2.3).

Coordinate convention
---------------------
MediaPipe returns x, y normalised by image width / height. Because 20BN-Jester
clips (~176x100) and a webcam (640x480) have different aspect ratios, every
landmark is converted to *aspect-corrected, centre-origin* coordinates in
image-height units before anything else::

    x' = (x - 0.5) * (W / H),   y' = y - 0.5,   z' = z * (W / H)

This makes the global (wrist trajectory) stream comparable between training
data and live webcam input, and a horizontal flip becomes exactly ``x <- -x``.

Feature layout
--------------
Sequence models (BiLSTM / TCN) — ``(T, 131)`` per clip::

    [  0: 63]  local joints   p̂_{t,i} = (p_{t,i} - p_{t,0}) / (||p_{t,9} - p_{t,0}|| + eps)
    [ 63:126]  local velocity Δp̂_{t,i}
    [126:129]  global wrist position (x_0, y_0, z_0)            ┐ global stream
    [129:131]  planar wrist velocity (Δx_0, Δy_0)               ┘

Graph model (ST-GCN) — ``(10, T, 21)`` per clip, per joint channels::

    [0:3] local xyz, [3:6] local velocity, [6:9] wrist xyz broadcast to every
    joint, [9] per-joint global speed ||p_{t,i} - p_{t-1,i}||   (6:10 = global)

With ``use_global=False`` (ablation #1) the global block is dropped, giving
126 sequence features / 6 graph channels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

NUM_JOINTS = 21
WRIST, MIDDLE_MCP = 0, 9
SEQ_GLOBAL_SLICE = slice(126, 131)
GRAPH_GLOBAL_CHANNELS = (6, 7, 8, 9)

SEQUENCE_MODELS = {"bilstm", "tcn"}
GRAPH_MODELS = {"stgcn"}


def input_kind(model_name: str) -> str:
    """Return ``"sequence"`` or ``"graph"`` for a model name."""
    name = model_name.lower()
    if name in SEQUENCE_MODELS:
        return "sequence"
    if name in GRAPH_MODELS:
        return "graph"
    raise ValueError(f"Unknown model '{model_name}'")


def feature_dim(kind: str, use_global: bool = True) -> int:
    """Per-frame feature size (sequence) or channel count (graph)."""
    if kind == "sequence":
        return 131 if use_global else 126
    return 10 if use_global else 6


# -----------------------------------------------------------------------------
# Coordinate handling
# -----------------------------------------------------------------------------
def to_centered_coords(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert MediaPipe-normalised landmarks ``(..., 21, 3)`` to centred height units."""
    aspect = float(width) / float(height)
    out = np.asarray(landmarks, dtype=np.float32).copy()
    out[..., 0] = (out[..., 0] - 0.5) * aspect
    out[..., 1] = out[..., 1] - 0.5
    out[..., 2] = out[..., 2] * aspect
    return out


def interpolate_missing(seq: np.ndarray) -> tuple[Optional[np.ndarray], float]:
    """Linearly interpolate frames whose landmarks are NaN.

    ``p_t = p_{t1} + (t - t1) / (t2 - t1) * (p_{t2} - p_{t1})`` between the
    nearest valid frames; leading / trailing gaps copy the nearest valid frame.

    Returns ``(filled_seq, missing_ratio)``; ``filled_seq`` is ``None`` if no
    frame contains a hand.
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    valid = ~np.isnan(seq).reshape(n, -1).any(axis=1)
    missing_ratio = 1.0 - valid.mean() if n else 1.0
    if not valid.any():
        return None, 1.0
    if valid.all():
        return seq.copy(), 0.0
    t = np.arange(n)
    flat = seq.reshape(n, -1).copy()
    for d in range(flat.shape[1]):
        # np.interp clamps outside [t_valid_min, t_valid_max] -> nearest-valid fill
        flat[:, d] = np.interp(t, t[valid], flat[valid, d])
    return flat.reshape(seq.shape), float(missing_ratio)


def resample_uniform(seq: np.ndarray, T: int) -> np.ndarray:
    """Uniform temporal resampling of ``(N, ...)`` to exactly ``T`` frames (linear)."""
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n == T:
        return seq.copy()
    if n == 1:
        return np.repeat(seq, T, axis=0)
    src = np.linspace(0.0, n - 1, T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo).reshape((T,) + (1,) * (seq.ndim - 1)).astype(np.float32)
    return (1.0 - w) * seq[lo] + w * seq[hi]


def _temporal_diff(x: np.ndarray) -> np.ndarray:
    """First-order difference along axis 0 with a zero first frame."""
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


# -----------------------------------------------------------------------------
# Dual-stream features
# -----------------------------------------------------------------------------
def local_stream(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Wrist-centred, Middle-MCP-scaled joints ``(T, 21, 3)``."""
    wrist = seq[:, WRIST : WRIST + 1, :]
    centred = seq - wrist
    scale = np.linalg.norm(seq[:, MIDDLE_MCP, :] - seq[:, WRIST, :], axis=-1) + eps
    return centred / scale[:, None, None]


def sequence_features(seq: np.ndarray, use_global: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Raw ``(T, 21, 3)`` landmarks -> ``(T, 131)`` (or ``(T, 126)``) features."""
    T = seq.shape[0]
    local = local_stream(seq, eps)
    local_vel = _temporal_diff(local)
    parts = [local.reshape(T, -1), local_vel.reshape(T, -1)]
    if use_global:
        wrist = seq[:, WRIST, :]
        wrist_vel_xy = _temporal_diff(wrist[:, :2])
        parts += [wrist, wrist_vel_xy]
    return np.concatenate(parts, axis=1).astype(np.float32)


def graph_features(seq: np.ndarray, use_global: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Raw ``(T, 21, 3)`` landmarks -> ``(C=10, T, V=21)`` (or C=6) node features."""
    T, V, _ = seq.shape
    local = local_stream(seq, eps)
    local_vel = _temporal_diff(local)
    parts = [local, local_vel]
    if use_global:
        wrist = np.broadcast_to(seq[:, WRIST : WRIST + 1, :], (T, V, 3))
        speed = np.linalg.norm(_temporal_diff(seq), axis=-1, keepdims=True)
        parts += [wrist, speed]
    feats = np.concatenate(parts, axis=-1)  # (T, V, C)
    return np.ascontiguousarray(feats.transpose(2, 0, 1)).astype(np.float32)


def mean_joint_speed(seq: np.ndarray) -> float:
    """Average per-frame joint displacement (used by the real-time motion gate)."""
    if seq.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(seq, axis=0), axis=-1).mean())


# -----------------------------------------------------------------------------
# Z-score scaler (train statistics only)
# -----------------------------------------------------------------------------
@dataclass
class Scaler:
    """Per-feature (sequence) / per-channel (graph) z-score statistics."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    graph_mean: np.ndarray
    graph_std: np.ndarray

    @classmethod
    def fit(cls, clips: Iterable[np.ndarray], eps: float = 1e-6) -> "Scaler":
        """Fit on full-feature (``use_global=True``) representations of raw clips."""
        seq_sum = np.zeros(131, np.float64)
        seq_sq = np.zeros(131, np.float64)
        g_sum = np.zeros(10, np.float64)
        g_sq = np.zeros(10, np.float64)
        n_seq = n_g = 0
        for clip in clips:
            s = sequence_features(clip, True, eps).astype(np.float64)
            g = graph_features(clip, True, eps).astype(np.float64)
            seq_sum += s.sum(0)
            seq_sq += (s**2).sum(0)
            n_seq += s.shape[0]
            g_sum += g.sum(axis=(1, 2))
            g_sq += (g**2).sum(axis=(1, 2))
            n_g += g.shape[1] * g.shape[2]
        seq_mean = seq_sum / max(n_seq, 1)
        g_mean = g_sum / max(n_g, 1)
        seq_std = np.sqrt(np.maximum(seq_sq / max(n_seq, 1) - seq_mean**2, 0.0))
        g_std = np.sqrt(np.maximum(g_sq / max(n_g, 1) - g_mean**2, 0.0))
        return cls(
            seq_mean.astype(np.float32),
            np.maximum(seq_std, 1e-6).astype(np.float32),
            g_mean.astype(np.float32),
            np.maximum(g_std, 1e-6).astype(np.float32),
        )

    @classmethod
    def identity(cls) -> "Scaler":
        """No-op scaler (used for smoke tests / dummy exports)."""
        return cls(np.zeros(131, np.float32), np.ones(131, np.float32),
                   np.zeros(10, np.float32), np.ones(10, np.float32))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, seq_mean=self.seq_mean, seq_std=self.seq_std,
                 graph_mean=self.graph_mean, graph_std=self.graph_std)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        d = np.load(path)
        return cls(d["seq_mean"], d["seq_std"], d["graph_mean"], d["graph_std"])

    def stats(self, kind: str, use_global: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) aligned with the selected feature layout."""
        if kind == "sequence":
            mean, std = self.seq_mean, self.seq_std
            return (mean, std) if use_global else (mean[:126], std[:126])
        mean, std = self.graph_mean, self.graph_std
        return (mean, std) if use_global else (mean[:6], std[:6])


class FeaturePipeline:
    """Raw landmark window ``(T, 21, 3)`` -> standardised model input (no batch dim).

    Shared by the PyTorch ``Dataset`` and the real-time ONNX engine so that the
    training and deployment feature paths are byte-for-byte identical.
    """

    def __init__(self, kind: str, scaler: Scaler, use_global: bool = True, eps: float = 1e-6):
        if kind not in ("sequence", "graph"):
            raise ValueError(kind)
        self.kind = kind
        self.use_global = use_global
        self.eps = eps
        mean, std = scaler.stats(kind, use_global)
        if kind == "sequence":
            self.mean, self.std = mean[None, :], std[None, :]
        else:
            self.mean, self.std = mean[:, None, None], std[:, None, None]

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if self.kind == "sequence":
            x = sequence_features(seq, self.use_global, self.eps)
        else:
            x = graph_features(seq, self.use_global, self.eps)
        return ((x - self.mean) / self.std).astype(np.float32)


# -----------------------------------------------------------------------------
# CLI: interim per-clip landmarks -> stratified splits + scaler
# -----------------------------------------------------------------------------
def build_splits(cfg: dict, T: int) -> None:
    """Stratified 70/15/15 split, resample every clip to ``T`` and fit the scaler."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    from src.utils import resolve_path

    interim = resolve_path(cfg, "interim_dir")
    index_path = interim / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found — run src.data.extract_landmarks first.")
    index = pd.read_csv(index_path, dtype={"video_id": str})
    index = index[index["status"] == "ok"].reset_index(drop=True)
    if index.empty:
        raise RuntimeError("No valid clips in index.csv")

    sp = cfg["split"]
    seed = sp["random_state"]
    train_df, rest = train_test_split(index, train_size=sp["train"], stratify=index["label"],
                                      random_state=seed)
    rel_val = sp["val"] / (sp["val"] + sp["test"])
    val_df, test_df = train_test_split(rest, train_size=rel_val, stratify=rest["label"],
                                       random_state=seed)

    out_dir = resolve_path(cfg, "processed_dir") / f"T{T}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_clips = None
    for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        X = np.zeros((len(df), T, NUM_JOINTS, 3), np.float32)
        for i, vid in enumerate(tqdm(df["video_id"], desc=f"{split} (T={T})")):
            X[i] = resample_uniform(np.load(interim / f"{vid}.npy"), T)
        y = df["label"].to_numpy(np.int64)
        np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y,
                            ids=df["video_id"].to_numpy(str))
        counts = np.bincount(y, minlength=len(cfg["classes"]["names"]))
        print(f"[{split}] {len(df)} clips | per-class: {counts.tolist()}")
        if split == "train":
            train_clips = X

    scaler = Scaler.fit(train_clips, cfg["features"]["eps"])
    scaler.save(out_dir / "scaler_stats.npz")
    if T == cfg["features"]["T"]:
        scaler.save(resolve_path(cfg, "scaler_stats"))
    print(f"Saved splits + scaler_stats.npz to {out_dir}")


def main() -> None:
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Build stratified T-frame splits + z-score scaler")
    ap.add_argument("--config", default=None)
    ap.add_argument("--T", type=int, nargs="+", default=None,
                    help="window lengths to build (default: features.T; ablation uses 16 32 48)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for T in args.T or [cfg["features"]["T"]]:
        build_splits(cfg, T)


if __name__ == "__main__":
    main()
