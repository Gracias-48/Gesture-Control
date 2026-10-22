"""PyTorch Dataset / DataLoader with spatio-temporal augmentation (TRD §2.4).

Augmentations operate on *raw* aspect-corrected landmarks ``(T, 21, 3)`` before
dual-stream feature computation, so the global stream (wrist trajectory and
velocity) is perturbed consistently with the local stream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import FeaturePipeline, Scaler, input_kind, resample_uniform


# -----------------------------------------------------------------------------
# Augmentations (each takes / returns raw (T, 21, 3) float32)
# -----------------------------------------------------------------------------
def _centroid(seq: np.ndarray) -> np.ndarray:
    return seq[..., :2].reshape(-1, 2).mean(axis=0)


def random_rotation(seq: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate in the (x, y) plane by θ ~ U(-max_deg, +max_deg) around the clip centroid."""
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], np.float32)
    out = seq.copy()
    ctr = _centroid(seq)
    out[..., :2] = (seq[..., :2] - ctr) @ rot.T + ctr
    return out


def random_scale(seq: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic scaling s ~ U(lo, hi) around the clip centroid (z scaled too)."""
    s = rng.uniform(lo, hi)
    out = seq.copy()
    ctr = _centroid(seq)
    out[..., :2] = (seq[..., :2] - ctr) * s + ctr
    out[..., 2] = seq[..., 2] * s
    return out


def gaussian_jitter(seq: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Additive N(0, std²) noise on every joint coordinate."""
    return (seq + rng.normal(0.0, std, size=seq.shape)).astype(np.float32)


def speed_perturbation(seq: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Play the clip at factor f ~ U(lo, hi) then restore length T.

    f > 1 (faster) -> fewer frames -> edge-padded back to T;
    f < 1 (slower) -> more frames -> randomly cropped back to T.
    """
    T = seq.shape[0]
    f = rng.uniform(lo, hi)
    L = max(2, int(round(T / f)))
    warped = resample_uniform(seq, L)
    if L == T:
        return warped
    if L > T:
        start = rng.integers(0, L - T + 1)
        return warped[start : start + T]
    pad = T - L
    left = rng.integers(0, pad + 1)
    return np.concatenate([np.repeat(warped[:1], left, 0), warped,
                           np.repeat(warped[-1:], pad - left, 0)], axis=0)


def frame_dropout(seq: np.ndarray, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Drop 5–10 % of random frames and re-fill them by linear interpolation.

    Simulates transient MediaPipe detection failures exactly the way the
    offline/online pipeline repairs them (zero-filling would inject spurious
    velocity spikes that never occur at inference time).
    """
    T = seq.shape[0]
    k = max(1, int(round(rng.uniform(lo, hi) * T)))
    drop = rng.choice(np.arange(1, T - 1), size=min(k, T - 2), replace=False)
    keep = np.setdiff1d(np.arange(T), drop)
    flat = seq.reshape(T, -1)
    out = np.empty_like(flat)
    t = np.arange(T)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(t, keep, flat[keep, d])
    return out.reshape(seq.shape)


def horizontal_flip(seq: np.ndarray, label: int, swap_pairs: Sequence[Sequence[int]]) -> tuple[np.ndarray, int]:
    """Mirror ``x <- -x`` and swap direction-dependent labels (Swipe Left <-> Swipe Right)."""
    out = seq.copy()
    out[..., 0] = -out[..., 0]
    for a, b in swap_pairs:
        if label == a:
            label = b
        elif label == b:
            label = a
    return out, label


class Augmenter:
    """Composes the train-time augmentations according to ``cfg['augmentation']``."""

    def __init__(self, aug_cfg: dict, swap_pairs: Sequence[Sequence[int]], seed: Optional[int] = None):
        self.c = aug_cfg
        self.swap_pairs = swap_pairs
        self.rng = np.random.default_rng(seed)

    def __call__(self, seq: np.ndarray, label: int) -> tuple[np.ndarray, int]:
        c, r = self.c, self.rng
        if r.random() < c["p_speed"]:
            seq = speed_perturbation(seq, *c["speed_range"], r)
        if r.random() < c["p_dropout"]:
            seq = frame_dropout(seq, *c["frame_dropout_range"], r)
        if r.random() < c["p_flip"]:
            seq, label = horizontal_flip(seq, label, self.swap_pairs)
        if r.random() < c["p_rotation"]:
            seq = random_rotation(seq, c["rotation_deg"], r)
        if r.random() < c["p_scale"]:
            seq = random_scale(seq, *c["scale_range"], r)
        if r.random() < c["p_jitter"]:
            seq = gaussian_jitter(seq, c["jitter_std"], r)
        return seq.astype(np.float32), label


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class GestureDataset(Dataset):
    """Loads ``{split}.npz`` (``X: (N, T, 21, 3)``, ``y: (N,)``) and emits model tensors.

    * sequence models -> ``(T, 131)``; graph model -> ``(10, T, 21)``.
    * ``exclude_background`` drops class 7 (ablation #4); remaining labels keep
      their ids 0–6, which remain valid for a 7-way classifier.
    * Non-augmented splits are featurised once and cached in memory.
    """

    def __init__(self, npz_path: str | Path, model_name: str, scaler: Scaler, *,
                 use_global: bool = True, augmenter: Optional[Augmenter] = None,
                 exclude_background: bool = False, background_id: int = 7, eps: float = 1e-6):
        data = np.load(npz_path)
        X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
        if exclude_background:
            keep = y != background_id
            X, y = X[keep], y[keep]
        self.X, self.y = X, y
        self.augmenter = augmenter
        self.pipeline = FeaturePipeline(input_kind(model_name), scaler, use_global, eps)
        self._cache = None
        if augmenter is None:
            self._cache = np.stack([self.pipeline(x) for x in self.X]) if len(X) else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return torch.from_numpy(self._cache[idx]), int(self.y[idx])
        seq, label = self.augmenter(self.X[idx], int(self.y[idx]))
        return torch.from_numpy(self.pipeline(seq)), label

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=8)


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker an independent augmentation RNG stream."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if getattr(ds, "augmenter", None) is not None:
        ds.augmenter.rng = np.random.default_rng(torch.initial_seed() % (2**32))


def build_dataloaders(cfg: dict, model_name: str, batch_size: int, seed: int,
                      splits: Sequence[str] = ("train", "val", "test")) -> dict:
    """Build DataLoaders for the requested splits using the active config."""
    from src.utils import resolve_path

    T = int(cfg["features"]["T"])
    data_dir = resolve_path(cfg, "processed_dir") / f"T{T}"
    scaler_path = data_dir / "scaler_stats.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"{scaler_path} missing — run `python -m src.data.preprocessing --T {T}`")
    scaler = Scaler.load(scaler_path)
    use_global = bool(cfg["features"]["use_global"])
    excl = bool(cfg["classes"].get("exclude_background", False))
    bg = int(cfg["classes"]["background_id"])
    eps = float(cfg["features"]["eps"])
    workers = int(cfg["training"]["num_workers"])

    loaders = {}
    for split in splits:
        aug = None
        if split == "train" and cfg["augmentation"]["enabled"]:
            aug = Augmenter(cfg["augmentation"], cfg["classes"]["flip_swap_pairs"], seed)
        ds = GestureDataset(data_dir / f"{split}.npz", model_name, scaler, use_global=use_global,
                            augmenter=aug, exclude_background=excl, background_id=bg, eps=eps)
        g = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), drop_last=False,
            num_workers=workers, pin_memory=torch.cuda.is_available(), generator=g,
            worker_init_fn=_worker_init if workers > 0 else None,
            persistent_workers=workers > 0,
        )
    return loaders
