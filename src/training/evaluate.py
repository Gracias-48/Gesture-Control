"""Evaluation: Macro-F1, per-class P/R/F1, normalised confusion matrix, inference latency.

Usage::

    python -m src.training.evaluate --checkpoint runs/E10/seed42/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str],
                    control_classes: Sequence[int] | None = None) -> dict:
    """Return accuracy, macro P/R/F1, per-class table and row-normalised confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "per_class": [
            {"class_id": i, "name": n, "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])}
            for i, n in enumerate(class_names)
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }
    if control_classes:
        cc = [c for c in control_classes if c < len(class_names)]
        out["control_precision"] = float(p[cc].mean())  # prioritised over recall (PRD §5)
        out["control_recall"] = float(r[cc].mean())
    return out


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, probs)`` for every sample in ``loader``."""
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        pr = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(pr)
        preds.append(pr.argmax(1))
        ys.append(np.asarray(y))
    if not ys:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 0))
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


# -----------------------------------------------------------------------------
# Latency
# -----------------------------------------------------------------------------
def _summarise(times_ms: list[float]) -> dict:
    a = np.asarray(times_ms)
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


@torch.no_grad()
def measure_torch_latency(model: torch.nn.Module, input_shape: Sequence[int], warmup: int = 20,
                          runs: int = 200, threads: int | None = None) -> dict:
    """Single-window CPU latency of the PyTorch model (batch = 1)."""
    if threads:
        torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    x = torch.randn(*input_shape)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return _summarise(times)


def measure_onnx_latency(onnx_path: str | Path, input_shape: Sequence[int], warmup: int = 20,
                         runs: int = 200, threads: int = 2) -> dict:
    """Single-window latency with ONNX Runtime CPUExecutionProvider."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    return _summarise(times)


# -----------------------------------------------------------------------------
# Plotting / persistence
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm_norm: Sequence[Sequence[float]], class_names: Sequence[str],
                          out_path: str | Path, title: str = "Normalized Confusion Matrix") -> None:
    """Save a Raycast-dark styled, row-normalised confusion matrix PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(cm_norm)
    cmap = LinearSegmentedColormap.from_list("raycast", ["#101111", "#3a1f22", "#ff6363"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=130, facecolor="#07080a")
    ax.set_facecolor("#07080a")
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right", color="#9c9c9d", fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, color="#9c9c9d", fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#f9f9f9" if cm[i, j] > 0.35 else "#6a6b6c", family="monospace")
    ax.set_xlabel("Predicted", color="#f9f9f9")
    ax.set_ylabel("True", color="#f9f9f9")
    ax.set_title(title, color="#f9f9f9", fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_color("#242728")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color="#6a6b6c", labelcolor="#9c9c9d")
    cb.outline.set_edgecolor("#242728")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_report(metrics: dict, out_dir: str | Path, class_names: Sequence[str], prefix: str = "test") -> None:
    """Write ``{prefix}_metrics.json``, ``{prefix}_per_class.csv`` and the CM image."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics["per_class"]).to_csv(out_dir / f"{prefix}_per_class.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix_normalized"], class_names,
                          out_dir / f"{prefix}_confusion_matrix.png",
                          f"{prefix.upper()} — Macro F1 {metrics['macro_f1']:.3f}")


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device="cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild a model from a checkpoint written by ``train.py``."""
    from src.models import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["model"], meta["num_classes"], meta["dropout"], meta["use_global"],
                        meta["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def main() -> None:
    from src.data.dataset import build_dataloaders
    from src.models import dummy_input_shape
    from src.utils import apply_overrides, load_config

    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: checkpoint dir)")
    args = ap.parse_args()

    model, meta = load_checkpoint(args.checkpoint)
    cfg = apply_overrides(load_config(args.config), meta.get("overrides", {}))
    loaders = build_dataloaders(cfg, meta["model"], 256, meta["seed"], splits=[args.split])
    y, yp, _ = predict(model, loaders[args.split], torch.device("cpu"))
    metrics = compute_metrics(y, yp, meta["class_names"], cfg["evaluation"]["control_classes"])
    ev = cfg["evaluation"]
    metrics["latency_torch_cpu"] = measure_torch_latency(
        model, dummy_input_shape(meta["model"], meta["T"], meta["use_global"]),
        ev["latency_warmup"], ev["latency_runs"])
    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent
    save_report(metrics, out_dir, meta["class_names"], args.split)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "macro_precision",
                                              "macro_recall")}, indent=2))
    print(f"Latency (PyTorch CPU, batch=1): {metrics['latency_torch_cpu']['mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
