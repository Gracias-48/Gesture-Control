"""PyTorch -> ONNX export with verification (TRD §4).

* opset 17, constant folding, dynamic batch axis
* ``onnx.checker.check_model`` + numerical parity vs PyTorch (atol 1e-5)
* size check (< 5 MB) and ONNX Runtime CPU latency benchmark
* writes ``<onnx_dir>/<model>.onnx`` + ``<model>.json`` (sidecar metadata) +
  ``<model>_scaler.npz`` so the real-time engine is fully self-contained

Usage::

    python -m src.training.export_onnx --best                       # best ckpt of each model
    python -m src.training.export_onnx --checkpoint runs/E10/seed42/best.pt
    python -m src.training.export_onnx --dummy                      # UNTRAINED models (pipeline smoke test)
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import Scaler, feature_dim, input_kind
from src.models import DISPLAY_NAMES, MODEL_NAMES, build_model, count_parameters, dummy_input_shape
from src.training.evaluate import load_checkpoint, measure_onnx_latency
from src.utils import load_config, resolve_path


class ProbabilityHead(nn.Module):
    """Wrap a classifier so the exported graph emits softmax probabilities directly."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=-1)


def export_model(model: nn.Module, meta: dict, scaler_src: Path | None, out_dir: Path, cfg: dict,
                 dummy: bool = False) -> dict:
    """Export, verify and benchmark one model. Returns the sidecar metadata."""
    import onnx
    import onnxruntime as ort

    name = meta["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    wrapped = ProbabilityHead(model.cpu().eval()).eval()
    shape = dummy_input_shape(name, meta["T"], meta["use_global"])
    x = torch.randn(*shape)

    kwargs = dict(export_params=True, opset_version=int(cfg["onnx"]["opset"]), do_constant_folding=True,
                  input_names=["input"], output_names=["probs"],
                  dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # legacy TorchScript exporter: stable for LSTM / einsum
    torch.onnx.export(wrapped, (x,), str(onnx_path), **kwargs)

    # --- structural check + embedded metadata
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    for k, v in {"model": name, "T": str(meta["T"]), "use_global": str(meta["use_global"]),
                 "class_names": json.dumps(meta["class_names"]), "dummy": str(dummy)}.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = k, v
    onnx.save(m, str(onnx_path))

    # --- numerical parity (batch 1 and batch 4 to exercise the dynamic axis)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    atol, rtol = float(cfg["onnx"]["atol"]), float(cfg["onnx"]["rtol"])
    max_diff = 0.0
    for b in (1, 4):
        xb = torch.randn(b, *shape[1:])
        with torch.no_grad():
            ref = wrapped(xb).numpy()
        got = sess.run(None, {"input": xb.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)

    size_mb = onnx_path.stat().st_size / 2**20
    max_mb = float(cfg["onnx"]["max_size_mb"])
    if size_mb >= max_mb:
        raise RuntimeError(f"{onnx_path.name} is {size_mb:.2f} MB (limit {max_mb} MB)")

    ev = cfg["evaluation"]
    latency = measure_onnx_latency(onnx_path, shape, ev["latency_warmup"], ev["latency_runs"],
                                   int(cfg["inference"]["onnx_intra_op_threads"]))

    scaler_dst = out_dir / f"{name}_scaler.npz"
    if scaler_src is not None and Path(scaler_src).exists():
        shutil.copyfile(scaler_src, scaler_dst)
    else:
        Scaler.identity().save(scaler_dst)

    kind = input_kind(name)
    sidecar = {
        "model": name, "display_name": DISPLAY_NAMES[name], "kind": kind,
        "input_layout": "B,T,F" if kind == "sequence" else "B,C,T,V",
        "T": int(meta["T"]), "feature_dim": feature_dim(kind, meta["use_global"]),
        "use_global": bool(meta["use_global"]), "num_classes": int(meta["num_classes"]),
        "class_names": meta["class_names"], "input_name": "input", "output_name": "probs",
        "scaler": scaler_dst.name, "params": count_parameters(model), "size_mb": round(size_mb, 3),
        "parity_max_abs_diff": max_diff, "onnx_latency": latency, "opset": int(cfg["onnx"]["opset"]),
        "source_checkpoint": meta.get("checkpoint"), "exp_id": meta.get("exp_id"),
        "val_macro_f1": meta.get("val_macro_f1"), "dummy": dummy,
    }
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[OK] {onnx_path.name}: {size_mb:.2f} MB | params {sidecar['params']:,} | parity max diff {max_diff:.2e} "
          f"| ORT latency {latency['mean_ms']:.2f} ms (p95 {latency['p95_ms']:.2f})"
          + ("  [UNTRAINED DUMMY]" if dummy else ""))
    return sidecar


def export_checkpoint(ckpt: str | Path, cfg: dict) -> dict:
    model, meta = load_checkpoint(ckpt)
    meta["checkpoint"] = str(ckpt)
    scaler = resolve_path(cfg, "processed_dir") / f"T{meta['T']}" / "scaler_stats.npz"
    return export_model(model, meta, scaler, resolve_path(cfg, "onnx_dir"), cfg)


def export_best(cfg: dict) -> None:
    """Export the best-seed checkpoint of the best E-scenario (by val Macro-F1) per model."""
    import pandas as pd

    path = resolve_path(cfg, "results_dir") / "experiments_summary.csv"
    if not path.exists():
        raise FileNotFoundError("results/experiments_summary.csv missing — train E1–E12 first.")
    df = pd.read_csv(path)
    df = df[df["ablation"] == "none"]
    for model in MODEL_NAMES:
        sub = df[df["model"] == model]
        if sub.empty:
            print(f"- {model}: no trained runs, skipped")
            continue
        row = sub.loc[sub["val_f1_mean"].idxmax()]
        print(f"- {model}: exporting {row['run_id']} (seed {row['best_seed']}, val F1 {row['val_f1_mean']:.4f})")
        export_checkpoint(row["best_checkpoint"], cfg)


def export_dummy(cfg: dict) -> None:
    """Export randomly initialised models — ONLY for verifying the real-time pipeline/UI."""
    T = int(cfg["features"]["T"])
    names = list(cfg["classes"]["names"])
    for name in MODEL_NAMES:
        model = build_model(name, len(names), 0.0, True, cfg["models"][name])
        meta = {"model": name, "T": T, "use_global": True, "num_classes": len(names),
                "class_names": names, "exp_id": "untrained"}
        export_model(model, meta, None, resolve_path(cfg, "onnx_dir"), cfg, dummy=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export trained models to ONNX")
    ap.add_argument("--config", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="path to a best.pt checkpoint")
    g.add_argument("--best", action="store_true", help="export best checkpoint of each model")
    g.add_argument("--dummy", action="store_true", help="export UNTRAINED models for pipeline tests")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint:
        export_checkpoint(args.checkpoint, cfg)
    elif args.best:
        export_best(cfg)
    else:
        export_dummy(cfg)


if __name__ == "__main__":
    main()
