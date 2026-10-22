"""Training engine + E1–E12 experiment runner + ablation studies (TRD §3.2–3.3).

* CrossEntropy(label_smoothing=0.1), gradient clipping 1.0
* Early stopping on validation Macro-F1 (patience 10)
* Multi-seed runs (42, 100, 2026) reported as mean ± SD
* Schedulers: constant | cosine | plateau | onecycle | step | cosine_warmup

Usage::

    python -m src.training.train --exp E1                 # one scenario, 3 seeds
    python -m src.training.train --all                    # E1 … E12
    python -m src.training.train --ablation-suite         # 5 ablations × best config per model
    python -m src.training.train --exp E10 --ablation no_global
    python -m src.training.train --exp E9 --seeds 42 --epochs 2   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data.dataset import build_dataloaders
from src.models import build_model, count_parameters, dummy_input_shape
from src.training.evaluate import compute_metrics, measure_torch_latency, predict, save_report
from src.utils import apply_overrides, class_names, get_device, load_config, resolve_path, set_seed

SUMMARY_FILE = "experiments_summary.csv"


# -----------------------------------------------------------------------------
# Optimiser & scheduler factories
# -----------------------------------------------------------------------------
def build_optimizer(model: nn.Module, exp: dict) -> torch.optim.Optimizer:
    name = exp["optimizer"].lower()
    lr, wd = float(exp["lr"]), float(exp["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(opt: torch.optim.Optimizer, exp: dict, steps_per_epoch: int):
    """Return ``(scheduler | None, mode)`` where mode ∈ {none, epoch, batch, plateau}."""
    sc = exp.get("scheduler", {"name": "constant"})
    name = sc["name"].lower()
    epochs = int(exp["epochs"])
    if name == "constant":
        return None, "none"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), "epoch"
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=sc.get("factor", 0.5), patience=sc.get("patience", 5)), "plateau"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(exp["lr"]), epochs=epochs, steps_per_epoch=max(steps_per_epoch, 1)), "batch"
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=sc.get("milestones", [40, 60]), gamma=sc.get("gamma", 0.1)), "epoch"
    if name == "cosine_warmup":
        w = int(sc.get("warmup_epochs", 5))

        def lr_lambda(e: int) -> float:
            if e < w:
                return (e + 1) / w
            return 0.5 * (1 + math.cos(math.pi * (e - w) / max(epochs - w, 1)))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), "epoch"
    raise ValueError(f"Unknown scheduler {name}")


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, opt, device, grad_clip, sched, sched_mode) -> tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if sched_mode == "batch":
            sched.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int) -> tuple[float, float, float]:
    model.eval()
    total, loss_sum, ys, ps = 0, 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        total += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    f1 = f1_score(y, p, labels=list(range(num_classes)), average="macro", zero_division=0)
    return loss_sum / max(total, 1), float((y == p).mean()), float(f1)


# -----------------------------------------------------------------------------
# Single (experiment, seed) run
# -----------------------------------------------------------------------------
def run_single(cfg: dict, exp_id: str, exp: dict, seed: int, run_id: str,
               overrides: Optional[dict] = None) -> dict:
    """Train one model with one seed; returns validation + test metrics."""
    set_seed(seed)
    device = get_device(cfg["training"]["device"])
    names = class_names(cfg)
    n_cls = len(names)
    T = int(cfg["features"]["T"])
    use_global = bool(cfg["features"]["use_global"])
    model_name = exp["model"]

    loaders = build_dataloaders(cfg, model_name, int(exp["batch_size"]), seed)
    model = build_model(model_name, n_cls, float(exp["dropout"]), use_global,
                        cfg["models"][model_name]).to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
    opt = build_optimizer(model, exp)
    sched, mode = build_scheduler(opt, exp, len(loaders["train"]))

    run_dir = resolve_path(cfg, "runs_dir") / run_id / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    meta = {
        "exp_id": exp_id, "run_id": run_id, "seed": seed, "model": model_name,
        "num_classes": n_cls, "class_names": names, "dropout": float(exp["dropout"]),
        "use_global": use_global, "T": T, "model_cfg": cfg["models"][model_name],
        "overrides": overrides or {}, "experiment": exp, "params": n_params,
    }

    patience = int(cfg["training"]["patience"])
    grad_clip = float(cfg["training"]["grad_clip"])
    best_f1, best_epoch, bad, history = -1.0, 0, 0, []
    print(f"\n> {run_id} | seed {seed} | {model_name} | {n_params/1e6:.3f}M params | device {device}")
    for epoch in range(1, int(exp["epochs"]) + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, opt, device,
                                          grad_clip, sched, mode)
        va_loss, va_acc, va_f1 = validate(model, loaders["val"], criterion, device, n_cls)
        if mode == "epoch":
            sched.step()
        elif mode == "plateau":
            sched.step(va_f1)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc, "val_macro_f1": va_f1, "lr": lr, "time_s": time.time() - t0})
        improved = va_f1 > best_f1
        if improved:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            torch.save({"state_dict": model.state_dict(), "meta": {**meta, "epoch": epoch,
                        "val_macro_f1": va_f1}}, ckpt_path)
        else:
            bad += 1
        print(f"  ep {epoch:03d} | loss {tr_loss:.4f}/{va_loss:.4f} | acc {tr_acc:.3f}/{va_acc:.3f} "
              f"| valF1 {va_f1:.4f}{' *' if improved else ''} | lr {lr:.2e} | {time.time()-t0:.1f}s")
        if bad >= patience:
            print(f"  early stopping at epoch {epoch} (best {best_epoch}, F1 {best_f1:.4f})")
            break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # --- test evaluation with the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    y, yp, _ = predict(model, loaders["test"], device)
    metrics = compute_metrics(y, yp, names, cfg["evaluation"]["control_classes"])
    ev = cfg["evaluation"]
    metrics["latency_torch_cpu"] = measure_torch_latency(
        model, dummy_input_shape(model_name, T, use_global), ev["latency_warmup"], ev["latency_runs"])
    metrics.update(best_epoch=best_epoch, val_macro_f1=best_f1, params=n_params, seed=seed)
    res_dir = resolve_path(cfg, "results_dir") / run_id / f"seed{seed}"
    save_report(metrics, res_dir, names, "test")
    return {**metrics, "checkpoint": str(ckpt_path), "history": str(run_dir / "history.csv")}


# -----------------------------------------------------------------------------
# Multi-seed experiment
# -----------------------------------------------------------------------------
def _ms(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def upsert_summary(results_dir: Path, row: dict) -> None:
    path = results_dir / SUMMARY_FILE
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not df.empty and "run_id" in df:
        df = df[df["run_id"] != row["run_id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def run_experiment(base_cfg: dict, exp_id: str, seeds: list[int], ablation: Optional[str] = None,
                   epochs_override: Optional[int] = None) -> dict:
    """Run ``exp_id`` (optionally with an ablation) over ``seeds`` and record mean ± SD."""
    overrides = dict(base_cfg["ablations"]["studies"][ablation]) if ablation else {}
    cfg = apply_overrides(base_cfg, overrides)
    exp = dict(cfg["experiments"][exp_id])
    if epochs_override:
        exp["epochs"] = epochs_override
    run_id = f"{exp_id}__{ablation}" if ablation else exp_id

    results = [run_single(cfg, exp_id, exp, s, run_id, overrides) for s in seeds]
    best = max(results, key=lambda r: r["val_macro_f1"])
    names = class_names(cfg)
    res_dir = resolve_path(cfg, "results_dir") / run_id
    save_report(best, res_dir, names, "test_bestseed")

    vf, vs = _ms([r["val_macro_f1"] for r in results])
    tf, ts = _ms([r["macro_f1"] for r in results])
    ta, tas = _ms([r["accuracy"] for r in results])
    row = {
        "run_id": run_id, "exp_id": exp_id, "model": exp["model"], "ablation": ablation or "none",
        "optimizer": exp["optimizer"], "lr": exp["lr"], "batch_size": exp["batch_size"],
        "epochs": exp["epochs"], "dropout": exp["dropout"], "weight_decay": exp["weight_decay"],
        "scheduler": exp["scheduler"]["name"], "T": cfg["features"]["T"],
        "use_global": cfg["features"]["use_global"], "augmentation": cfg["augmentation"]["enabled"],
        "num_classes": len(names), "params": best["params"], "n_seeds": len(seeds),
        "val_f1_mean": vf, "val_f1_std": vs, "test_f1_mean": tf, "test_f1_std": ts,
        "test_acc_mean": ta, "test_acc_std": tas,
        "test_precision_mean": _ms([r["macro_precision"] for r in results])[0],
        "test_recall_mean": _ms([r["macro_recall"] for r in results])[0],
        "control_precision_mean": _ms([r.get("control_precision", np.nan) for r in results])[0],
        "latency_ms_mean": _ms([r["latency_torch_cpu"]["mean_ms"] for r in results])[0],
        "best_epoch_mean": _ms([r["best_epoch"] for r in results])[0],
        "best_seed": best["seed"], "best_checkpoint": best["checkpoint"],
    }
    upsert_summary(resolve_path(cfg, "results_dir"), row)
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"row": row, "seeds": [{k: r[k] for k in ("seed", "macro_f1", "accuracy",
                   "val_macro_f1", "best_epoch", "checkpoint")} for r in results]}, f, indent=2)
    print(f"\n[OK] {run_id}: val F1 {vf:.4f}+/-{vs:.4f} | test F1 {tf:.4f}+/-{ts:.4f} | acc {ta:.4f}+/-{tas:.4f}")
    return row


def select_best_experiments(cfg: dict) -> dict:
    """Best E-scenario per model by mean *validation* Macro-F1 (test set stays untouched)."""
    spec = cfg["ablations"].get("best_experiments", "auto")
    if isinstance(spec, dict):
        return spec
    path = resolve_path(cfg, "results_dir") / SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError("Run E1–E12 first (results/experiments_summary.csv missing).")
    df = pd.read_csv(path)
    df = df[df["ablation"] == "none"]
    best = df.loc[df.groupby("model")["val_f1_mean"].idxmax()]
    return dict(zip(best["model"], best["exp_id"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train E1–E12 scenarios and ablations")
    ap.add_argument("--config", default=None)
    ap.add_argument("--exp", nargs="+", default=None, help="experiment ids, e.g. E1 E5 E9")
    ap.add_argument("--all", action="store_true", help="run E1 … E12")
    ap.add_argument("--ablation", default=None, help="ablation name applied to --exp")
    ap.add_argument("--ablation-suite", action="store_true",
                    help="run every ablation on the best scenario of every model")
    ap.add_argument("--models", nargs="+", default=None, help="restrict --ablation-suite models")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or cfg["training"]["seeds"]

    if args.ablation_suite:
        best = select_best_experiments(cfg)
        for model, exp_id in best.items():
            if args.models and model not in args.models:
                continue
            for study in cfg["ablations"]["studies"]:
                run_experiment(cfg, exp_id, seeds, study, args.epochs)
        return

    exp_ids = list(cfg["experiments"]) if args.all else (args.exp or [])
    if not exp_ids:
        ap.error("specify --exp, --all or --ablation-suite")
    for exp_id in exp_ids:
        if exp_id not in cfg["experiments"]:
            ap.error(f"unknown experiment {exp_id}")
        run_experiment(cfg, exp_id, seeds, args.ablation, args.epochs)


if __name__ == "__main__":
    main()
