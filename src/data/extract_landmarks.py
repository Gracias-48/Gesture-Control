"""Offline MediaPipe Hands extraction for the 20BN-Jester 8-class subset (TRD §2.2).

Pipeline per clip
-----------------
1. Read every JPEG frame of ``<frames_dir>/<video_id>/``.
2. Bicubic upscale to height 256 px (native Jester height is 100 px).
3. MediaPipe Hands (``static_image_mode=False``, ``max_num_hands=1``,
   ``min_detection_confidence=0.5``) -> 21 (x, y, z) landmarks; a fresh tracker
   is created per clip so tracking state never leaks across videos.
4. Convert to aspect-corrected centred coordinates (see ``preprocessing``).
5. Linear interpolation of missing frames; clips with > 30 % missing frames
   are discarded.
6. Save ``<interim_dir>/<video_id>.npy`` (variable length ``(N, 21, 3)``) and
   append a row to ``<interim_dir>/index.csv``.

Usage::

    python -m src.data.extract_landmarks                # full subset
    python -m src.data.extract_landmarks --limit 200    # quick smoke run
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.preprocessing import NUM_JOINTS, interpolate_missing, to_centered_coords

# Populated per worker process by ``_init_worker``.
_WORKER_CFG: dict = {}


# -----------------------------------------------------------------------------
# Clip selection
# -----------------------------------------------------------------------------
def build_clip_list(cfg: dict) -> "list[tuple[str, int, str]]":
    """Return ``[(video_id, class_id, jester_label), ...]`` for the 8-class subset.

    The official labelled train + validation CSVs are merged (the Jester test
    set is unlabelled); Background is a seeded subsample of ``No gesture`` and
    ``Doing other things``.
    """
    import pandas as pd

    from src.utils import resolve_path

    frames = []
    for key in ("jester_train_csv", "jester_val_csv"):
        path = resolve_path(cfg, key)
        if not path.exists():
            raise FileNotFoundError(f"Jester annotation file missing: {path}")
        frames.append(pd.read_csv(path, sep=";", header=None, names=["video_id", "label"],
                                  dtype={"video_id": str}))
    df = pd.concat(frames, ignore_index=True)

    mapping: dict = cfg["classes"]["jester_mapping"]
    df = df[df["label"].isin(mapping.keys())].copy()

    bg_sources = set(cfg["classes"]["background_sources"])
    n_bg = int(cfg["classes"]["background_samples_per_source"])
    seed = int(cfg["split"]["random_state"])
    parts = [df[~df["label"].isin(bg_sources)]]
    for src in bg_sources:
        sub = df[df["label"] == src]
        parts.append(sub.sample(n=min(n_bg, len(sub)), random_state=seed))
    df = pd.concat(parts, ignore_index=True)
    df["class_id"] = df["label"].map(mapping).astype(int)
    return list(df[["video_id", "class_id", "label"]].itertuples(index=False, name=None))


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def _init_worker(cfg: dict) -> None:
    import cv2

    cv2.setNumThreads(1)
    _WORKER_CFG.clear()
    _WORKER_CFG.update(cfg)


def extract_clip(frame_paths: "list[Path]", ext_cfg: dict) -> tuple[np.ndarray, int, int]:
    """Run MediaPipe on ordered frames; returns ``(landmarks_with_nan, width, height)``."""
    import cv2
    import mediapipe as mp_lib

    target_h = int(ext_cfg["target_height"])
    seq = np.full((len(frame_paths), NUM_JOINTS, 3), np.nan, np.float32)
    width = height = 0
    with mp_lib.solutions.hands.Hands(
        static_image_mode=bool(ext_cfg["static_image_mode"]),
        max_num_hands=int(ext_cfg["max_num_hands"]),
        model_complexity=int(ext_cfg["model_complexity"]),
        min_detection_confidence=float(ext_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(ext_cfg["min_tracking_confidence"]),
    ) as hands:
        for t, fp in enumerate(frame_paths):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = target_h / float(h)
            img = cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)
            height, width = img.shape[:2]
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                seq[t] = [(p.x, p.y, p.z) for p in lm]
    return seq, width, height


def _process(item: tuple[str, int, str]) -> dict:
    video_id, class_id, label = item
    cfg = _WORKER_CFG
    frames_dir = Path(cfg["_frames_dir"]) / video_id
    out_path = Path(cfg["_interim_dir"]) / f"{video_id}.npy"
    row = {"video_id": video_id, "label": class_id, "jester_label": label,
           "n_frames": 0, "missing_ratio": 1.0, "status": "error"}
    try:
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            row["status"] = "no_frames"
            return row
        raw, w, h = extract_clip(frame_paths, cfg["extraction"])
        row["n_frames"] = len(frame_paths)
        valid = ~np.isnan(raw).reshape(len(raw), -1).any(axis=1)
        if valid.any():
            raw[valid] = to_centered_coords(raw[valid], w, h)
        filled, missing = interpolate_missing(raw)
        row["missing_ratio"] = round(float(missing), 4)
        if filled is None or missing > float(cfg["extraction"]["max_missing_ratio"]):
            row["status"] = "discarded_missing"
            return row
        np.save(out_path, filled.astype(np.float32))
        row["status"] = "ok"
    except Exception as exc:  # keep the pool alive; record the failure
        row["status"] = f"error:{type(exc).__name__}"
    return row


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(cfg: dict, limit: Optional[int] = None, workers: Optional[int] = None) -> None:
    import pandas as pd
    from tqdm import tqdm

    from src.utils import resolve_path

    interim = resolve_path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    clips = build_clip_list(cfg)
    index_path = interim / "index.csv"
    old = pd.read_csv(index_path, dtype={"video_id": str}) if index_path.exists() else None
    if old is not None:  # resumable: skip clips already processed in a previous run
        done = set(old["video_id"])
        clips = [c for c in clips if c[0] not in done]
    if limit:
        rng = np.random.default_rng(cfg["split"]["random_state"])
        clips = [clips[i] for i in rng.permutation(len(clips))[:limit]]
    worker_cfg = dict(cfg)
    worker_cfg["_frames_dir"] = str(resolve_path(cfg, "jester_frames_dir"))
    worker_cfg["_interim_dir"] = str(interim)
    n_workers = workers or int(cfg["extraction"]["num_workers"])
    print(f"Extracting {len(clips)} clips with {n_workers} workers -> {interim}")

    rows = []
    if n_workers <= 1:
        _init_worker(worker_cfg)
        rows = [_process(c) for c in tqdm(clips)]
    else:
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker,
                                          initargs=(worker_cfg,)) as pool:
            for r in tqdm(pool.imap_unordered(_process, clips, chunksize=8), total=len(clips)):
                rows.append(r)

    index = pd.DataFrame(rows)
    if old is not None:
        index = pd.concat([old, index]).drop_duplicates("video_id", keep="last")
    index.to_csv(index_path, index=False)
    print(index["status"].value_counts().to_string())
    print(f"Index written to {index_path}")


def main() -> None:
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Offline MediaPipe Hands extraction (Jester subset)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="process only N random clips")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.limit, args.workers)


if __name__ == "__main__":
    main()
