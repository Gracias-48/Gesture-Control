"""Generate synthetic landmark sequences for smoke-testing the end-to-end pipeline.

Menghasilkan file `.npy` pada `data/interim_landmarks/` dan mengeksekusi
`preprocessing.build_splits` untuk membuat train/val/test splits (T=32)
serta scaler_stats.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import NUM_JOINTS, build_splits
from src.utils.config import load_config, resolve_path


def generate_synthetic_data(cfg: dict, samples_per_class: int = 15, T_raw: int = 36) -> None:
    interim = resolve_path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)

    names = cfg["classes"]["names"]
    rows = []
    rng = np.random.default_rng(cfg["project"].get("seed", 42))

    print(f"Generating synthetic landmark data ({samples_per_class} samples/class across {len(names)} classes)...")

    for class_id, class_name in enumerate(names):
        for s in range(samples_per_class):
            vid = f"synthetic_{class_id}_{s:03d}"
            # Landmark skeleton dasar: 21 titik di sekitar origin
            # Buat pola gerakan sederhana sesuai kelas
            base_hand = rng.normal(0.0, 0.1, size=(NUM_JOINTS, 3)).astype(np.float32)
            # Pastikan wrist di origin awal
            base_hand[0] = 0.0

            seq = np.zeros((T_raw, NUM_JOINTS, 3), dtype=np.float32)
            # Gerakan trajectory wrist (kelas 0 swipe left: dx < 0, kelas 1 swipe right: dx > 0, dll)
            dx, dy, dz = 0.0, 0.0, 0.0
            if class_id == 0:  # Swipe Left
                dx = -0.015
            elif class_id == 1:  # Swipe Right
                dx = 0.015
            elif class_id == 2:  # Scroll Up
                dy = -0.012
            elif class_id == 3:  # Scroll Down
                dy = 0.012
            elif class_id == 4:  # Zoom In
                dz = 0.015
            elif class_id == 5:  # Zoom Out
                dz = -0.015

            for t in range(T_raw):
                jitter = rng.normal(0.0, 0.005, size=(NUM_JOINTS, 3))
                offset = np.array([dx * t, dy * t, dz * t], dtype=np.float32)
                seq[t] = base_hand + offset + jitter

            out_path = interim / f"{vid}.npy"
            np.save(out_path, seq)
            rows.append({
                "video_id": vid,
                "label": class_id,
                "jester_label": class_name,
                "n_frames": T_raw,
                "missing_ratio": 0.0,
                "status": "ok",
            })

    index_df = pd.DataFrame(rows)
    index_df.to_csv(interim / "index.csv", index=False)
    print(f"Wrote index.csv with {len(rows)} clips to {interim}")

    # Build T=32 splits & scaler
    print("Building splits (T=32) and computing z-score scaler...")
    build_splits(cfg, T=int(cfg["features"]["T"]))
    print("Synthetic data setup complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic landmarks for testing")
    parser.add_argument("--config", default=None)
    parser.add_argument("--samples-per-class", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_synthetic_data(cfg, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
