"""High-fidelity kinematic gesture dataset generator (TRD §2.2–2.4).

Membuat dataset gestur spatio-temporal dengan kinematika tangan anatomis yang realistis
untuk 8 kelas perintah 20BN-Jester. Menghasilkan pola lintasan artikulasi jari dan vektor kecepatan
yang konsisten dengan gerakan tangan manusia di depan webcam, sehingga model tidak mengalami
mode collapse dan mampu mengenali gerakan tangan secara diskriminatif.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import NUM_JOINTS, build_splits
from src.utils.config import load_config, resolve_path

# Titik landmark MediaPipe standar telapak tangan netral (aspect-corrected, centered)
CANONICAL_HAND = np.array([
    [0.00, 0.20, 0.00],    # 0: Wrist
    [-0.04, 0.14, -0.01],  # 1: Thumb CMC
    [-0.08, 0.09, -0.02],  # 2: Thumb MCP
    [-0.11, 0.05, -0.03],  # 3: Thumb IP
    [-0.13, 0.02, -0.04],  # 4: Thumb Tip
    [-0.04, 0.03, -0.01],  # 5: Index MCP
    [-0.05, -0.04, -0.01], # 6: Index PIP
    [-0.05, -0.10, -0.02], # 7: Index DIP
    [-0.05, -0.15, -0.02], # 8: Index Tip
    [0.00, 0.02, 0.00],    # 9: Middle MCP
    [0.00, -0.06, 0.00],   # 10: Middle PIP
    [0.00, -0.13, 0.00],   # 11: Middle DIP
    [0.00, -0.19, 0.00],   # 12: Middle Tip
    [0.04, 0.03, 0.01],    # 13: Ring MCP
    [0.04, -0.04, 0.01],   # 14: Ring PIP
    [0.04, -0.10, 0.01],   # 15: Ring DIP
    [0.04, -0.16, 0.01],   # 16: Ring Tip
    [0.08, 0.05, 0.02],    # 17: Pinky MCP
    [0.09, -0.01, 0.02],   # 18: Pinky PIP
    [0.09, -0.06, 0.02],   # 19: Pinky DIP
    [0.09, -0.11, 0.02],   # 20: Pinky Tip
], dtype=np.float32)


def generate_kinematic_clip(
    class_id: int,
    T: int = 32,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 1 clip (T, 21, 3) koordinat tangan kontinu sesuai kelas gestur."""
    if rng is None:
        rng = np.random.default_rng()

    # Variasi anatomis ukuran tangan per pengguna
    hand_scale = rng.uniform(0.85, 1.15)
    base_hand = CANONICAL_HAND * hand_scale

    # Sedikit rotasi orientasi tangan awal
    roll = rng.uniform(-0.15, 0.15)
    c, s = math.cos(roll), math.sin(roll)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    base_hand = base_hand @ rot_matrix.T

    # Posisi awal pergelangan tangan (wrist)
    wrist_x0 = rng.uniform(-0.10, 0.10)
    wrist_y0 = rng.uniform(-0.05, 0.10)
    wrist_z0 = rng.uniform(-0.02, 0.02)

    seq = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

    # Buat kurva waktu ter-normalisasi [0, 1] dengan kurva kecepatan alami (smooth bell profile)
    t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)
    smooth_step = 0.5 * (1.0 - np.cos(np.pi * t_norm))  # 0 -> 1 smooth sigmoid

    if class_id == 0:
        # --- SWIPE LEFT: Wrist meluncur dari kanan ke kiri secara cepat & tegas ---
        amplitude = rng.uniform(0.35, 0.55)
        for t in range(T):
            dx = amplitude * (0.5 - smooth_step[t])  # dari +amp/2 ke -amp/2
            dy = 0.03 * math.sin(math.pi * t_norm[t]) # sedikit lengkungan vertikal
            seq[t] = base_hand + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    elif class_id == 1:
        # --- SWIPE RIGHT: Wrist meluncur dari kiri ke kanan ---
        amplitude = rng.uniform(0.35, 0.55)
        for t in range(T):
            dx = amplitude * (smooth_step[t] - 0.5)  # dari -amp/2 ke +amp/2
            dy = 0.03 * math.sin(math.pi * t_norm[t])
            seq[t] = base_hand + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    elif class_id == 2:
        # --- SCROLL UP: Telunjuk & Jari Tengah mengayun ke atas berulang kali ---
        cycle_freq = rng.uniform(1.8, 2.5)
        for t in range(T):
            wave = -0.07 * math.sin(2.0 * math.pi * cycle_freq * t_norm[t])  # ke atas (y negatif)
            hand_t = base_hand.copy()
            # Tekuk jari manis & kelingking ke bawah
            hand_t[[14, 15, 16, 18, 19, 20], 1] += 0.08
            # Ayunkan telunjuk dan jari tengah
            hand_t[[6, 7, 8, 10, 11, 12], 1] += wave
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 3:
        # --- SCROLL DOWN: Telunjuk & Jari Tengah mengayun ke bawah berulang kali ---
        cycle_freq = rng.uniform(1.8, 2.5)
        for t in range(T):
            wave = 0.07 * math.sin(2.0 * math.pi * cycle_freq * t_norm[t])  # ke bawah (y positif)
            hand_t = base_hand.copy()
            # Tekuk jari manis & kelingking
            hand_t[[14, 15, 16, 18, 19, 20], 1] += 0.08
            # Ayunkan telunjuk dan tengah ke bawah
            hand_t[[6, 7, 8, 10, 11, 12], 1] += wave
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 4:
        # --- ZOOM IN: Seluruh jari tangan mengembang melebar keluar (ekspansi radius) ---
        for t in range(T):
            exp_factor = 0.70 + 0.60 * smooth_step[t]  # dari 0.70x ke 1.30x
            hand_t = base_hand.copy()
            # Ujung jari mengembang menjauhi telapak tangan
            hand_t[1:] = hand_t[0:1] + (hand_t[1:] - hand_t[0:1]) * exp_factor
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 5:
        # --- ZOOM OUT: Seluruh jari menguncup rapat ke pusat telapak tangan ---
        for t in range(T):
            pinch_factor = 1.30 - 0.65 * smooth_step[t]  # dari 1.30x ke 0.65x
            hand_t = base_hand.copy()
            hand_t[1:] = hand_t[0:1] + (hand_t[1:] - hand_t[0:1]) * pinch_factor
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 6:
        # --- STOP SIGN: Telapak tangan terbuka tegak menghadap kamera, posisi sangat stabil ---
        for t in range(T):
            # Mikro-tremor natural tangan manusia yang sedang menahan pose diam
            tremor = rng.normal(0.0, 0.0015, size=(NUM_JOINTS, 3)).astype(np.float32)
            seq[t] = base_hand + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32) + tremor

    else:
        # --- BACKGROUND (ID 7): Gerakan santai, mengetik, diam, atau gerakan acak tanpa pola ---
        bg_type = rng.integers(0, 3)
        if bg_type == 0:
            # Tangan santai diam di bawah
            for t in range(T):
                tremor = rng.normal(0.0, 0.001, size=(NUM_JOINTS, 3)).astype(np.float32)
                seq[t] = base_hand + np.array([wrist_x0, wrist_y0 + 0.15, wrist_z0], dtype=np.float32) + tremor
        elif bg_type == 1:
            # Gerakan mengetik / artikulasi jari kecil di tempat
            for t in range(T):
                hand_t = base_hand.copy()
                finger_wiggle = rng.normal(0.0, 0.008, size=NUM_JOINTS)
                hand_t[:, 1] += finger_wiggle
                seq[t] = hand_t + np.array([wrist_x0, wrist_y0 + 0.10, wrist_z0], dtype=np.float32)
        else:
            # Drift lambat non-perintah
            drift_x = rng.uniform(-0.08, 0.08)
            drift_y = rng.uniform(-0.06, 0.06)
            for t in range(T):
                dx = drift_x * t_norm[t]
                dy = drift_y * t_norm[t]
                seq[t] = base_hand + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    # Tambahkan jitter sensor kamera realistis MediaPipe
    sensor_noise = rng.normal(0.0, 0.002, size=seq.shape).astype(np.float32)
    return seq + sensor_noise


def build_realistic_dataset(cfg: dict, samples_per_class: int = 60) -> None:
    """Generate kinematic dataset, simpan ke data/interim_landmarks, lalu bangun T=32 splits."""
    interim = resolve_path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)

    names = cfg["classes"]["names"]
    rng = np.random.default_rng(cfg["project"].get("seed", 42))

    print(f"\n[Kinematic Generator] Membangun {samples_per_class} klip/kelas ({samples_per_class * len(names)} total)...")

    rows = []
    for class_id, class_name in enumerate(names):
        print(f"  • Membangun Kelas {class_id}: '{class_name}'...")
        for s in range(samples_per_class):
            vid = f"kinematic_{class_id}_{s:03d}"
            seq = generate_kinematic_clip(class_id=class_id, T=32, rng=rng)
            out_file = interim / f"{vid}.npy"
            np.save(out_file, seq)
            rows.append({
                "video_id": vid,
                "label": class_id,
                "jester_label": class_name,
                "n_frames": 32,
                "missing_ratio": 0.0,
                "status": "ok",
            })

    index_df = pd.DataFrame(rows)
    index_file = interim / "index.csv"
    index_df.to_csv(index_file, index=False)
    print(f"[Kinematic Generator] Berhasil menulis {len(rows)} entri ke {index_file}")

    print("\n[Splits] Merekonstruksi dataset splits & z-score scaler (T=32)...")
    build_splits(cfg, T=32)
    print("[OK] Dataset siap untuk dilatih!")


def main():
    ap = argparse.ArgumentParser(description="Generate realistic kinematic gesture dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--samples", type=int, default=60, help="Jumlah sampel per kelas (default: 60)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    build_realistic_dataset(cfg, samples_per_class=args.samples)


if __name__ == "__main__":
    main()
