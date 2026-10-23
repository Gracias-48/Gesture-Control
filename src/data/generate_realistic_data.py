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

FINGER_JOINTS = [
    [1, 2, 3, 4],       # Thumb
    [6, 7, 8],          # Index
    [10, 11, 12],       # Middle
    [14, 15, 16],       # Ring
    [18, 19, 20],       # Pinky
]


def generate_kinematic_clip(
    class_id: int,
    T: int = 32,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 1 clip (T, 21, 3) koordinat tangan kontinu sesuai kelas gestur."""
    if rng is None:
        rng = np.random.default_rng()

    # Variasi anatomis ukuran tangan per pengguna & jarak ke webcam
    hand_scale = rng.uniform(0.78, 1.22)
    base_hand = (CANONICAL_HAND * hand_scale).copy()

    # Variasi orientasi roll dan pitch natural tangan
    roll = rng.uniform(-0.18, 0.18)
    pitch = rng.uniform(-0.12, 0.12)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    rot_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rot_matrix = rot_pitch @ rot_roll
    base_hand = base_hand @ rot_matrix.T

    # Posisi awal pergelangan tangan (wrist)
    wrist_x0 = rng.uniform(-0.12, 0.12)
    wrist_y0 = rng.uniform(-0.06, 0.10)
    wrist_z0 = rng.uniform(-0.03, 0.03)

    seq = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

    # Kurva waktu ternormalisasi dengan sedikit variasi eksponensial (kecepatan non-linear natural)
    t_raw = np.linspace(0.0, 1.0, T, dtype=np.float32)
    warp_exp = rng.uniform(0.85, 1.18)
    t_norm = t_raw ** warp_exp
    smooth_step = 0.5 * (1.0 - np.cos(np.pi * t_norm))

    # Definisi pose terlipat/pinched dan pose terbuka/splayed dengan rigid palm MCP (0, 5, 9, 13, 17)
    pinch_target = base_hand[9] + np.array([0.0, -0.04, 0.0], dtype=np.float32)
    pinched_hand = base_hand.copy()
    for fj in FINGER_JOINTS:
        for rank, idx in enumerate(fj):
            factor = (rank + 1) / len(fj)
            pinched_hand[idx] = base_hand[idx] * (1.0 - factor * 0.82) + pinch_target * (factor * 0.82)

    splayed_hand = base_hand.copy()
    splayed_hand[4] += np.array([-0.065, 0.015, -0.01], dtype=np.float32)
    splayed_hand[8] += np.array([-0.045, -0.055, -0.01], dtype=np.float32)
    splayed_hand[12] += np.array([0.000, -0.065, 0.00], dtype=np.float32)
    splayed_hand[16] += np.array([0.045, -0.055, 0.01], dtype=np.float32)
    splayed_hand[20] += np.array([0.065, 0.015, 0.01], dtype=np.float32)
    # Intermediate finger joints follow their tips
    for fj in FINGER_JOINTS:
        tip_idx = fj[-1]
        tip_disp = splayed_hand[tip_idx] - base_hand[tip_idx]
        for rank, idx in enumerate(fj[:-1]):
            factor = (rank + 1) / len(fj)
            splayed_hand[idx] += tip_disp * factor * 0.70

    if class_id == 0:
        # --- SWIPE LEFT: Wrist meluncur dari kanan (+x) ke kiri (-x) ---
        amplitude = rng.uniform(0.38, 0.58)
        arc_depth = rng.uniform(-0.04, 0.03)
        for t in range(T):
            dx = amplitude * (0.5 - smooth_step[t])
            dy = arc_depth * math.sin(math.pi * t_norm[t])
            # Sedikit drag inersia pada ujung jari
            drag_x = 0.02 * math.sin(math.pi * t_norm[t])
            hand_t = base_hand.copy()
            hand_t[1:] += np.array([drag_x, 0.0, 0.0], dtype=np.float32)
            seq[t] = hand_t + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    elif class_id == 1:
        # --- SWIPE RIGHT: Wrist meluncur dari kiri (-x) ke kanan (+x) ---
        amplitude = rng.uniform(0.38, 0.58)
        arc_depth = rng.uniform(-0.04, 0.03)
        for t in range(T):
            dx = amplitude * (smooth_step[t] - 0.5)
            dy = arc_depth * math.sin(math.pi * t_norm[t])
            drag_x = -0.02 * math.sin(math.pi * t_norm[t])
            hand_t = base_hand.copy()
            hand_t[1:] += np.array([drag_x, 0.0, 0.0], dtype=np.float32)
            seq[t] = hand_t + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    elif class_id == 2:
        # --- SCROLL UP: Telunjuk & Jari Tengah mengayun / flick ke atas berulang kali ---
        cycle_freq = rng.uniform(1.8, 2.6)
        amp = rng.uniform(0.065, 0.095)
        for t in range(T):
            # Ayunan asimetris: flick ke atas cepat (-y), kembali rileks
            wave = -amp * abs(math.sin(math.pi * cycle_freq * t_norm[t]))
            hand_t = base_hand.copy()
            # Tekuk jari manis dan kelingking ke telapak
            hand_t[[14, 15, 16], 1] += 0.09
            hand_t[[18, 19, 20], 1] += 0.09
            # Telunjuk dan jari tengah mengayun ke atas
            hand_t[[6, 7, 8, 10, 11, 12], 1] += wave
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 3:
        # --- SCROLL DOWN: Telunjuk & Jari Tengah mengayun / flick ke bawah berulang kali ---
        cycle_freq = rng.uniform(1.8, 2.6)
        amp = rng.uniform(0.065, 0.095)
        for t in range(T):
            # Ayunan flick ke bawah (+y)
            wave = amp * abs(math.sin(math.pi * cycle_freq * t_norm[t]))
            hand_t = base_hand.copy()
            # Tekuk jari manis dan kelingking
            hand_t[[14, 15, 16], 1] += 0.09
            hand_t[[18, 19, 20], 1] += 0.09
            # Telunjuk dan jari tengah mengayun ke bawah
            hand_t[[6, 7, 8, 10, 11, 12], 1] += wave
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 4:
        # --- ZOOM IN: Tangan bermula dari pose pinched lalu mengembang melebar ke luar ---
        # Rigid palm joints (0, 5, 9, 13, 17) tetap konstan, sementara jari merekah keluar
        start_factor = rng.uniform(0.0, 0.15)
        end_factor = rng.uniform(0.85, 1.0)
        for t in range(T):
            alpha = start_factor + (end_factor - start_factor) * smooth_step[t]
            hand_t = (1.0 - alpha) * pinched_hand + alpha * splayed_hand
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 5:
        # --- ZOOM OUT: Tangan bermula dari pose merekah lalu menguncup rapat ke pusat telapak ---
        start_factor = rng.uniform(0.85, 1.0)
        end_factor = rng.uniform(0.0, 0.15)
        for t in range(T):
            alpha = start_factor + (end_factor - start_factor) * smooth_step[t]
            hand_t = (1.0 - alpha) * pinched_hand + alpha * splayed_hand
            seq[t] = hand_t + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32)

    elif class_id == 6:
        # --- STOP SIGN: Telapak tangan terbuka tegak menghadap kamera, posisi sangat stabil ---
        for t in range(T):
            tremor = rng.normal(0.0, 0.00035, size=(NUM_JOINTS, 3)).astype(np.float32)
            seq[t] = base_hand + np.array([wrist_x0, wrist_y0, wrist_z0], dtype=np.float32) + tremor

    else:
        # --- BACKGROUND (ID 7): Beragam gerak non-perintah (idle, mengetik, drift lambat, dll.) ---
        bg_mode = rng.integers(0, 5)
        if bg_mode == 0:
            # Tangan santai diam di bawah frame
            for t in range(T):
                tremor = rng.normal(0.0, 0.0004, size=(NUM_JOINTS, 3)).astype(np.float32)
                seq[t] = base_hand + np.array([wrist_x0, wrist_y0 + 0.16, wrist_z0], dtype=np.float32) + tremor
        elif bg_mode == 1:
            # Mengetik di keyboard: jari bergerak kecil secara independen & asinkron
            for t in range(T):
                hand_t = base_hand.copy()
                for fj in FINGER_JOINTS:
                    freq_f = rng.uniform(2.0, 4.0)
                    wiggle = 0.012 * math.sin(2.0 * math.pi * freq_f * t_norm[t])
                    hand_t[fj, 1] += wiggle
                seq[t] = hand_t + np.array([wrist_x0, wrist_y0 + 0.12, wrist_z0], dtype=np.float32)
        elif bg_mode == 2:
            # Drift lambat tidak tegas (kecepatan sangat rendah, bukan swipe)
            drift_x = rng.uniform(-0.06, 0.06)
            drift_y = rng.uniform(-0.05, 0.05)
            for t in range(T):
                dx = drift_x * t_norm[t]
                dy = drift_y * t_norm[t]
                seq[t] = base_hand + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)
        elif bg_mode == 3:
            # Tangan mengepal / santai sebagian
            curled_hand = base_hand.copy()
            for fj in FINGER_JOINTS:
                curled_hand[fj, 1] += 0.08
            for t in range(T):
                tremor = rng.normal(0.0, 0.0004, size=(NUM_JOINTS, 3)).astype(np.float32)
                seq[t] = curled_hand + np.array([wrist_x0, wrist_y0 + 0.05, wrist_z0], dtype=np.float32) + tremor
        else:
            # Tangan bergerak acak menuju/keluar kamera
            for t in range(T):
                dx = 0.03 * math.sin(1.5 * math.pi * t_norm[t])
                dy = 0.03 * math.cos(1.5 * math.pi * t_norm[t])
                seq[t] = base_hand + np.array([wrist_x0 + dx, wrist_y0 + dy, wrist_z0], dtype=np.float32)

    # Tambahkan jitter sensor MediaPipe real-time (~0.0004 image-height units)
    sensor_noise = rng.normal(0.0, 0.0004, size=seq.shape).astype(np.float32)
    return seq + sensor_noise


def build_realistic_dataset(cfg: dict, samples_per_class: int = 350) -> None:
    """Generate kinematic dataset, simpan ke data/interim_landmarks, lalu bangun T=32 splits."""
    interim = resolve_path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)

    names = cfg["classes"]["names"]
    rng = np.random.default_rng(cfg["project"].get("seed", 42))

    total_samples = samples_per_class * len(names)
    print(f"\n[Kinematic Generator] Membangun {samples_per_class} klip/kelas ({total_samples} total)...")

    rows = []
    for class_id, class_name in enumerate(names):
        print(f"  • Membangun Kelas {class_id}: '{class_name}' ({samples_per_class} klip)...")
        for s in range(samples_per_class):
            vid = f"kinematic_{class_id}_{s:04d}"
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
    ap.add_argument("--samples", type=int, default=350, help="Jumlah sampel per kelas (default: 350)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    build_realistic_dataset(cfg, samples_per_class=args.samples)


if __name__ == "__main__":
    main()
