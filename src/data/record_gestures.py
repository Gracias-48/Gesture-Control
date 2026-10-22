"""Interactive webcam gesture recorder tool (TRD §2.2).

Merekam 32-frame sekuens landmark MediaPipe Hands per repetisi langsung dari webcam
untuk 8 kelas target (Swipe Left, Swipe Right, Scroll Up, Scroll Down, Zoom In, Zoom Out, Stop, Background).

Fitur:
- Countdown visual (3.. 2.. 1.. REKAM) pada window OpenCV
- Ekstraksi 21 koordinat 3D MediaPipe real-time pada 12 FPS
- Penyimpanan otomatis ke data/interim_landmarks/<video_id>.npy & index.csv
- Mode rekam per kelas atau seluruh 8 kelas secara berurutan
- Opsi otomatis rebuild train/val/test splits (T=32) & scaler_stats.npz

Penggunaan:
    python -m src.data.record_gestures --reps 25
    python -m src.data.record_gestures --class-id 0 --reps 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.preprocessing import (
    NUM_JOINTS,
    build_splits,
    interpolate_missing,
    to_centered_coords,
)
from src.inference.hand_tracker import HandTracker, draw_skeleton
from src.utils.config import load_config, resolve_path


def draw_hud_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: tuple[int, int, int] = (10, 10, 10),
) -> None:
    """Gambar teks dengan shadow/background kontras tinggi."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, bg_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def record_clip(
    cap: cv2.VideoCapture,
    tracker: HandTracker,
    target_frames: int = 32,
    target_fps: float = 12.0,
    prompt_text: str = "",
) -> tuple[np.ndarray | None, float]:
    """Rekam tepat `target_frames` pada `target_fps` (dt ≈ 83.3 ms)."""
    frame_period = 1.0 / target_fps
    collected = []
    t_next = time.perf_counter()

    while len(collected) < target_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.perf_counter()
        display = cv2.flip(frame, 1)

        # Progress bar perekaman
        progress = len(collected) / target_frames
        bar_w = int(display.shape[1] * progress)
        cv2.rectangle(display, (0, 0), (bar_w, 8), (50, 200, 50), -1)

        draw_hud_text(display, f"RECORDING: {len(collected) + 1}/{target_frames}", (20, 40), scale=0.8, color=(0, 255, 100))
        draw_hud_text(display, prompt_text, (20, 80), scale=0.6, color=(255, 255, 255))
        draw_hud_text(display, "Tekan 'q' untuk batal", (20, display.shape[0] - 20), scale=0.5, color=(160, 160, 160))

        cv2.imshow("Gesture Data Recorder (COMP6826001)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, 1.0

        if now >= t_next:
            t_next = now + frame_period
            centered, norm = tracker.process(frame)
            if centered is not None:
                collected.append(centered)
            else:
                collected.append(np.full((NUM_JOINTS, 3), np.nan, dtype=np.float32))

    raw_arr = np.array(collected, dtype=np.float32)
    filled, missing_ratio = interpolate_missing(raw_arr)
    return filled, missing_ratio


def run_countdown(cap: cv2.VideoCapture, seconds: int = 3, message: str = "") -> bool:
    """Hitung mundur sebelum perekaman agar user siap melakukan gerakan."""
    t_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t_start
        remaining = seconds - int(elapsed)
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        # Overlay gelap transparan
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

        draw_hud_text(display, message, (40, 70), scale=0.8, color=(255, 215, 0), thickness=2)
        draw_hud_text(display, f"Bersiap dalam: {remaining}", (w // 2 - 120, h // 2), scale=1.4, color=(0, 255, 255), thickness=3)
        draw_hud_text(display, "Posisikan tangan di depan kamera", (w // 2 - 180, h // 2 + 60), scale=0.6, color=(200, 200, 200))

        cv2.imshow("Gesture Data Recorder (COMP6826001)", display)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            return False
    return True


def record_sessions(cfg: dict, class_ids: list[int], reps_per_class: int = 25) -> None:
    """Jalankan sesi perekaman interaktif untuk kelas yang dipilih."""
    import pandas as pd

    interim = resolve_path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    index_path = interim / "index.csv"

    index_df = pd.read_csv(index_path, dtype={"video_id": str}) if index_path.exists() else pd.DataFrame()

    classes = cfg["classes"]["names"]
    cam_idx = int(cfg["inference"].get("camera_index", 0))

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_idx, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[Error] Tidak dapat membuka webcam {cam_idx}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    instructions = {
        0: "Gerakkan tangan meluncur ke KIRI secara konsisten",
        1: "Gerakkan tangan meluncur ke KANAN secara konsisten",
        2: "Geser DUA JARI ke ATAS (Scroll Up)",
        3: "Geser DUA JARI ke BAWAH (Scroll Down)",
        4: "Buka/lebarkan jari tangan ke arah luar (Zoom In)",
        5: "Kuncupkan/rapatkan jari tangan ke tengah (Zoom Out)",
        6: "Buka telapak tangan tegak menghadap kamera dan tahan diam (Stop Sign)",
        7: "Tangan santai, diam, garuk kepala, berbicara (Background / No Gesture)",
    }

    print("\n" + "=" * 65)
    print("🎯 TOUCHLESS GESTURE CONTROL — REAL WEBCAM RECORDER TOOL")
    print("=" * 65)
    print("Kontrol:")
    print(" - 'Space' : Mulai rekam repetisi berikutnya")
    print(" - 'q'     : Keluar / selesai sesi")
    print("=" * 65 + "\n")

    new_rows = []

    try:
        for cid in class_ids:
            cname = classes[cid]
            cinst = instructions.get(cid, "")
            print(f"\n▶ Merekam Kelas [{cid}]: '{cname}' ({reps_per_class} repetisi)")
            print(f"  Instruksi: {cinst}")

            rep = 0
            while rep < reps_per_class:
                prompt = f"[{cid}] {cname} (Rep {rep + 1}/{reps_per_class}): Tekan [SPACE] untuk mulai"

                # Wait for user SPACE trigger
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    disp = cv2.flip(frame, 1)
                    draw_hud_text(disp, f"Kelas: {cname} (ID: {cid})", (20, 40), scale=0.8, color=(255, 200, 50))
                    draw_hud_text(disp, f"Repetisi: {rep + 1} dari {reps_per_class}", (20, 75), scale=0.6, color=(220, 220, 220))
                    draw_hud_text(disp, cinst, (20, 110), scale=0.6, color=(0, 255, 200))
                    draw_hud_text(disp, "Tekan [SPACE] untuk countdown 3 detik | [Q] Keluar", (20, disp.shape[0] - 25), scale=0.55, color=(160, 160, 160))

                    cv2.imshow("Gesture Data Recorder (COMP6826001)", disp)
                    k = cv2.waitKey(25) & 0xFF
                    if k == ord(" "):
                        waiting = False
                    elif k == ord("q"):
                        print("[Recorder] Dibatalkan oleh pengguna.")
                        return

                # Countdown
                ok = run_countdown(cap, seconds=3, message=f"Siap-siap: {cname.upper()}")
                if not ok:
                    return

                # Record 32 frames
                clip_id = f"real_{cid}_{int(time.time())}_{rep:03d}"
                seq, missing_ratio = record_clip(
                    cap, tracker, target_frames=32, target_fps=12.0, prompt_text=f"LAKUKAN: {cname.upper()}"
                )

                if seq is None:
                    print("  Repetisi dibatalkan.")
                    continue

                if missing_ratio > 0.35:
                    print(f"  [Peringatan] Missing hand frame terlalu banyak ({missing_ratio*100:.1f}%). Ulangi repetisi {rep + 1}.")
                    continue

                out_file = interim / f"{clip_id}.npy"
                np.save(out_file, seq)
                new_rows.append({
                    "video_id": clip_id,
                    "label": cid,
                    "jester_label": cname,
                    "n_frames": 32,
                    "missing_ratio": round(missing_ratio, 3),
                    "status": "ok",
                })
                print(f"  [OK] Repetisi {rep + 1}/{reps_per_class} tersimpan (Missing: {missing_ratio*100:.1f}%)")
                rep += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

        if new_rows:
            all_df = pd.concat([index_df, pd.DataFrame(new_rows)], ignore_index=True) if not index_df.empty else pd.DataFrame(new_rows)
            all_df.drop_duplicates("video_id", keep="last", inplace=True)
            all_df.to_csv(index_path, index=False)
            print(f"\n[Index] Menyimpan {len(all_df)} total rekaman ke {index_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Real Webcam Gesture Recorder (32-frame continuous)")
    ap.add_argument("--config", default=None, help="Path ke config.yaml")
    ap.add_argument("--class-id", type=int, default=None, help="ID kelas tunggal (0-7) untuk direkam")
    ap.add_argument("--reps", type=int, default=25, help="Jumlah repetisi per kelas (default: 25)")
    ap.add_argument("--build-splits", action="store_true", help="Otomatis rebuild train/val/test splits & scaler setelah rekam")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_classes = [args.class_id] if args.class_id is not None else list(range(len(cfg["classes"]["names"])))

    record_sessions(cfg, target_classes, reps_per_class=args.reps)

    if args.build_splits:
        print("\n[Splits] Merekonstruksi data/processed_landmarks/T32...")
        build_splits(cfg, T=32)
        print("[Splits] Selesai!")


if __name__ == "__main__":
    main()
