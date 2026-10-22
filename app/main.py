"""Streamlit Real-Time Gesture Control HUD & Evaluation Dashboard (Raycast Theme).

COMP6826001 - Deep Learning Final Project
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Pastikan root workspace ada di sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import importlib
import app.styles
importlib.reload(app.styles)

from app.styles import (
    RAYCAST_CSS,
    render_confidence_bars,
    render_state_badge,
    render_telemetry_cards,
)
from src.inference.engine import GestureInferenceEngine, TelemetryData
from src.utils.config import load_config, resolve_path

# Setup Halaman Streamlit
st.set_page_config(
    page_title="Touchless Gesture Control HUD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeksi CSS Raycast Design System
st.markdown(RAYCAST_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_engine() -> GestureInferenceEngine:
    """Singleton instance mesin inferensi di dalam memori Streamlit."""
    cfg = load_config()
    eng = GestureInferenceEngine(cfg)
    return eng


def render_sidebar(engine: GestureInferenceEngine, cfg: Dict[str, Any]):
    """Sidebar interaktif untuk konfigurasi model, gate, dan aksi OS."""
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0; border-bottom: 1px solid #242728; margin-bottom: 16px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f9f9f9;">RAYCAST HUD</div>
            <div style="font-size: 0.75rem; color: #9c9c9d;">Touchless Gesture Control • COMP6826001</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="hud-section-title">Model Hot-Swapping</div>', unsafe_allow_html=True)
    models = ["TCN", "BiLSTM", "ST-GCN"]
    cur_model = engine.active_model_name.upper()
    cur_idx = 0
    if cur_model in ("TCN", "CAUSAL_TCN"):
        cur_idx = 0
    elif cur_model in ("BILSTM", "BILSTM_BASELINE"):
        cur_idx = 1
    elif cur_model in ("STGCN", "ST-GCN"):
        cur_idx = 2

    chosen_model = st.sidebar.selectbox("Active Inference Architecture", models, index=cur_idx)
    target_key = {"TCN": "tcn", "BiLSTM": "bilstm", "ST-GCN": "stgcn"}[chosen_model]
    if target_key != engine.active_model_name:
        success = engine.load_model(target_key)
        if success:
            st.sidebar.success(f"Loaded {chosen_model} ONNX Session")
        else:
            st.sidebar.warning(f"ONNX model '{target_key}' not found. Run export first.")

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="hud-section-title">Anti-False-Trigger Gate</div>', unsafe_allow_html=True)
    
    thresh = st.sidebar.slider(
        "Confidence Threshold (min probability)",
        min_value=0.50,
        max_value=0.98,
        value=float(engine.telemetry.confidence_threshold),
        step=0.02,
        help="Probabilitas softmax minimum agar gestur dapat dieksekusi",
    )
    if abs(thresh - engine.telemetry.confidence_threshold) > 0.005:
        engine.set_confidence_threshold(thresh)

    cooldown = st.sidebar.slider(
        "Cooldown Window (seconds)",
        min_value=0.4,
        max_value=2.0,
        value=float(engine.telemetry.cooldown_target),
        step=0.1,
        help="Jeda waktu penguncian pemicu setelah aksi sukses",
    )
    if abs(cooldown - engine.telemetry.cooldown_target) > 0.05:
        engine.set_cooldown(cooldown)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="hud-section-title">OS Action Injection (PyAutoGUI)</div>', unsafe_allow_html=True)
    
    arm_toggle = st.sidebar.checkbox(
        "Master Arm OS Control",
        value=engine.telemetry.os_armed,
        help="Bila dicentang, perintah keyboard/scroll dikirim langsung ke OS via PyAutoGUI. Gerakkan kursor ke sudut layar bila ingin abort darurat.",
    )
    if arm_toggle != engine.telemetry.os_armed:
        engine.set_os_control(arm_toggle)

    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; color: #6a6b6c; margin-top: 10px; line-height: 1.4;">
            <b>Emergency Fail-Safe:</b> Dorong kursor mouse ke salah satu sudut layar (corner) kapan saja untuk memutus kontrol otomatis secara instan.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_hud(engine: GestureInferenceEngine, cfg: Dict[str, Any]):
    """Tab 1: Live Webcam Viewport, Skeletal Overlay, dan Real-time Telemetry HUD."""
    # Top Header Command Bar
    col_hdr1, col_hdr2 = st.columns([0.65, 0.35])
    with col_hdr1:
        st.markdown(
            f"""
            <div class="ray-header">
                <div class="ray-title">
                    <span class="dot"></span>
                    <span>TOUCHLESS GESTURE HUD</span>
                    <span style="font-size: 0.75rem; color: #9c9c9d; margin-left: 8px;">[Arch: {engine.telemetry.active_model}]</span>
                </div>
                <div id="state-badge-container">
                    {render_state_badge(engine.telemetry.state)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_hdr2:
        armed_status = "ARMED (⌘)" if engine.telemetry.os_armed else "DISARMED"
        st.markdown(
            f"""
            <div class="ray-header" style="justify-content: flex-end; gap: 12px;">
                <span style="font-size: 0.8rem; color: #9c9c9d;">OS Injection:</span>
                <span class="ray-badge {'badge-triggered' if engine.telemetry.os_armed else 'badge-standby'}">{armed_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Grid (Left 65% Viewport, Right 35% Telemetry)
    col_view, col_tele = st.columns([0.62, 0.38])

    with col_view:
        # Kontrol Kamera Play/Pause
        cam_col1, cam_col2 = st.columns([0.5, 0.5])
        with cam_col1:
            if not engine.running:
                if st.button("▶ Start Camera Stream", use_container_width=True):
                    engine.start()
                    st.rerun()
            else:
                if st.button("⏹ Stop Camera Stream", use_container_width=True):
                    engine.stop()
                    st.rerun()
        with cam_col2:
            if st.button("🔄 Reset Buffer & Gate", use_container_width=True):
                if engine.gate:
                    engine.gate.reset()
                engine.raw_landmark_buffer.clear()
                st.rerun()

        # Frame container
        frame_placeholder = st.empty()
        toast_placeholder = st.empty()

        # Riwayat Log Aksi
        st.markdown('<div class="hud-section-title" style="margin-top: 14px;">Recent Command Triggers</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()

    with col_tele:
        telemetry_placeholder = st.empty()
        conf_placeholder = st.empty()

    # Loop refresh telemetri & frame
    if engine.running:
        for _ in range(30):  # Stream 30 update cycles per render
            frame, tele = engine.get_latest_frame_and_telemetry()
            if frame is not None and getattr(frame, "size", 0) > 0:
                ret, jpg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_placeholder.image(jpg_buf.tobytes(), use_container_width=True)
                else:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                frame_placeholder.markdown(
                    """
                    <div style="background: #101111; border: 1px solid #242728; border-radius: 8px; height: 360px; display: flex; align-items: center; justify-content: center; color: #6a6b6c;">
                        Menunggu frame kamera... Pastikan webcam terhubung dan izin kamera aktif.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Toast Pill
            if tele.toast_message:
                toast_placeholder.markdown(
                    f"""
                    <div style="text-align: center; margin-top: -60px; margin-bottom: 20px; position: relative; z-index: 10;">
                        <div class="toast-pill">
                            <span>⚡</span>
                            <span>{tele.toast_message}</span>
                            <span class="shortcut">Triggered</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                toast_placeholder.empty()

            # Telemetry Cards & Confidence Bars
            telemetry_placeholder.markdown(
                render_telemetry_cards(
                    fps=tele.fps,
                    latency_ms=tele.inference_latency_ms,
                    buffer_fill=tele.buffer_fill,
                    buffer_max=tele.buffer_max,
                    cooldown_s=tele.cooldown_remaining,
                    os_armed=tele.os_armed,
                ),
                unsafe_allow_html=True,
            )

            conf_placeholder.markdown(
                render_confidence_bars(
                    class_names=tele.class_names,
                    probs=tele.probabilities,
                    threshold=tele.confidence_threshold,
                ),
                unsafe_allow_html=True,
            )

            # Action Log
            if tele.action_history:
                log_html = ['<div class="action-log-card">']
                for item in tele.action_history[:5]:
                    log_html.append(f"""
                    <div class="action-log-item">
                        <span class="action-log-ts">{item['timestamp']}</span>
                        <span class="action-log-gesture">{item['gesture']}</span>
                        <span class="action-log-key">{item['action']} ({item['shortcut']})</span>
                        <span style="color: #56c2ff;">{item['confidence']}</span>
                        <span style="color: #6a6b6c;">{item['latency_ms']}</span>
                    </div>
                    """)
                log_html.append('</div>')
                log_placeholder.markdown("".join(log_html), unsafe_allow_html=True)
            else:
                log_placeholder.markdown(
                    '<div class="action-log-card" style="color: #6a6b6c; font-size: 0.8rem; text-align: center; padding: 20px;">Belum ada aksi yang dipicu</div>',
                    unsafe_allow_html=True,
                )

            time.sleep(0.06)
        st.rerun()
    else:
        # Tampilan static saat camera OFF
        frame_placeholder.markdown(
            """
            <div style="background: #101111; border: 1px solid #242728; border-radius: 8px; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #9c9c9d; gap: 10px;">
                <div style="font-size: 1.5rem; color: #ff6363;">⚡</div>
                <div style="font-weight: 600; color: #f9f9f9;">Camera Inactive</div>
                <div style="font-size: 0.8rem; color: #6a6b6c;">Klik "Start Camera Stream" di atas untuk memulai inferensi real-time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tele = engine.telemetry
        telemetry_placeholder.markdown(
            render_telemetry_cards(0.0, 0.0, 0, 32, 0.0, tele.os_armed),
            unsafe_allow_html=True,
        )
        conf_placeholder.markdown(
            render_confidence_bars(tele.class_names, np.zeros(len(tele.class_names)), tele.confidence_threshold),
            unsafe_allow_html=True,
        )


def render_experiments_tab(cfg: Dict[str, Any]):
    """Tab 2: Visualisasi Skenario Eksperimen E1-E12, Hasil Training Multi-Seed, dan Studi Ablasi."""
    st.markdown('<div class="hud-section-title">E1–E12 Hyperparameter Experiment Matrix & Benchmark</div>', unsafe_allow_html=True)

    results_dir = resolve_path(cfg, "results_dir")
    summary_path = results_dir / "experiments_summary.csv"

    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        
        # Ringkasan KPI Model
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            best_f1 = df_summary["test_f1_mean"].max()
            best_row = df_summary.loc[df_summary["test_f1_mean"].idxmax()]
            st.metric("Top Test Macro F1", f"{best_f1:.4f}", f"{best_row['run_id']} ({best_row['model'].upper()})")
        with c2:
            min_lat = df_summary["latency_ms_mean"].min()
            lat_row = df_summary.loc[df_summary["latency_ms_mean"].idxmin()]
            st.metric("Fastest CPU Latency", f"{min_lat:.2f} ms", f"{lat_row['model'].upper()} ({lat_row['run_id']})")
        with c3:
            total_runs = len(df_summary)
            st.metric("Completed Scenarios", f"{total_runs}", "Multi-seed Mean ± SD")
        with c4:
            ctrl_p = df_summary["control_precision_mean"].max()
            st.metric("Peak Control Precision", f"{ctrl_p*100:.1f}%", "False-trigger defense")

        st.markdown("---")
        st.markdown("### Experiment Table (E1–E12 & Ablations)")
        st.dataframe(
            df_summary[[
                "run_id", "model", "ablation", "optimizer", "lr", "batch_size", "epochs",
                "val_f1_mean", "test_f1_mean", "test_f1_std", "test_acc_mean", "latency_ms_mean", "params"
            ]].style.format({
                "lr": "{:.2e}",
                "val_f1_mean": "{:.4f}",
                "test_f1_mean": "{:.4f}",
                "test_f1_std": "{:.4f}",
                "test_acc_mean": "{:.4f}",
                "latency_ms_mean": "{:.2f} ms",
                "params": "{:,}",
            }),
            use_container_width=True,
        )

        # Plot Perbandingan Metrik
        st.markdown("### Architecture Comparison: Test Macro F1 vs CPU Latency")
        chart_data = df_summary[["model", "run_id", "test_f1_mean", "latency_ms_mean", "params"]].copy()
        st.scatter_chart(
            chart_data,
            x="latency_ms_mean",
            y="test_f1_mean",
            color="model",
            size="params",
        )
    else:
        st.info(
            "Belum ada data eksekusi eksperimen di `results/experiments_summary.csv`. "
            "Jalankan skrip `python -m src.training.train --all` atau `--exp E1 E5 E9` untuk melatih dan mengevaluasi model."
        )

        # Tampilkan matriks skenario dari config
        st.markdown("### Target 12-Scenario Hyperparameter Matrix (E1–E12)")
        exp_matrix = []
        for eid, exp in cfg.get("experiments", {}).items():
            exp_matrix.append({
                "Scenario": eid,
                "Model": exp["model"].upper(),
                "Optimizer": exp["optimizer"],
                "LR": exp["lr"],
                "Batch Size": exp["batch_size"],
                "Max Epochs": exp["epochs"],
                "Dropout": exp["dropout"],
                "Weight Decay": exp["weight_decay"],
                "Scheduler": exp["scheduler"]["name"],
            })
        st.table(pd.DataFrame(exp_matrix))


def render_user_study_tab(cfg: Dict[str, Any]):
    """Tab 3: Protokol Uji Pengguna (T1-T5) dan Kalkulator Kuesioner SUS Digital (0-100)."""
    st.markdown('<div class="hud-section-title">Within-Subject User Study Protocol (n = 8–10)</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        Pengujian pengguna dilakukan menggunakan protokol *within-subject* untuk mengevaluasi akurasi gestur alami, 
        waktu respons, dan *false trigger rate* saat kondisi mengetik/istirahat (*idle*).
        """
    )

    t1, t2 = st.tabs(["📋 Task Protocol (T1–T5)", "📊 Digital SUS Questionnaire"])

    with t1:
        st.markdown("#### Skenario Pengujian Terstruktur")
        tasks = cfg.get("user_study", {}).get("tasks", {})
        for task_id, tdata in tasks.items():
            with st.expander(f"**{task_id}**: {tdata['title']} (Target: {tdata['instruction']})"):
                st.write(f"- **Repetisi:** {tdata['repetitions']} kali")
                st.write(f"- **Target Kelas ID:** {tdata['targets']}")
                st.write(f"- **Instruksi Penguji:** {tdata['instruction']}")

    with t2:
        st.markdown("#### System Usability Scale (SUS) Digital Form")
        st.markdown("Isi lembar kuesioner standar SUS (Skala Likert 1 = Sangat Tidak Setuju, 5 = Sangat Setuju).")

        participant_id = st.text_input("Participant ID (e.g., P01, P02)", "P01")
        hand_dom = st.selectbox("Tangan Dominan", ["Kanan", "Kiri", "Keduanya (Ambidextrous)"])

        sus_questions = [
            "1. Saya berpikir akan sering menggunakan sistem touchless control ini.",
            "2. Saya merasa sistem ini terlalu rumit untuk digunakan.",
            "3. Saya merasa sistem ini mudah digunakan.",
            "4. Saya merasa membutuhkan bantuan teknisi untuk dapat menggunakan sistem ini.",
            "5. Saya merasa berbagai fungsi pada sistem ini terintegrasi dengan sangat baik.",
            "6. Saya merasa ada terlalu banyak inkonsistensi pada sistem ini.",
            "7. Saya membayangkan kebanyakan orang akan dapat mempelajari sistem ini dengan sangat cepat.",
            "8. Saya merasa sistem ini sangat janggal atau membingungkan saat dioperasikan.",
            "9. Saya merasa sangat percaya diri saat mengoperasikan sistem ini.",
            "10. Saya perlu mempelajari banyak hal baru sebelum dapat mahir menggunakan sistem ini.",
        ]

        scores = []
        for i, q in enumerate(sus_questions):
            val = st.slider(q, min_value=1, max_value=5, value=3, key=f"sus_q_{i}")
            scores.append(val)

        # Hitung skor SUS resmi (0 - 100)
        # Untuk pertanyaan ganjil (1,3,5,7,9): skor = nilai - 1
        # Untuk pertanyaan genap (2,4,6,8,10): skor = 5 - nilai
        # Total sum dikalikan 2.5
        odd_sum = sum(scores[i] - 1 for i in range(0, 10, 2))
        even_sum = sum(5 - scores[i] for i in range(1, 10, 2))
        final_sus = (odd_sum + even_sum) * 2.5

        st.markdown("---")
        score_color = "#30d158" if final_sus >= 68.0 else "#ffd60a"
        st.markdown(
            f"""
            <div style="background: #101111; border: 1px solid #242728; border-radius: 8px; padding: 18px; text-align: center;">
                <div style="font-size: 0.8rem; color: #9c9c9d; text-transform: uppercase;">Final Calculated SUS Score</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {score_color}; font-family: 'JetBrains Mono', monospace;">
                    {final_sus:.1f} / 100
                </div>
                <div style="font-size: 0.8rem; color: #6a6b6c; margin-top: 4px;">
                    {'Di atas rata-rata industri (Above Average ≥ 68)' if final_sus >= 68 else 'Di bawah ambang rata-rata (< 68)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("💾 Simpan Hasil Kuesioner SUS ke CSV"):
            study_dir = resolve_path(cfg, "user_study_dir")
            study_dir.mkdir(parents=True, exist_ok=True)
            csv_file = study_dir / "sus_responses.csv"
            
            row = [participant_id, hand_dom, time.strftime("%Y-%m-%d %H:%M:%S"), final_sus] + scores
            headers = ["participant_id", "hand_dominant", "timestamp", "final_sus"] + [f"q{i+1}" for i in range(10)]
            
            write_header = not csv_file.exists()
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)
            st.success(f"Data partisipan {participant_id} berhasil disimpan ke {csv_file}")


def main():
    cfg = load_config()
    engine = get_engine()

    render_sidebar(engine, cfg)

    tab1, tab2, tab3 = st.tabs([
        "⚡ Live Controller HUD",
        "🔬 Experiments E1–E12 & Ablation",
        "👥 User Study & SUS Testing",
    ])

    with tab1:
        render_live_hud(engine, cfg)
    with tab2:
        render_experiments_tab(cfg)
    with tab3:
        render_user_study_tab(cfg)


if __name__ == "__main__":
    main()
