# Project Implementation Plan (`IMPLEMENTATION_PLAN.md`)
**Course:** COMP6826001 – Deep Learning (Final Project)  
**Final Deadline:** Week 14, Sunday, 23:59 WIB  
**Deliverables:** Functional Application (40%), Academic Report 10–15 pages (40%), Demo Video < 5 mins (20%)

---

## 1. Repository & Directory Structure

```text
touchless-gesture-control/
├── configs/
│   └── experiments_e1_e12.yaml       # Konfigurasi 12 skenario hyperparameter (E1-E12)
├── data/
│   ├── raw_jester_subset/            # Subset 8 kelas dataset 20BN-Jester
│   ├── processed_landmarks/          # File hasil ekstraksi .npy (train/val/test)
│   └── scaler_stats.npz              # Statistik mean & std z-score dari train split
├── src/
│   ├── preprocessing/
│   │   ├── extract_mediapipe.py      # Offline upscale 256px + MediaPipe 21 keypoints
│   │   ├── interpolate_clean.py      # Interpolasi linear & filter >30% missing frames
│   │   ├── dual_stream_norm.py       # Normalisasi lokal + global + kecepatan sendi
│   │   └── augmentations.py          # Rotasi, skala, jitter, speed, flip Left<->Right
│   ├── models/
│   │   ├── bilstm_baseline.py        # Model 1: 2-Layer BiLSTM + Attention Pooling
│   │   ├── st_gcn.py                 # Model 2: 21-Node Hand Graph ST-GCN (4 blocks)
│   │   └── causal_tcn.py             # Model 3: Causal Dilated TCN (dilation 1,2,4,8)
│   ├── training/
│   │   ├── train_runner.py           # Loop pelatihan multi-seed (3 seeds), early stopping
│   │   ├── ablation_runner.py        # Eksekusi 4 studi ablasi wajib
│   │   └── evaluate_metrics.py       # Macro F1, Precision/Recall, Confusion Matrix
│   └── deployment/
│       ├── export_onnx.py            # Konversi PyTorch -> ONNX (< 5 MB) + verifikasi
│       ├── realtime_engine.py        # Thread kamera 12 FPS, ring buffer 32, ONNX session
│       ├── postprocess_gate.py       # Motion gate, threshold 0.80, 3-window, cooldown
│       └── os_controller.py          # Integrasi PyAutoGUI lintas platform
├── app/
│   ├── streamlit_app.py              # Main Raycast HUD Dashboard + User Study Mode
│   └── assets/
│       └── raycast_theme.css         # Token styling dari design-md/raycast/DESIGN.md
├── notebooks/
│   └── visualization_analysis.ipynb  # Plot kurva training, tabel E1-E12, analisis error
├── requirements.txt
└── README.md
```

---

## 2. Phased Execution Roadmap (Weeks 6 – 14)

### Phase 1: Data Pipeline & Offline Landmark Extraction (Week 6 – Week 7)
- [ ] Mengunduh subset 8 kelas dari 20BN-Jester (~40.000 klip) dan melakukan *stratified split* (70/15/15, `seed=42`).
- [ ] Membangun skrip `extract_mediapipe.py` dengan pemrosesan paralel (`multiprocessing`) untuk melakukan upscale ke $256\text{ px}$ dan mengekstrak 21 titik $(x, y, z)$.
- [ ] Mengimplementasikan interpolasi linear untuk frame kosong, membuang klip dengan $>30\%$ frame gagal deteksi, dan menyeragamkan panjang sekuens ke $T = 32$.
- [ ] Menghitung normalisasi dua aliran (lokal + global) dan menyimpan parameter standardisasi `scaler_stats.npz`.

### Phase 2: Model Implementation & Experimentation Matrix E1–E12 (Week 8 – Week 9)
- [ ] Mengimplementasikan `BiLSTM`, `ST-GCN` (dengan matriks ketetanggaan 21 sendi tangan), dan `Causal TCN` di PyTorch.
- [ ] Menguji alur *forward/backward pass* dan memverifikasi jumlah parameter masing-masing model ($\approx 0.2\text{M} - 1.0\text{M}$ parameter).
- [ ] Menjalankan 12 skenario eksperimen (**E1–E12**) masing-masing dengan 3 seed acak (`42`, `100`, `2026`) menggunakan *Cross-Entropy Loss* (`label_smoothing=0.1`) dan *gradient clipping* `1.0`.
- [ ] Menjalankan 4 eksperimen studi ablasi (tanpa aliran global, tanpa augmentasi, variasi $T \in \{16, 32, 48\}$, dan tanpa kelas `Background`).

### Phase 3: ONNX Optimization & Real-Time Engine (Week 10 – Week 11)
- [ ] Mengekspor checkpoint terbaik dari BiLSTM, ST-GCN, dan TCN ke format `.onnx` dan memvalidasi ukuran berkas ($< 5\text{ MB}$).
- [ ] Mengukur *benchmark* latensi inferensi per window pada CPU laptop (target $< 20\text{ ms}$) dan memverifikasi throughput end-to-end ($\ge 25\text{ FPS}$).
- [ ] Mengimplementasikan `postprocess_gate.py` (ambang konfidensi $0.80$, konsistensi $3/3$ window, cooldown $0.8 - 1.0\text{ detik}$, dan deteksi gerakan minimum) serta modul aksi `PyAutoGUI`.

### Phase 4: Raycast HUD Interface & User Testing (Week 12)
- [ ] Membangun antarmuka `streamlit_app.py` dengan injeksi CSS kustom (`raycast_theme.css`) menggunakan warna dasar `#07080a`, permukaan `#101111`, border `#242728`, aksen `#ff6363`, dan font monospasi untuk metrik real-time.
- [ ] Melaksanakan sesi pengujian pengguna (*within-subject*, $n = 8 - 10$ partisipan) untuk skenario tugas **T1–T5** serta mengumpulkan kuesioner **SUS**.

### Phase 5: Final Report & Demo Video Production (Week 13 – Week 14)
- [ ] Menyusun laporan akhir akademik (10–15 halaman) mengikuti struktur resmi COMP6826001 (Title Page, Abstract, Introduction, Related Work, Methodology, Implementation & Results, Discussion & Limitations, Conclusion, References, Appendix).
- [ ] Merekam dan menyunting video demonstrasi ($< 5\text{ menit}$) yang menampilkan latar belakang masalah, arsitektur spasio-temporal, eksekusi HUD bergaya Raycast secara langsung, serta hasil pengujian pengguna.

---

## 3. Team Task Allocation & Contribution Matrix

Sesuai ketentuan *Team Contribution Statement* pada pedoman Final Project:

| Nama Anggota | NIM | Tanggung Jawab Utama (Deliverables) |
| :--- | :---: | :--- |
| **Gracias Kumara Winata** | 2802459683 | **Data Engineering & ST-GCN Architecture:** Ekstraksi MediaPipe offline, pipeline normalisasi dua aliran & augmentasi, implementasi model ST-GCN (E5–E8), serta penulisan bab *Methodology* laporan. |
| **Vincensius Kevin Mulyono** | 2802476424 | **Baseline BiLSTM, TCN & Ablation Studies:** Implementasi model BiLSTM (E1–E4) dan TCN (E9–E12), eksekusi 4 studi ablasi, evaluasi statistik multi-seed, serta penulisan bab *Implementation & Results*. |
| **Malvin Raditya Nugraha** | 2802489206 | **ONNX Deployment, Raycast HUD & User Study:** Ekspor model ke ONNX Runtime, pengembangan *post-processing gate* & PyAutoGUI, antarmuka Streamlit bertema Raycast, koordinasi pengujian 10 pengguna (SUS), dan produksi video demo. |