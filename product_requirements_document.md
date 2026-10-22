# Product Requirements Document (PRD)
**Project Name:** Touchless Gesture Control HUD (Spatio-Temporal Deep Learning)  
**Course:** COMP6826001 – Deep Learning (Final Project)  
**Design System Reference:** Raycast (`DESIGN.md` — Dark UI, Command Utility, Precision Monospace Telemetry)  
**Team Members:**  
- Gracias Kumara Winata (2802459683)  
- Vincensius Kevin Mulyono (2802476424)  
- Malvin Raditya Nugraha (2802489206)  

---

## 1. Executive Summary & Product Vision
**Touchless Gesture Control** adalah utilitas kontrol antarmuka komputer tanpa sentuh (*touchless HCI*) real-time berbasis webcam standar. Produk ini dirancang untuk menjawab masalah kontaminasi silang pada layar sentuh di lingkungan steril (seperti ruang operasi medis) dan keausan perangkat di fasilitas publik (kios informasi, ATM, panel lift).

Mengambil inspirasi dari filosofi desain **Raycast**, aplikasi ini tampil sebagai *desktop utility & telemetry HUD* bertema gelap (*inky dark canvas*) yang ringan, responsif, bebas distraksi, dan berjalan mulus pada CPU laptop standar tanpa memerlukan GPU diskrit.

---

## 2. Target Users & Use Cases
1. **Tenaga Medis di Lingkungan Steril:** Menavigasi dokumen medis, menggulir rekam medis, dan memperbesar citra radiologi/slide tanpa menyentuh mouse atau layar.
2. **Pengguna Presentasi & Kios Publik:** Berpindah halaman presentasi (*slide*), menggulir layar, atau menjeda media dari jarak dekat menggunakan gerakan tangan alami.
3. **Evaluator Akademik & Peneliti Deep Learning:** Membandingkan performa inferensi real-time antara tiga arsitektur model sekuensial/graf (**BiLSTM**, **ST-GCN**, dan **TCN**) secara langsung.

---

## 3. Core Functional Requirements (FR)

### FR-01: Real-Time Hand Landmark Tracking & Visualization
- Sistem wajib menangkap aliran video dari webcam pengguna pada ~30 FPS dan melakukan *timestamp-based resampling* ke **12 FPS** agar selaras dengan distribusi temporal dataset 20BN-Jester.
- Sistem mengekstraksi **21 titik koordinat 3D** $(x, y, z)$ menggunakan **MediaPipe Hands** dan menggambar kerangka sendi (*skeletal overlay*) di atas umpan video dengan kontras tinggi.

### FR-02: Continuous Spatio-Temporal Gesture Classification (8 Classes)
Sistem wajib mengenali 7 gestur perintah aktif dan 1 kelas latar belakang dari jendela berjalan (*sliding window*) sepanjang $T = 32$ frame:

| Class ID | Nama Kelas | Label Dataset 20BN-Jester | Aksi OS (PyAutoGUI) | Shortcut Key / Trigger |
| :---: | :--- | :--- | :--- | :--- |
| `0` | **Swipe Left** | `Swiping Left` | Halaman / Slide Sebelumnya | `Left Arrow` |
| `1` | **Swipe Right** | `Swiping Right` | Halaman / Slide Berikutnya | `Right Arrow` |
| `2` | **Scroll Up** | `Sliding Two Fingers Up` | Gulir Layar ke Atas | `scroll(+clicks)` |
| `3` | **Scroll Down** | `Sliding Two Fingers Down` | Gulir Layar ke Bawah | `scroll(-clicks)` |
| `4` | **Zoom In** | `Zooming In With Full Hand` | Perbesar Tampilan | `Ctrl` + `+` |
| `5` | **Zoom Out** | `Zooming Out With Full Hand` | Perkecil Tampilan | `Ctrl` + `-` |
| `6` | **Stop** | `Stop Sign` | Jeda Video / Media | `Space` / `PlayPause` |
| `7` | **Background** | `No gesture` + `Doing other things` | *Idle* / Tidak Ada Aksi | `None` |

### FR-03: Multi-Model Hot-Swapping
- Pengguna dapat memilih dan mengganti model inferensi aktif secara langsung dari antarmuka tanpa memulai ulang aplikasi:
  1. **BiLSTM** (Baseline — Bidirectional LSTM + Attention Pooling)
  2. **ST-GCN** (Alternatif 1 — Spatio-Temporal Graph Convolutional Network)
  3. **TCN** (Alternatif 2 — Causal Dilated Temporal Convolutional Network)

### FR-04: False-Trigger Mitigation Engine (Post-Processing Gate)
Untuk mencegah perintah tak disengaja (*false trigger*), aksi OS hanya dieksekusi apabila seluruh kondisi berikut terpenuhi:
1. **Motion Gate:** Kecepatan rata-rata pergelangan tangan/sendi melampaui ambang batas diam minimum.
2. **Confidence Threshold:** Probabilitas *softmax* kelas prediksi $\ge 0.80$ (dapat diatur melalui slider UI).
3. **Temporal Consistency:** Kelas prediksi yang sama muncul secara konsisten pada **3 window inferensi berturut-turut**.
4. **Cooldown Timer:** Jeda waktu **0.8 – 1.0 detik** setelah sebuah perintah berhasil dieksekusi sebelum perintah berikutnya dapat dipicu.

### FR-05: OS Action Execution & Safety Master Toggle
- Menyediakan tombol **Master Arm/Disarm Toggle** (`OS Control: ENABLED / DISABLED`) untuk mengaktifkan atau mematikan injeksi perintah PyAutoGUI saat pengguna hanya ingin menguji akurasi visual.

---

## 4. UI/UX Design Requirements (Raycast Design System Adaptation)

Mengacu pada `design-md/raycast/DESIGN.md`, antarmuka Streamlit/Web HUD wajib menerapkan spesifikasi visual berikut:

### 4.1 Color Palette Tokens
- **Primary Background Canvas:** `#07080a` (Deep Inky Black)
- **Elevated Card / HUD Surface:** `#101111` hingga `#18191a` dengan efek translusensi kaca (*glassmorphism* `backdrop-filter: blur(16px)`)
- **Hairline Border Token:** `1px solid #242728` (dan `rgba(255, 255, 255, 0.08)` untuk pemisah komponen)
- **Typography Colors:**
  - Primary Text: `#f9f9f9`
  - Secondary / Muted Text: `#9c9c9d`
  - Disabled / Technical Label: `#6a6b6c`
- **Semantic Accent Tokens:**
  - **Raycast Coral Red (Brand / Active Recording):** `#ff6363`
  - **Trigger Success / Valid Gesture Pill:** `#56c2ff` (Cyan Blue) atau `#30d158` (Emerald Green)
  - **Cooldown / Gate Locked Warning:** `#ffd60a` (Amber)

### 4.2 Typography & Scannable Hierarchy
- **UI Sans Font:** `Inter`, `-apple-system`, atau `SF Pro Display` (`font-weight: 500/600`, `letter-spacing: -0.02em`).
- **Telemetry Monospace Font:** `Geist Mono` atau `JetBrains Mono` untuk seluruh angka latensi (`ms`), `FPS`, ukuran buffer (`32/32`), dan nilai probabilitas kelas agar tidak bergeser (*tabular-nums*).

### 4.3 Layout Components
1. **Top Command Header Bar:** Menampilkan status sistem (`READY`, `TRIGGERED`, `COOLDOWN`), pilihan model (`BiLSTM | ST-GCN | TCN`), dan Master Toggle OS Control.
2. **Main Viewport (Left 65%):** Umpan kamera berbingkai `#242728` (`border-radius: 12px`), dilengkapi overlay titik sendi MediaPipe dan *Floating Action Toast* bergaya Raycast Command Pill saat gestur terpicu.
3. **Telemetry & Probability HUD (Right 35%):**
   - **Real-Time Metrics Grid:** 4 kartu metrik monospasi (`Inference Latency`, `End-to-End FPS`, `Buffer Fill`, `Cooldown Status`).
   - **8-Class Softmax Confidence Bars:** Grafik batang horizontal real-time yang menunjukkan distribusi probabilitas ke-8 kelas beserta garis batas ambang konfidensi (`0.80`).
   - **Recent Action Log:** Riwayat 5 perintah terakhir lengkap dengan *timestamp* dan latensi eksekusi.

---

## 5. Success Criteria & KPIs
Berdasarkan proposal dan rubrik penilaian **COMP6826001**:

| Kategori | Indikator Keberhasilan (Target) |
| :--- | :--- |
| **Kinerja Model (Test Set)** | Macro F1-Score tertinggi dipilih; Precision kelas kontrol diprioritaskan di atas Recall untuk meminimalkan *false trigger*. |
| **Latensi Inferensi Model** | $< 20\text{ ms}$ per window (32 frame) pada CPU menggunakan ONNX Runtime. |
| **Latensi Gestur-ke-Aksi** | $< 1.0\text{ detik}$ dari akhir gerakan hingga perintah OS terpicu. |
| **Kecepatan Aliran (Throughput)** | $\ge 25\text{ FPS}$ end-to-end pada laptop non-GPU. |
| **Ukuran Berkas Model** | $< 5\text{ MB}$ dalam format `.onnx`. |
| **Uji Pengguna (n = 8–10)** | Skor **System Usability Scale (SUS) $\ge 68$** (target $\ge 80$) dan **False Trigger Rate $< 1$ kejadian per 5 menit** pada pengujian *idle* (T5). |

---

## 6. Ethics, Privacy & Responsible AI
- **Zero Video Storage:** Seluruh pemrosesan frame webcam dilakukan di memori volatil (RAM) secara lokal. Tidak ada citra wajah atau rekaman video pengguna yang disimpan ke diska atau dikirim ke server eksternal.
- **Informed Consent:** Seluruh partisipan pengujian pengguna (8–10 orang) menandatangani lembar persetujuan pengujian dan dievaluasi melintasi variasi tangan dominan serta warna kulit.