# Technical Requirements Document (TRD)
**Project Name:** Touchless Gesture Control HUD  
**Target Runtime:** Local CPU Execution (Windows / macOS / Linux) via ONNX Runtime  
**Primary Frameworks:** PyTorch (Training), MediaPipe Hands (Feature Extraction), ONNX Runtime (Inference), Streamlit + Custom CSS (Raycast UI), PyAutoGUI (OS Action)

---

## 1. System Architecture Overview

Sistem dibagi menjadi dua jalur utama (*pipelines*):
1. **Offline Training & Export Pipeline:** Mengubah klip video JPEG dari dataset 20BN-Jester menjadi tensor koordinat `.npy`, melatih 3 arsitektur Deep Learning di PyTorch pada 12 skenario hyperparameter (E1–E12), mengevaluasi metrik, dan mengekspor bobot terbaik ke format `.onnx`.
2. **Real-Time Streaming Inference Pipeline:** Menangkap webcam dengan OpenCV pada thread terpisah, mengekstrak landmark dengan MediaPipe pada 12 FPS, menyusun *ring buffer* 32 frame, mengeksekusi model ONNX setiap 4 frame, menerapkan logika *post-processing gate*, dan memicu aksi OS via PyAutoGUI.

---

## 2. Data Engineering & Feature Extraction Specifications

### 2.1 Dataset Subset & Splitting
- **Source:** 20BN-Jester Dataset (~40.000 klip untuk 8 kelas target; kelas `Background` merupakan gabungan subsampling `No gesture` dan `Doing other things` sebanyak ~6.000 klip).
- **Splitting Strategy:** Penggabungan set berlabel resmi, dibagi ulang secara **Stratified Split (70% Train / 15% Validation / 15% Test)** dengan `random_state = 42`.

### 2.2 Offline Landmark Extraction Protocol
- Setiap frame JPEG (tinggi asli $100\text{ px}$) di-*upscale* menggunakan interpolasi bikubik ke tinggi **$256\text{ px}$** sebelum dimasukkan ke **MediaPipe Hands** (`static_image_mode=False`, `max_num_hands=1`, `min_detection_confidence=0.5`) untuk menjaga stabilitas deteksi.
- **Missing Frame Handling:**
  - Jika frame gagal mendeteksi tangan, koordinat diinterpolasi secara linear dari frame valid terdekat:
    $$\mathbf{p}_t = \mathbf{p}_{t_1} + \frac{t - t_1}{t_2 - t_1}(\mathbf{p}_{t_2} - \mathbf{p}_{t_1})$$
  - Klip dengan $> 30\%$ frame hilang dibuang dari dataset pelatihan.
- **Temporal Standardization:** Setiap klip diseragamkan menjadi panjang tetap **$T = 32\text{ frame}$** ($\approx 2.7\text{ detik}$ pada $12\text{ FPS}$) melalui *uniform temporal resampling* atau *center cropping/padding*.

### 2.3 Two-Stream Feature Normalization & Tensor Shapes
Agar model mengenali artikulasi jari sekaligus arah lintasan global (seperti *Swipe Left* vs *Swipe Right*), fitur dinormalisasi menjadi dua aliran:

1. **Aliran Lokal (Artikulasi Jari Invarian Skala):**
   - Titik pusat ditranslasikan ke pergelangan tangan (*wrist*, landmark $0$): $\tilde{\mathbf{p}}_{t,i} = \mathbf{p}_{t,i} - \mathbf{p}_{t,0}$.
   - Diskalakan dengan jarak Euclidean antara pergelangan tangan ($0$) dan pangkal jari tengah (*Middle MCP*, landmark $9$):
     $$d_t = \|\mathbf{p}_{t,9} - \mathbf{p}_{t,0}\|_2 + \epsilon, \quad \hat{\mathbf{p}}_{t,i} = \frac{\tilde{\mathbf{p}}_{t,i}}{d_t}$$
2. **Aliran Global & Kinematika (Lintasan & Kecepatan):**
   - Posisi absolut pergelangan tangan $(x_{t,0}, y_{t,0}, z_{t,0})$ dan vektor kecepatan pergelangan tangan $\Delta \mathbf{p}_{t,0} = \mathbf{p}_{t,0} - \mathbf{p}_{t-1,0}$.
   - Kecepatan sendi lokal $\Delta \hat{\mathbf{p}}_{t,i} = \hat{\mathbf{p}}_{t,i} - \hat{\mathbf{p}}_{t-1,i}$ serta magnitudo kecepatan global.
3. **Standardisasi Z-Score:** Dihitung hanya dari statistik *training split* ($\mu_{\text{train}}, \sigma_{\text{train}}$) dan disimpan ke berkas `scaler_stats.npz`.

#### Spesifikasi Dimensi Input Tensor:
- **Untuk BiLSTM & TCN:** Tensor berdimensi `(Batch, T=32, Features=131)`
  - $21 \text{ sendi} \times 3 \text{ koordinat lokal } (\hat{x}, \hat{y}, \hat{z}) = 63$
  - $21 \text{ sendi} \times 3 \text{ kecepatan lokal } (\Delta\hat{x}, \Delta\hat{y}, \Delta\hat{z}) = 63$
  - $3 \text{ posisi global wrist} + 2 \text{ kecepatan planar wrist } (\Delta x_0, \Delta y_0) = 5$
  - Total = **$131\text{ fitur per frame}$**.
- **Untuk ST-GCN:** Tensor graf berdimensi `(Batch, Channels=10, T=32, Vertices=21)`
  - Setiap simpul sendi $v \in \{0,\dots,20\}$ memuat: $3$ koordinat lokal + $3$ kecepatan lokal + $3$ koordinat wrist terdistribusi + $1$ magnitudo kecepatan = **$10\text{ kanal per simpul}$**.

### 2.4 Online Training Augmentation (Train Split Only)
- **Rotasi Spasial 2D:** Rotasi acak pada bidang $(x,y)$ sebesar $\theta \sim \mathcal{U}(-15^\circ, +15^\circ)$.
- **Skalasi:** Pengali skala acak $s \sim \mathcal{U}(0.8, 1.2)$.
- **Gaussian Jitter:** Penambahan derau $\mathcal{N}(0, 0.01^2)$ pada koordinat sendi.
- **Speed Perturbation:** Resampling temporal dengan faktor kecepatan $0.8\times \text{ hingga } 1.2\times$ lalu dikembalikan ke $T = 32$.
- **Frame Dropout:** Mengatur $5\% - 10\%$ frame acak menjadi nol/interpolasi untuk menyimulasikan kegagalan deteksi sesaat.
- **Label-Preserving Horizontal Flip:** Membalik sumbu $x \leftarrow -x$ dengan **menukar label kelas `0 (Swipe Left)` $\leftrightarrow$ `1 (Swipe Right)`** secara otomatis.

---

## 3. Deep Learning Architectures & Experimental Matrix

### 3.1 Model Specifications
1. **Model 1 — BiLSTM (Baseline, $\approx 0.6\text{M}$ params):**
   - Proyeksi Input Linear `131 -> 128` + LayerNorm + ReLU.
   - 2 lapis `nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)`.
   - Temporal Attention Pooling: menghitung bobot perhatian $\alpha_t = \text{softmax}(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_t))$ di sepanjang $T = 32$.
   - Classifier Head: `FC(256 -> 128) -> GELU -> Dropout -> FC(128 -> 8)`.
2. **Model 2 — ST-GCN (Alternatif 1, $\approx 0.3\text{M} - 1.0\text{M}$ params):**
   - Graf anatomis 21 titik tangan MediaPipe ($\mathbf{A} \in \mathbb{R}^{3 \times 21 \times 21}$ dengan partisi spasial *root, centripetal, centrifugal* sesuai Yan et al., 2018).
   - 4 Blok ST-GCN berurutan dengan kanal `(64, 64, 128, 128)`, kernel konvolusi temporal $K_t = 9$, koneksi residual per blok, serta *learnable edge importance weighting* $\mathbf{M}$.
   - Global Average Pooling melintasi $(T, V)$ diikuti `FC(128 -> 8)`.
3. **Model 3 — Causal Dilated TCN (Alternatif 2, $\approx 0.2\text{M}$ params):**
   - 4 Blok Residual Temporal dengan konvolusi 1D ter-dilasi kausal (*causal padding*), ukuran kernel $k = 3$, faktor dilasi $d \in \{1, 2, 4, 8\}$, dan $64\text{ kanal}$ tersembunyi (Bai et al., 2018).
   - Mengambil representasi langkah waktu terakhir atau *masked average pooling* diikuti `FC(64 -> 8)`.

### 3.2 Training Protocol & 12-Scenario Matrix (E1–E12)
- **Loss Function:** `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Gradient Clipping:** `max_norm = 1.0`
- **Early Stopping:** Memantau `Validation Macro F1-Score` dengan `patience = 10` epoch.
- **Multi-Seed Reproducibility:** Setiap konfigurasi dijalankan pada 3 seed berbeda (`42`, `100`, `2026`), dilaporkan sebagai $\text{Mean} \pm \text{SD}$.

| Exp ID | Model | Optimizer | Learning Rate | Batch Size | Max Epochs | Dropout | Weight Decay | LR Scheduler |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1** | BiLSTM | Adam | $1 \times 10^{-3}$ | 64 | 60 | 0.3 | $0$ | Constant |
| **E2** | BiLSTM | AdamW | $1 \times 10^{-3}$ | 64 | 60 | 0.3 | $1 \times 10^{-2}$ | Cosine Annealing |
| **E3** | BiLSTM | AdamW | $5 \times 10^{-4}$ | 128 | 80 | 0.4 | $1 \times 10^{-2}$ | ReduceLROnPlateau |
| **E4** | BiLSTM | AdamW | $1 \times 10^{-3}$ | 32 | 60 | 0.5 | $5 \times 10^{-2}$ | OneCycleLR |
| **E5** | ST-GCN | Adam | $1 \times 10^{-3}$ | 64 | 80 | 0.3 | $0$ | StepLR ($\gamma=0.1$ @ ep 40, 60) |
| **E6** | ST-GCN | AdamW | $1 \times 10^{-3}$ | 64 | 80 | 0.3 | $1 \times 10^{-2}$ | Cosine Annealing |
| **E7** | ST-GCN | AdamW | $5 \times 10^{-4}$ | 128 | 100 | 0.4 | $5 \times 10^{-2}$ | Cosine + Warmup (5 ep) |
| **E8** | ST-GCN | AdamW | $2 \times 10^{-3}$ | 32 | 80 | 0.5 | $1 \times 10^{-2}$ | OneCycleLR |
| **E9** | TCN | Adam | $1 \times 10^{-3}$ | 64 | 60 | 0.2 | $0$ | Constant |
| **E10** | TCN | AdamW | $1 \times 10^{-3}$ | 64 | 60 | 0.2 | $1 \times 10^{-2}$ | Cosine Annealing |
| **E11** | TCN | AdamW | $2 \times 10^{-3}$ | 128 | 80 | 0.3 | $5 \times 10^{-2}$ | OneCycleLR |
| **E12** | TCN | AdamW | $5 \times 10^{-4}$ | 32 | 100 | 0.4 | $1 \times 10^{-2}$ | ReduceLROnPlateau |

### 3.3 Mandatory Ablation Studies
Dilakukan pada konfigurasi terbaik masing-masing model:
1. **Tanpa Aliran Global:** Hanya menggunakan fitur lokal sendi (menguji penurunan akurasi pada *Swipe* dan *Scroll*).
2. **Tanpa Augmentasi Data:** Melatih tanpa *flip/rotation/speed perturbation*.
3. **Variasi Panjang Window Temporal:** Membandingkan $T \in \{16, 32, 48\}$ terhadap Macro F1 dan latensi inferensi.
4. **Tanpa Kelas Background:** Melatih hanya pada 7 kelas aktif untuk membuktikan pentingnya kelas `Background` terhadap *False Trigger Rate*.

---

## 4. Real-Time Inference & Deployment Stack

- **ONNX Export Specification:**
  - `opset_version = 17`, `do_constant_folding = True`, dynamic batch axis.
  - Diverifikasi menggunakan `onnx.checker.check_model()` dan diuji kesamaan output numerik antara PyTorch dan `onnxruntime.InferenceSession` ($\text{atol} < 10^{-5}$).
- **Threading & Concurrency:**
  - **Thread 1 (Camera Producer):** Membaca frame dari `cv2.VideoCapture(0)` secara non-blocking pada ~30 FPS dan melakukan *timestamp downsampling* setiap $\Delta t \approx 83.3\text{ ms}$ ($12\text{ FPS}$).
  - **Thread 2 (Inference & Gate Consumer):** Mengelola `collections.deque(maxlen=32)`, menjalankan normalisasi dua aliran dan sesi ONNX setiap **stride = 4 frame** ($\approx 3\text{ kali inferensi per detik}$), serta mengevaluasi *post-processing gate*.
  - **Main Thread (Streamlit Raycast UI):** Merender video overlay, indikator konfidensi, dan telemetri tanpa memblokir pembacaan kamera.