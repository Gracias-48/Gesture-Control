# Application Flow & State Architecture (`APP_FLOW.md`)
**Project:** Touchless Gesture Control HUD  
**Design Theme:** Raycast Command Utility (`#07080a` Canvas, Hairline Borders `#242728`, Coral Accent `#ff6363`)

---

## 1. End-to-End Real-Time Processing Flow

```
+-------------------------------------------------------------------------+
| [1] WEBCAM CAPTURE THREAD (OpenCV @ ~30 FPS)                            |
|     └── Timestamp Resampler (Δt >= 83.3 ms -> Target 12 FPS)            |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| [2] MEDIAPIPE HANDS LANDMARK EXTRACTOR                                  |
|     ├── Hand Detected?                                                  |
|     │   ├── YES -> Extract 21 (x, y, z) landmarks                       |
|     │   └── NO  -> Linear interpolation (if gap <= 30% of buffer)       |
|     │              or Reset Buffer if hand absent > 1.0 second          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| [3] SLIDING RING BUFFER (Deque MaxLen = 32 Frames)                      |
|     ├── Compute Local Stream (Wrist-centered, Middle-MCP scaled)        |
|     ├── Compute Global Stream (Wrist position + velocity Δp)            |
|     └── Apply Z-Score Standardization (from train scaler_stats.npz)     |
+------------------------------------+------------------------------------+
                                     | (Triggered every 4 new frames)
                                     v
+-------------------------------------------------------------------------+
| [4] PRE-INFERENCE MOTION GATE                                           |
|     └── Is Mean Joint Velocity > v_min?                                 |
|         ├── NO  -> Force state = IDLE (Skip ONNX or output Background)  |
|         └── YES -> Proceed to ONNX Runtime Inference                    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| [5] ONNX RUNTIME ENGINE (CPU ExecutionProvider, Latency < 20 ms)        |
|     ├── Model Selection: [ BiLSTM.onnx | STGCN.onnx | TCN.onnx ]        |
|     └── Output: 8-Class Softmax Probability Vector P = [p_0 ... p_7]    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| [6] POST-PROCESSING DECISION GATE                                       |
|     ├── Check 1: argmax(P) != 7 (Not Background)                        |
|     ├── Check 2: max(P) >= Confidence Threshold (default 0.80)          |
|     ├── Check 3: Prediction matches last 3 consecutive windows (3/3)    |
|     └── Check 4: Cooldown timer expired (t_now - t_last >= 0.8s)        |
+------------------------------------+------------------------------------+
                                     | (All 4 Checks PASS)
                                     v
+-------------------------------------------------------------------------+
| [7] ACTION DISPATCHER & RAYCAST HUD FEEDBACK                            |
|     ├── If OS Control ARMED -> Execute PyAutoGUI command                |
|     ├── Activate Cooldown Timer (0.8s - 1.0s)                           |
|     └── Render Raycast Toast Pill + Update Telemetry Log                |
+-------------------------------------------------------------------------+
```

---

## 2. Application State Machine

Sistem HUD memiliki **5 status operasional (States)** yang langsung direpresentasikan melalui badge warna bergaya Raycast di bagian atas layar:

1. **`STATE_STANDBY` (Badge: Gray `#6a6b6c`)**
   - Kamera belum diaktifkan atau tangan tidak terlihat di depan kamera selama lebih dari 1 detik. Ring buffer dikosongkan.
2. **`STATE_BUFFERING` (Badge: Blue `#56c2ff`)**
   - Tangan baru terdeteksi dan sistem sedang mengumpulkan frame hingga ring buffer mencapai kapasitas $32/32\text{ frame}$.
3. **`STATE_MONITORING` (Badge: Emerald `#30d158`)**
   - Buffer penuh ($32/32$), inferensi berjalan setiap 4 frame, namun gerakan saat ini terklasifikasi sebagai `Background` atau belum melewati ambang konsistensi 3 window.
4. **`STATE_TRIGGERED` (Badge: Raycast Coral `#ff6363` / Cyan `#56c2ff`)**
   - Gestur valid terkonfirmasi (konfidensi $\ge 0.80$ selama 3 window berturut-turut). Perintah PyAutoGUI langsung ditembakkan dan *Floating Action Toast* muncul di layar.
5. **`STATE_COOLDOWN` (Badge: Amber `#ffd60a`)**
   - Selama $0.8 - 1.0\text{ detik}$ pasca-eksekusi perintah, sistem mengabaikan pemicu baru agar gerakan kembali tangan (*return stroke*) tidak memicu perintah berlawanan (misal: tangan kembali ke kanan setelah melakukan *Swipe Left*).

---

## 3. Raycast-Inspired UI Wireframe & Component Layout

```
+-----------------------------------------------------------------------------------+
| [●] TOUCHLESS GESTURE HUD   [Model: ST-GCN v]  [Threshold: 0.80]  [OS: ARMED (⌘)] |
+---------------------------------------------------+-------------------------------+
|                                                   | TELEMETRY (MONOSPACE)         |
|   +-------------------------------------------+   | +------------+ +------------+ |
|   | [LIVE 12 FPS]              [STATE: READY] |   | | LATENCY    | | THROUGHPUT | |
|   |                                           |   | |  4.8 ms    | |  28.5 FPS  | |
|   |               (o)---(o)                   |   | +------------+ +------------+ |
|   |              /  |   |  \                  |   | +------------+ +------------+ |
|   |            (o) (o) (o) (o)                |   | | BUFFER     | | CONSISTENCY| |
|   |               \  |  /                     |   | |  32 / 32   | |   3 / 3    | |
|   |                 (o) Wrist                 |   | +------------+ +------------+ |
|   |                                           |   |                               |
|   |     +-------------------------------+     |   | CLASS CONFIDENCE (8 CLASSES)  |
|   |     |  [→] SWIPE RIGHT  •  94.2%    |     |   | Swipe Right  [█████████░] 94% |
|   |     |      Action: Next Slide       |     |   | Background   [█░░░░░░░░░]  4% |
|   |     +-------------------------------+     |   | Swipe Left   [░░░░░░░░░░]  1% |
|   +-------------------------------------------+   | Scroll Up    [░░░░░░░░░░]  1% |
|                                                   | Scroll Down  [░░░░░░░░░░]  0% |
|   RECENT COMMAND TRIGGER LOG                      | Zoom In      [░░░░░░░░░░]  0% |
|   10:42:18.420  [SWIPE_RIGHT] -> Key.Right (4.8ms)| Zoom Out     [░░░░░░░░░░]  0% |
|   10:42:12.105  [ZOOM_IN]     -> Ctrl + +  (5.1ms)| Stop         [░░░░░░░░░░]  0% |
+---------------------------------------------------+-------------------------------+
```

---

## 4. User Testing Flow (Within-Subject Protocol, $n = 8 - 10$)

Aplikasi menyediakan tab/mode khusus **`[Evaluation & User Study Mode]`** untuk memandu skenario pengujian T1 hingga T5 secara otomatis dan mencatat hasilnya ke berkas CSV:

1. **Onboarding & Calibration (3 Menit):**
   - Penguji membaca lembar persetujuan (*informed consent*), mengisi metadata anonim (tangan dominan, pengalaman gestur), dan mencoba gerakan bebas selama 2 menit.
2. **Guided Task Execution — T1 s/d T4 (15 Menit):**
   - Sistem menampilkan instruksi target di layar (misal: *"Lakukan Swipe Left sebanyak 10 kali"*).
   - Setiap percobaan dicatat sebagai `Correct Trigger`, `Wrong Class Trigger`, atau `Missed Gesture`, beserta waktu responsnya.
3. **Idle False-Trigger Stress Test — T5 (3 Menit):**
   - Sistem diaktifkan pada mode pemantauan sementara penguji diminta berbicara, mengetik, dan memegang gelas minum tanpa niat memberi perintah.
   - Sistem menghitung jumlah *false trigger* yang lolos selama 3 menit (Target: $< 1\text{ per } 5\text{ menit}$).
4. **Digital SUS Questionnaire (4 Menit):**
   - Penguji mengisi 10 pertanyaan **System Usability Scale (Skala Likert 1–5)** langsung di dalam antarmuka Streamlit, dan sistem otomatis menghitung skor SUS akhir ($0 - 100$).