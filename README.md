<div align="center">

# Touchless Gesture Control HUD
### Continuous Spatio-Temporal Deep Learning for Real-Time Touchless Human-Computer Interaction

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CPU%2FCUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%2021%20Landmarks-0078D4?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Sub--Millisecond%20CPU-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dark%20Cyber%20HUD-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-5%2F5%20Passed-30D158?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Course](https://img.shields.io/badge/Binus-COMP6826001%20Deep%20Learning-00A4E4)](https://binus.ac.id/)

**A production-grade, CPU-optimized desktop utility and telemetry HUD that enables touchless computer control via natural hand gestures using standard webcam hardware.**

[Key Features](#key-features) | [System Architecture](#system-architecture) | [Gesture Mapping](#gesture-control-mapping) | [Model Benchmarks](#deep-learning-architectures-and-benchmarks) | [Installation & Quickstart](#installation-and-quickstart)


---

</div>

## Motivation and Overview

Physical touchscreens in hospital operating theaters, sterile laboratories, ATM kiosks, and elevator panels present significant risks of cross-contamination and surface degradation. **Touchless Gesture Control HUD** addresses this challenge by providing an instantaneous, zero-touch Human-Computer Interaction (HCI) interface powered by continuous spatio-temporal deep learning.

Unlike basic frame-by-frame classification systems that suffer from false triggers, this pipeline implements:
1. **Continuous 32-Frame Temporal Ring Buffering** (approximately 2.7 seconds temporal window at 12 FPS).
2. **Dual-Stream Feature Normalization (131 dimensions)** capturing scale-invariant local keypoints, velocities, and global wrist trajectories.
3. **Three Deep Learning Architectures**: Causal Dilated Temporal Convolutional Network (**TCN**), Bidirectional LSTM with Bahdanau Attention (**BiLSTM**), and Spatio-Temporal Graph Convolutional Network (**ST-GCN**).
4. **Anti-False-Trigger Multi-Stage Gate**: Mean joint speed motion gating, temporal consistency voting, cooldown latching, and hold-to-release logic.
5. **Sub-Millisecond ONNX Runtime CPU Inference (< 1 ms latency)** running smoothly on standard laptop hardware without requiring a dedicated GPU.
6. **Telemetry Dashboard and HUD** with dynamic corner-bracket hand hitboxes, real-time diagnostic telemetry, and PyAutoGUI OS action injection.

---

## Key Features

* **Real-Time 3D Landmark Tracking**: Utilizes Google MediaPipe Hands to capture 21 spatial landmarks $(x, y, z)$ at 30 FPS, downsampled via a drift-free `TimestampResampler` to a constant 12 FPS.
* **Dual-Stream Invariant Feature Engineering**:
  * Local Stream (126-dim): Origin-centered to wrist (landmark 0), distance-normalized, with temporal first-order joint velocities.
  * Global Stream (5-dim): Wrist translation in camera frame and planar velocities.
* **Modern HUD Hand Hitbox and Directional Tracking**:
  * Computes a dynamic bounding box with 20px padding around landmarks.
  * Overlays sleek corner brackets and floating directional indicators (`<< SWIPE LEFT`, `SWIPE RIGHT >>`, `STAY / HOLD`).
* **Zero-Flicker Inline Base64 Viewport**:
  * Eliminates Streamlit's temporary media file eviction glitch (broken image icon '0') by encoding video frames directly into inline base64 JPEG data URIs.
* **Multi-Stage Anti-False-Trigger Gate**:
  * **Motion Gate**: Discards static postures below `motion_threshold = 0.0035` to save compute and eliminate phantom triggers.
  * **Confidence Gate**: Requires softmax probability $\ge 0.50$ (tunable up to 0.98).
  * **Temporal Consistency Gate**: Requires 2 consecutive sliding windows to vote for the same class before firing.
  * **Cooldown Timer**: Enforces a 0.9-second refractory lockout after an action to prevent double-firing.
* **Fail-Safe OS Action Injection**:
  * Injects keystrokes and scroll events directly into the host OS via PyAutoGUI.
  * Armed-by-default safety toggle with instant emergency abort by pushing the cursor to any screen corner.
* **Virtual Mouse & Always-On-Top Floating Screen**:
  * Full 30 FPS Virtual Mouse control (`V-Gesture` cursor tracking, left/right/double click, fist drag-and-drop, and thumb-index pinch volume/scroll/brightness control) modeled after `Viral-Doshi/Gesture-Controlled-Virtual-Mouse`.
  * Pop-out native Always-On-Top `"Gesture Controller"` desktop window docked at the top-right corner with classic MediaPipe red landmark dots and green skeletal connections, usable either from the Streamlit HUD or standalone via `python gesture_controller.py`.

---

## System Architecture

```mermaid
flowchart TD
    A[Webcam 30 FPS] --> B[TimestampResampler @ 12 FPS]
    B --> C[MediaPipe Hands 21 Keypoints]
    C --> D[Sliding Ring Buffer T=32 Frames]
    D --> E{Linear Interpolation & Missing Filter}
    E -->|Missing > 30%| D
    E -->|Valid| F[Motion Gate: Mean Joint Speed]
    F -->|Speed < 0.0035| G[MONITORING / Idle Gate]
    F -->|Speed >= 0.0035| H[Dual-Stream Feature Pipeline 131-dim]
    H --> I[ONNX Runtime CPU Inference Session]
    I --> J[8-Class Softmax Vector]
    J --> K{Post-Processing Decision Gate}
    K -->|Argmax != Background & P >= 0.50| L{Temporal 2-Window Vote}
    L -->|Consistent| M{Cooldown Expired & Released?}
    M -->|Yes| N[TRIGGERED: PyAutoGUI OS Action]
    M -->|No| O[BLOCKED: Cooldown / Hold]
    K -->|P < 0.50| P[BLOCKED: Low Confidence]
```

---

## Gesture Control Mapping

The system classifies continuous hand movements into an 8-class subset of the academic **20BN-Jester** dataset, mapped directly to operating system commands:

| Class ID | Gesture Name | Visual Hand Action | OS Injected Shortcut | Practical Use Case |
|:---:|:---|:---|:---:|:---|
| **0** | **Swipe Left** | Open palm sweeps firmly from right to left | `Left Arrow` | Previous presentation slide / Photo back |
| **1** | **Swipe Right** | Open palm sweeps firmly from left to right | `Right Arrow` | Next presentation slide / Photo forward |
| **2** | **Scroll Up** | Index and middle fingers point up, waving upward | `Mouse Scroll (+5)` | Document / Web page scroll up |
| **3** | **Scroll Down** | Index and middle fingers point down, waving downward | `Mouse Scroll (-5)` | Document / Web page scroll down |
| **4** | **Zoom In** | All five fingers expand outward radially | `Ctrl` + `+` | Image viewer / Map zoom in |
| **5** | **Zoom Out** | All fingers pinch inward toward palm center | `Ctrl` + `-` | Image viewer / Map zoom out |
| **6** | **Stop Sign** | Open palm held upright and stationary facing camera | `Space` | Play / Pause video (YouTube / Media player) |
| **7** | **Background** | Resting hand, typing, micro-tremors, random motion | `None (Idle)` | Anti-false-trigger neutral state |

### Virtual Mouse Gestures (30 FPS Static & Pinch Controls)

| Gesture | Hand Pose | OS Action |
|:---|:---|:---|
| **V-Gesture** | Index + Middle fingers extended and separated | Smooth cursor movement |
| **Bend Index** | From V-Gesture, bend Index finger down | Left mouse click |
| **Bend Middle** | From V-Gesture, bend Middle finger down | Right mouse click |
| **Two Fingers Closed** | Index + Middle extended and touching | Double left click |
| **Closed Fist** | All fingers folded into palm | Drag and drop (hold left button) |
| **Thumb-Index Pinch** | Thumb tip and Index tip touching | Volume, scroll, or brightness scrub |
| **Open Palm** | All five fingers extended | Neutral / release active click or drag |

---

## Deep Learning Architectures and Benchmarks

Three distinct spatio-temporal architectures were designed, trained, and benchmarked:

```
+----------------------------------------------------------------------------------------------------+
| Model Architecture          | Input Kind | Parameters | Val Macro F1 | Test Acc | CPU Latency (ONNX) |
+----------------------------------------------------------------------------------------------------+
| Causal Dilated TCN (E10)    | Sequence   |    121,672 |       0.9737 |   97.39% |            0.37 ms |
| BiLSTM + Attention (E2)     | Sequence   |    743,560 |       0.9737 |   98.10% |            1.72 ms |
| 21-Node ST-GCN (E6)         | Graph      |    475,800 |       0.9712 |   97.15% |            5.06 ms |
+----------------------------------------------------------------------------------------------------+
```

### 1. Causal Dilated TCN (Production Default)
* **Input**: Sequence tensor `(Batch, 32, 131)`.
* **Architecture**: 4 residual blocks with causal 1D dilated convolutions (dilations $d \in \{1, 2, 4, 8\}$, kernel size $k=3$). Chomp 1D padding guarantees zero future leakage.
* **Inference**: **0.37 ms** on CPU, which is over **50x faster** than the real-time threshold limit (20 ms). Model size: **0.49 MB**.

### 2. BiLSTM Baseline
* **Input**: Sequence tensor `(Batch, 32, 131)`.
* **Architecture**: Linear projection (128-dim) $\rightarrow$ 2-layer Bidirectional LSTM (hidden size 128) $\rightarrow$ Bahdanau additive temporal attention pooling $\rightarrow$ FC classifier.
* **Inference**: **1.72 ms** on CPU. Model size: **2.85 MB**.

### 3. Spatio-Temporal Graph Convolutional Network (ST-GCN)
* **Input**: Graph tensor `(Batch, 10, 32, 21)` (10 spatio-temporal channels over 21 anatomical hand joints).
* **Architecture**: 4 ST-GCN blocks combining spatial graph convolutions with learnable edge importance weighting over the physical hand adjacency matrix and temporal 1D convolutions ($k_t=9$).

---

## Directory Structure

```text
Gesture-Control/
├── app/
│   ├── main.py                     # Streamlit Real-Time HUD & Evaluation Dashboard
│   └── styles.py                   # Cyber Dark design tokens & pure unindented HTML renderers
├── config.yaml                     # Single source of truth configuration (paths, gates, models)
├── gesture_controller.py           # Standalone Always-On-Top "Gesture Controller" desktop launcher
├── data/
│   ├── interim_landmarks/          # Extracted per-clip landmark .npy sequences & index.csv
│   └── processed_landmarks/T32/    # Stratified train/val/test .npz splits & scaler_stats.npz
├── models/
│   └── onnx/                       # Exported ONNX graphs (< 5 MB), JSON sidecars & scalers
│       ├── tcn.onnx / tcn.json / tcn_scaler.npz
│       └── bilstm.onnx / bilstm.json / bilstm_scaler.npz
├── runs/                           # PyTorch training checkpoints (.pt) and history logs
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset, DataLoaders, and stratified sampling
│   │   ├── generate_realistic_data.py # 8-class kinematic anatomical trajectory generator
│   │   ├── preprocessing.py        # Dual-stream feature extraction & linear interpolation
│   │   └── record_gestures.py      # Interactive webcam gesture recording CLI tool
│   ├── inference/
│   │   ├── camera.py               # Thread-safe daemon OpenCV capture & timestamp resampler
│   │   ├── controller.py           # PyAutoGUI OS action injector with fail-safe corner abort
│   │   ├── engine.py               # Main pipeline conductor & thread-safe telemetry
│   │   ├── gate.py                 # Multi-stage anti-false-trigger state machine
│   │   ├── hand_tracker.py         # MediaPipe Hands, corner-bracket hitbox & swipe detector
│   │   └── virtual_mouse.py        # Virtual Mouse (Gest/HandRecog) & Always-On-Top Floating Screen
│   ├── models/
│   │   ├── bilstm.py               # Bidirectional LSTM with Bahdanau attention pooling
│   │   ├── stgcn.py                # 21-node hand skeleton ST-GCN with edge importance
│   │   └── tcn.py                  # Causal dilated temporal convolutional network
│   ├── training/
│   │   ├── evaluate.py             # Macro F1, confusion matrices, latency benchmark
│   │   ├── export_onnx.py          # PyTorch to ONNX export, checker & parity validator
│   │   └── train.py                # Multi-seed training engine for scenarios E1-E12
│   └── utils/                      # Seed setup, device auto-detect, path resolver
└── tests/
    ├── test_app_styles.py          # Base64 data URI camera viewport unit tests
    ├── test_pipeline.py            # Feature dimension & pipeline integrity tests
    └── test_virtual_mouse.py       # Virtual Mouse gesture & overlay unit tests
```

---

## Installation and Quickstart

### 1. Prerequisites
* Windows 10/11, macOS, or Linux.
* Python 3.10 to 3.12.
* A standard built-in or USB webcam.

### 2. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Gracias-48/Gesture-Control.git
cd Gesture-Control

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Real-Time Streamlit HUD
```bash
streamlit run app/main.py
```
* Open your browser at `http://localhost:8501`.
* Click **Start Camera Stream** to begin live inference (the Always-On-Top `"Gesture Controller"` floating window opens automatically in `hybrid` mode and can be toggled in the sidebar).
* *(Optional)* Toggle **Master Arm OS Control** in the sidebar to enable live keyboard/mouse injection into your operating system.

### 3b. Launch the Standalone Floating Screen (`Gesture Controller`)
Run just the Always-On-Top desktop floating camera window and Virtual Mouse at 30 FPS without Streamlit:
```bash
python gesture_controller.py

# Preview mode without dispatching OS mouse/keyboard events:
python gesture_controller.py --no-os

# Switch skeleton style or camera index:
python gesture_controller.py --style hud --camera 0
```

### 4. Interactive Gesture Data Recorder (CLI)
Record custom webcam sequences directly for fine-tuning or expanding classes:
```bash
# Record 25 repetitions for all 8 classes with 3.. 2.. 1.. visual countdown:
python -m src.data.record_gestures --reps 25 --build-splits

# Record a specific class only (e.g., Class 0: Swipe Left):
python -m src.data.record_gestures --class-id 0 --reps 20
```

### 5. Training & ONNX Export
```bash
# Train Causal TCN (E10) and BiLSTM (E2):
python -m src.training.train --exp E10 E2 --epochs 35

# Export best checkpoints to verified ONNX Runtime graphs:
python -m src.training.export_onnx --checkpoint runs/E10/seed42/best.pt
python -m src.training.export_onnx --checkpoint runs/E2/seed42/best.pt

# Run automated unit tests:
pytest -v
```
