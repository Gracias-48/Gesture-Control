"""Real-Time Inference Engine (TRD §4, APP_FLOW §1-2).

Menyatukan:
- Thread 1: CameraStream (capture ~30 FPS -> resample ke 12 FPS via TimestampResampler)
- HandTracker (MediaPipe Hands 21 landmarks + draw skeleton)
- Ring Buffer: collections.deque(maxlen=32) dengan penanganan missing frame & interpolasi linear
- Motion Gate: deteksi pergerakan minimum pergelangan/sendi
- ONNX Runtime Inference Session: inferensi setiap stride = 4 frame
- Post-processing Gate: confidence threshold, 3-window consecutive vote, cooldown timer
- OS Action Controller: PyAutoGUI execution dengan safety fail-safe toggle
- Telemetri Real-Time Thread-Safe untuk Streamlit HUD
"""
from __future__ import annotations

import collections
import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from src.data.preprocessing import (
    FeaturePipeline,
    Scaler,
    interpolate_missing,
    mean_joint_speed,
)
from src.inference.camera import CameraStream, TimestampResampler
from src.inference.controller import ActionResult, OSActionController
from src.inference.gate import GateDecision, PostProcessGate, SystemState
from src.inference.hand_tracker import HandTracker, draw_hud_hand_overlay, draw_skeleton
from src.inference.virtual_mouse import (
    FloatingScreenWindow,
    HLabel,
    HandRecog,
    VirtualMouseController,
    draw_classic_mediapipe_overlay,
    gesture_to_label,
)
from src.utils.config import load_config, resolve_path


@dataclass
class TelemetryData:
    state: SystemState = SystemState.STANDBY
    active_model: str = "TCN"
    fps: float = 0.0
    inference_latency_ms: float = 0.0
    buffer_fill: int = 0
    buffer_max: int = 32
    consistency_count: int = 0
    consistency_target: int = 3
    cooldown_remaining: float = 0.0
    cooldown_target: float = 0.9
    motion_speed: float = 0.0
    motion_gate_passed: bool = False
    confidence_threshold: float = 0.80
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    class_names: List[str] = field(default_factory=list)
    predicted_class: Optional[int] = None
    predicted_confidence: float = 0.0
    gate_reason: str = "STANDBY"
    last_action: Optional[ActionResult] = None
    os_armed: bool = False
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    toast_message: Optional[str] = None
    toast_expiry: float = 0.0
    control_mode: str = "hybrid"
    floating_window: bool = True
    overlay_style: str = "classic"
    vm_gesture_name: str = "None"
    cursor_pos: Optional[Tuple[int, int]] = None
    error_message: Optional[str] = None


class GestureInferenceEngine:
    """Mesin inferensi real-time kontinu touchless gesture control."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_config()
        self._lock = threading.Lock()
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Konfigurasi parameter
        self.target_fps = float(self.cfg["inference"].get("target_fps", 12.0))
        self.buffer_size = int(self.cfg["inference"].get("buffer_size", 32))
        self.stride = int(self.cfg["inference"].get("stride", 4))
        self.mirror_display = bool(self.cfg["inference"].get("mirror_display", True))
        self.hand_absent_reset_s = float(self.cfg["postprocessing"].get("hand_absent_reset_s", 1.0))
        self.max_missing_ratio = float(self.cfg["postprocessing"].get("max_missing_ratio", 0.30))

        # Modul sub-komponen
        self.camera: Optional[CameraStream] = None
        self.resampler = TimestampResampler(target_fps=self.target_fps)
        self.hand_tracker: Optional[HandTracker] = None
        self.gate: Optional[PostProcessGate] = None
        self.os_controller: Optional[OSActionController] = None
        self.hand_recog: Optional[HandRecog] = None
        self.vm_controller: Optional[VirtualMouseController] = None
        self.floating_window: Optional[FloatingScreenWindow] = None

        self.control_mode: str = str(self.cfg["inference"].get("control_mode", "hybrid"))
        self.floating_window_enabled: bool = bool(self.cfg["inference"].get("floating_window", True))
        self.overlay_style: str = str(self.cfg["inference"].get("overlay_style", "classic"))
        self.camera_index: int = int(self.cfg["inference"].get("camera_index", 0))
        self.last_error: Optional[str] = None

        # Model ONNX & Session
        self.active_model_name: str = self.cfg["inference"].get("default_model", "tcn").lower()
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.model_metadata: Dict[str, Any] = {}
        self.pipeline: Optional[FeaturePipeline] = None
        self.class_names_list: List[str] = list(self.cfg["classes"]["names"])

        # Buffers dan Pelacak Status
        self.raw_landmark_buffer: collections.deque = collections.deque(maxlen=self.buffer_size)
        self.frames_since_last_infer: int = 0
        self.last_hand_seen_time: float = 0.0
        self.last_processed_frame: Optional[np.ndarray] = None
        self.last_processed_seq: int = -1

        # Telemetri data
        self.telemetry = TelemetryData(
            buffer_max=self.buffer_size,
            class_names=self.class_names_list,
            cooldown_target=float(self.cfg["postprocessing"]["cooldown_s"]),
            confidence_threshold=float(self.cfg["postprocessing"]["confidence_threshold"]),
            consistency_target=int(self.cfg["postprocessing"]["consecutive_windows"]),
            control_mode=self.control_mode,
            floating_window=self.floating_window_enabled,
            overlay_style=self.overlay_style,
        )

        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Inisialisasi komponen gate, os controller, virtual mouse, dan model ONNX."""
        from src.inference.gate import GateConfig

        gate_cfg = GateConfig.from_cfg(self.cfg)
        self.gate = PostProcessGate(gate_cfg, background_id=int(self.cfg["classes"]["background_id"]))

        act_cfg = self.cfg.get("actions", {})
        self.os_controller = OSActionController(
            mapping=act_cfg.get("mapping", {}),
            failsafe=bool(act_cfg.get("failsafe", True)),
            armed=bool(act_cfg.get("start_armed", False)),
        )
        self.hand_recog = HandRecog(hand_label=HLabel.MAJOR)
        self.vm_controller = VirtualMouseController(
            failsafe=bool(act_cfg.get("failsafe", True)),
        )
        self.floating_window = FloatingScreenWindow(
            window_name="Gesture Controller",
            width=440,
            height=330,
        )

        self.load_model(self.active_model_name)

    def load_model(self, model_name: str) -> bool:
        """Hot-swap model ONNX (bilstm, stgcn, tcn) tanpa mematikan thread kamera."""
        with self._lock:
            model_name = model_name.lower()
            onnx_dir = resolve_path(self.cfg, "onnx_dir")
            onnx_path = onnx_dir / f"{model_name}.onnx"
            meta_path = onnx_dir / f"{model_name}.json"
            scaler_path = onnx_dir / f"{model_name}_scaler.npz"

            if not onnx_path.exists():
                print(f"[Engine] File model {onnx_path} belum ditemukan. Harap ekspor model terlebih dahulu.")
                return False

            try:
                # Load metadata sidecar jika ada
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.model_metadata = json.load(f)
                else:
                    self.model_metadata = {
                        "model": model_name,
                        "display_name": model_name.upper(),
                        "kind": "sequence" if model_name in ("bilstm", "tcn") else "graph",
                        "use_global": True,
                        "class_names": self.class_names_list,
                    }

                # Load Scaler
                if scaler_path.exists():
                    scaler = Scaler.load(scaler_path)
                else:
                    # fallback ke global scaler_stats.npz
                    g_scaler_path = resolve_path(self.cfg, "scaler_stats")
                    if g_scaler_path.exists():
                        scaler = Scaler.load(g_scaler_path)
                    else:
                        scaler = Scaler.identity()

                kind = self.model_metadata.get("kind", "sequence" if model_name in ("bilstm", "tcn") else "graph")
                use_global = bool(self.model_metadata.get("use_global", True))
                self.pipeline = FeaturePipeline(
                    kind=kind,
                    scaler=scaler,
                    use_global=use_global,
                    eps=float(self.cfg["features"]["eps"]),
                )

                # Siapkan ONNX session
                opts = ort.SessionOptions()
                threads = int(self.cfg["inference"].get("onnx_intra_op_threads", 2))
                opts.intra_op_num_threads = threads
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )

                self.active_model_name = model_name
                self.telemetry.active_model = self.model_metadata.get("display_name", model_name.upper())
                if self.gate:
                    self.gate.reset()
                self.raw_landmark_buffer.clear()
                self.frames_since_last_infer = 0
                print(f"[Engine] Berhasil memuat model: {model_name.upper()} ({kind})")
                return True
            except Exception as e:
                print(f"[Engine] Gagal memuat model {model_name}: {e}")
                return False

    def start(self) -> None:
        """Mulai pipeline inferensi real-time."""
        if self.running:
            return

        self.last_error = None
        self.last_processed_frame = None
        self.last_processed_seq = -1
        self.frames_since_last_infer = 0
        self.raw_landmark_buffer.clear()
        self.resampler.reset()
        with self._lock:
            self.telemetry.error_message = None

        cam_cfg = self.cfg["inference"]
        self.camera = CameraStream(
            index=self.camera_index,
            width=int(cam_cfg.get("capture_width", 640)),
            height=int(cam_cfg.get("capture_height", 480)),
            fps=int(cam_cfg.get("capture_fps", 30)),
        ).start()

        if self.camera.error:
            self.last_error = self.camera.error
            with self._lock:
                self.telemetry.error_message = self.last_error
                self.telemetry.state = SystemState.STANDBY
            self.camera.stop()
            self.camera = None
            self.running = False
            print(f"[Engine] Gagal memulai kamera: {self.last_error}")
            return

        # Sinkronkan indeks kamera aktual jika CameraStream melakukan fallback (mis. 0 -> 1)
        self.camera_index = self.camera.index

        mp_cfg = cam_cfg.get("mediapipe", {})
        try:
            self.hand_tracker = HandTracker(
                model_complexity=int(mp_cfg.get("model_complexity", 0)),
                min_detection_confidence=float(mp_cfg.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
            )
        except Exception as exc:
            self.last_error = f"HandTracker init failed: {type(exc).__name__}: {exc}"
            with self._lock:
                self.telemetry.error_message = self.last_error
                self.telemetry.state = SystemState.STANDBY
            self.camera.stop()
            self.camera = None
            self.running = False
            print(f"[Engine] {self.last_error}")
            return

        self.running = True
        self._worker_thread = threading.Thread(target=self._inference_loop, name="inference-consumer", daemon=True)
        self._worker_thread.start()
        # Tunggu sebentar agar frame pertama siap sebelum UI rerun
        for _ in range(16):
            if self.last_processed_frame is not None or not self.running:
                break
            time.sleep(0.05)
        print("[Engine] Real-time inference engine started.")

    def stop(self) -> None:
        """Hentikan pipeline inferensi."""
        self.running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        self._worker_thread = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.hand_tracker:
            self.hand_tracker.close()
            self.hand_tracker = None
        self.last_processed_frame = None
        self.last_processed_seq = -1
        with self._lock:
            self.telemetry.state = SystemState.STANDBY
        print("[Engine] Real-time inference engine stopped.")

    def set_os_control(self, armed: bool) -> None:
        if self.os_controller:
            self.os_controller.set_armed(armed)
            with self._lock:
                self.telemetry.os_armed = self.os_controller.armed

    def set_confidence_threshold(self, threshold: float) -> None:
        if self.gate:
            self.gate.set_threshold(threshold)
            with self._lock:
                self.telemetry.confidence_threshold = threshold

    def set_cooldown(self, cooldown_s: float) -> None:
        if self.gate:
            self.gate.set_cooldown(cooldown_s)
            with self._lock:
                self.telemetry.cooldown_target = cooldown_s

    def set_control_mode(self, mode: str) -> None:
        if mode not in ("hybrid", "virtual_mouse", "ai_gestures"):
            return
        self.control_mode = mode
        with self._lock:
            self.telemetry.control_mode = mode

    def set_floating_window(self, enabled: bool) -> None:
        self.floating_window_enabled = bool(enabled)
        with self._lock:
            self.telemetry.floating_window = self.floating_window_enabled

    def set_overlay_style(self, style: str) -> None:
        if style not in ("classic", "cyber_hud"):
            return
        self.overlay_style = style
        with self._lock:
            self.telemetry.overlay_style = style

    def set_camera_index(self, index: int) -> None:
        idx = int(index)
        if idx == self.camera_index:
            return
        self.camera_index = idx
        if self.running:
            self.stop()
            self.start()

    def _inference_loop(self) -> None:
        """Thread consumer yang menjalankan MediaPipe, Virtual Mouse, Floating Screen, dan ONNX."""
        import cv2

        fps_tracker_time = time.perf_counter()
        fps_frame_count = 0
        computed_fps = 0.0
        last_vm_log_event: Optional[str] = None
        last_vm_log_time = 0.0

        try:
            while self.running:
                cam = self.camera
                tracker = self.hand_tracker
                if not cam or not tracker:
                    time.sleep(0.01)
                    continue

                seq, frame, ts = cam.read()
                if frame is None or seq == self.last_processed_seq:
                    time.sleep(0.005)
                    continue

                self.last_processed_seq = seq
                now = time.perf_counter()

                fps_frame_count += 1
                if now - fps_tracker_time >= 1.0:
                    computed_fps = fps_frame_count / (now - fps_tracker_time)
                    fps_frame_count = 0
                    fps_tracker_time = now

                # 1. Deteksi MediaPipe Hands @ ~30 FPS
                lm_centered, lm_norm = tracker.process(frame)
                if not self.running:
                    break
                h, w = frame.shape[:2]

                # Mirror frame terlebih dahulu jika mirror_display aktif
                display_frame = cv2.flip(frame, 1) if self.mirror_display else frame.copy()

                # 2. Virtual Mouse (Viral-Doshi) @ 30 FPS penuh
                vm_gesture_name = "Disabled"
                cursor_pos = self.vm_controller.last_cursor_pos if self.vm_controller else None
                os_armed_now = bool(self.os_controller and self.os_controller.armed)
                if self.control_mode in ("hybrid", "virtual_mouse") and self.hand_recog and self.vm_controller:
                    self.hand_recog.update_hand_result(lm_norm)
                    vm_gesture = self.hand_recog.get_gesture()
                    vm_gesture_name = gesture_to_label(vm_gesture)
                    vm_event = self.vm_controller.handle_controls(
                        gesture=vm_gesture,
                        lm_norm=lm_norm,
                        armed=os_armed_now,
                        mirrored=self.mirror_display,
                    )
                    cursor_pos = self.vm_controller.last_cursor_pos
                    if vm_event:
                        discrete_events = {"Left Click", "Right Click", "Double Click", "Drag Start", "Drag Release"}
                        should_log = (
                            vm_event in discrete_events
                            or vm_event != last_vm_log_event
                            or (now - last_vm_log_time) >= 0.45
                        )
                        if should_log:
                            last_vm_log_event = vm_event
                            last_vm_log_time = now
                            toast_exp = now + float(self.cfg["postprocessing"].get("toast_display_s", 1.2))
                            with self._lock:
                                self.telemetry.toast_message = f"{vm_event} (Virtual Mouse)"
                                self.telemetry.toast_expiry = toast_exp
                                self.telemetry.action_history.insert(0, {
                                    "timestamp": time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}",
                                    "gesture": f"VM: {vm_gesture_name}",
                                    "action": vm_event,
                                    "shortcut": "Mouse/HID",
                                    "confidence": "100.0%",
                                    "latency_ms": "0.0ms",
                                    "executed": os_armed_now,
                                })
                                if len(self.telemetry.action_history) > 10:
                                    self.telemetry.action_history.pop()

                # 3. Tentukan status temporal AI & render visual skeleton
                buf_len = len(self.raw_landmark_buffer)
                current_state = (
                    self.gate.phase(now)
                    if (self.control_mode in ("hybrid", "ai_gestures") and buf_len == self.buffer_size)
                    else (
                        SystemState.MONITORING
                        if (self.control_mode == "virtual_mouse" and lm_norm is not None)
                        else (SystemState.BUFFERING if lm_norm is not None else SystemState.STANDBY)
                    )
                )

                motion_state, bbox, _ = tracker.update_motion_state(
                    lm_norm, w, h, mirrored=self.mirror_display
                )
                if current_state == SystemState.TRIGGERED:
                    accent = "coral"
                elif current_state == SystemState.COOLDOWN:
                    accent = "amber"
                elif "SWIPE" in motion_state:
                    accent = "green"
                else:
                    accent = "cyan"

                cand_label = None
                if self.control_mode in ("hybrid", "ai_gestures"):
                    if self.telemetry.predicted_class is not None and self.telemetry.predicted_class != int(self.cfg["classes"]["background_id"]):
                        cname = self.class_names_list[self.telemetry.predicted_class]
                        cand_label = f"{cname} ({self.telemetry.predicted_confidence * 100:.0f}%)"
                elif lm_norm is not None:
                    cand_label = vm_gesture_name

                if self.overlay_style == "classic":
                    overlay_label = (
                        vm_gesture_name
                        if (self.control_mode in ("hybrid", "virtual_mouse") and lm_norm is not None)
                        else cand_label
                    )
                    draw_classic_mediapipe_overlay(
                        display_frame,
                        lm_norm,
                        mirrored=self.mirror_display,
                        gesture_label=overlay_label,
                        cursor_xy=cursor_pos,
                    )
                else:
                    draw_hud_hand_overlay(
                        display_frame,
                        lm_norm,
                        motion_state=motion_state,
                        bbox=bbox,
                        accent=accent,
                        mirrored=self.mirror_display,
                        candidate_label=cand_label,
                    )

                # 4. Floating Screen Window ("Gesture Controller") pada thread yang sama (diisolasi agar tidak memblokir HUD)
                try:
                    if self.floating_window_enabled and self.floating_window is not None:
                        keep_open = self.floating_window.show(display_frame)
                        if not keep_open:
                            self.floating_window_enabled = False
                            self.floating_window.close()
                            with self._lock:
                                self.telemetry.floating_window = False
                    elif self.floating_window is not None and self.floating_window.is_open:
                        self.floating_window.close()
                except Exception as fw_exc:
                    self.floating_window_enabled = False
                    print(f"[Engine] Floating window disabled after error: {fw_exc}")
                    with self._lock:
                        self.telemetry.floating_window = False

                # 5. Timestamp Downsampling ke 12 FPS untuk Buffer & ONNX AI Gestures
                probs = self.telemetry.probabilities
                pred_idx = self.telemetry.predicted_class
                pred_conf = self.telemetry.predicted_confidence
                lat_ms = self.telemetry.inference_latency_ms
                motion_speed = self.telemetry.motion_speed
                motion_ok = self.telemetry.motion_gate_passed

                if self.control_mode in ("hybrid", "ai_gestures") and self.resampler.accept(now):
                    if lm_centered is not None:
                        self.last_hand_seen_time = now
                        self.raw_landmark_buffer.append(lm_centered)
                    else:
                        if now - self.last_hand_seen_time > self.hand_absent_reset_s:
                            self.raw_landmark_buffer.clear()
                            self.gate.reset()
                            current_state = SystemState.STANDBY
                        else:
                            self.raw_landmark_buffer.append(np.full((21, 3), np.nan, dtype=np.float32))

                    buf_len = len(self.raw_landmark_buffer)
                    self.frames_since_last_infer += 1

                    if buf_len == 0:
                        current_state = SystemState.STANDBY
                    elif buf_len < self.buffer_size:
                        current_state = SystemState.BUFFERING
                    else:
                        current_state = self.gate.phase(now)

                    if buf_len == self.buffer_size and self.frames_since_last_infer >= self.stride:
                        self.frames_since_last_infer = 0
                        buf_array = np.array(self.raw_landmark_buffer, dtype=np.float32)

                        filled_seq, missing_ratio = interpolate_missing(buf_array)

                        if filled_seq is not None and missing_ratio <= self.max_missing_ratio:
                            motion_speed = mean_joint_speed(filled_seq)
                            motion_ok = motion_speed >= self.gate.cfg.motion_threshold

                            if self.onnx_session is not None and self.pipeline is not None:
                                t_infer_start = time.perf_counter()
                                x_feat = self.pipeline(filled_seq)
                                x_feat = np.expand_dims(x_feat, axis=0)

                                input_name = self.onnx_session.get_inputs()[0].name
                                outputs = self.onnx_session.run(None, {input_name: x_feat})
                                probs = outputs[0][0]
                                lat_ms = (time.perf_counter() - t_infer_start) * 1e3
                                pred_idx = int(np.argmax(probs))
                                pred_conf = float(probs[pred_idx])

                            decision: GateDecision = self.gate.update(probs, motion_ok, now)

                            if decision.fire and decision.class_id is not None:
                                action_res = self.os_controller.execute(decision.class_id)
                                current_state = SystemState.TRIGGERED
                                toast_msg = f"{action_res.label} ({action_res.shortcut})"
                                toast_exp = now + float(self.cfg["postprocessing"].get("toast_display_s", 1.6))

                                with self._lock:
                                    self.telemetry.last_action = action_res
                                    self.telemetry.toast_message = toast_msg
                                    self.telemetry.toast_expiry = toast_exp
                                    self.telemetry.action_history.insert(0, {
                                        "timestamp": time.strftime("%H:%M:%S") + f".{int((now % 1) * 1000):03d}",
                                        "gesture": self.class_names_list[decision.class_id],
                                        "action": action_res.label,
                                        "shortcut": action_res.shortcut,
                                        "confidence": f"{decision.confidence * 100:.1f}%",
                                        "latency_ms": f"{lat_ms:.1f}ms",
                                        "executed": action_res.executed,
                                    })
                                    if len(self.telemetry.action_history) > 10:
                                        self.telemetry.action_history.pop()

                # 6. Perbarui Snapshot Telemetri Thread-Safe
                gate_reason_str = (
                    "VIRTUAL_MOUSE"
                    if self.control_mode == "virtual_mouse"
                    else (
                        self.gate.last_decision.reason
                        if self.gate.last_decision
                        else ("STANDBY" if buf_len == 0 else ("BUFFERING" if buf_len < self.buffer_size else "MONITORING"))
                    )
                )
                with self._lock:
                    self.last_processed_frame = display_frame
                    self.telemetry.state = current_state
                    self.telemetry.fps = computed_fps
                    self.telemetry.inference_latency_ms = lat_ms
                    self.telemetry.buffer_fill = buf_len
                    self.telemetry.motion_speed = motion_speed
                    self.telemetry.motion_gate_passed = motion_ok
                    self.telemetry.probabilities = probs
                    self.telemetry.predicted_class = pred_idx
                    self.telemetry.predicted_confidence = pred_conf
                    self.telemetry.gate_reason = gate_reason_str
                    self.telemetry.cooldown_remaining = self.gate.cooldown_remaining(now)
                    self.telemetry.consistency_count = (
                        self.gate.last_decision.consistency if self.gate.last_decision else 0
                    )
                    self.telemetry.os_armed = os_armed_now
                    self.telemetry.control_mode = self.control_mode
                    self.telemetry.floating_window = self.floating_window_enabled
                    self.telemetry.overlay_style = self.overlay_style
                    self.telemetry.vm_gesture_name = vm_gesture_name
                    self.telemetry.cursor_pos = cursor_pos
                    self.telemetry.error_message = self.last_error
        except Exception as exc:
            traceback.print_exc()
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.running = False
            with self._lock:
                self.telemetry.error_message = self.last_error
                self.telemetry.state = SystemState.STANDBY
        finally:
            if self.floating_window is not None:
                try:
                    self.floating_window.close()
                except Exception:
                    pass

    def get_latest_frame_and_telemetry(self) -> Tuple[Optional[np.ndarray], TelemetryData]:
        """Ambil salinan frame terbaru dan telemetri untuk di-render oleh UI Streamlit."""
        with self._lock:
            frame = self.last_processed_frame.copy() if self.last_processed_frame is not None else None
            # Deepcopy data sederhana
            tele = TelemetryData(
                state=self.telemetry.state,
                active_model=self.telemetry.active_model,
                fps=self.telemetry.fps,
                inference_latency_ms=self.telemetry.inference_latency_ms,
                buffer_fill=self.telemetry.buffer_fill,
                buffer_max=self.telemetry.buffer_max,
                consistency_count=self.telemetry.consistency_count,
                consistency_target=self.telemetry.consistency_target,
                cooldown_remaining=self.telemetry.cooldown_remaining,
                cooldown_target=self.telemetry.cooldown_target,
                motion_speed=self.telemetry.motion_speed,
                motion_gate_passed=self.telemetry.motion_gate_passed,
                confidence_threshold=self.telemetry.confidence_threshold,
                probabilities=self.telemetry.probabilities.copy(),
                class_names=list(self.telemetry.class_names),
                predicted_class=self.telemetry.predicted_class,
                predicted_confidence=self.telemetry.predicted_confidence,
                gate_reason=self.telemetry.gate_reason,
                last_action=self.telemetry.last_action,
                os_armed=self.telemetry.os_armed,
                action_history=list(self.telemetry.action_history),
                toast_message=self.telemetry.toast_message if time.perf_counter() < self.telemetry.toast_expiry else None,
                toast_expiry=self.telemetry.toast_expiry,
                control_mode=self.telemetry.control_mode,
                floating_window=self.telemetry.floating_window,
                overlay_style=self.telemetry.overlay_style,
                vm_gesture_name=self.telemetry.vm_gesture_name,
                cursor_pos=self.telemetry.cursor_pos,
                error_message=self.telemetry.error_message,
            )
            return frame, tele
