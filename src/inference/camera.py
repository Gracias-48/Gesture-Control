"""Camera producer thread + timestamp resampler (TRD §4, Thread 1)."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np


class TimestampResampler:
    """Accept frames so the accepted stream averages ``target_fps`` (Δt ≈ 83.3 ms @ 12 FPS).

    The next deadline advances by exactly one period (drift-free); if the
    source stalls for > 2 periods the schedule re-anchors to the current frame.
    """

    def __init__(self, target_fps: float = 12.0, tolerance_s: float = 0.004):
        self.period = 1.0 / float(target_fps)
        self.tol = tolerance_s
        self._next: Optional[float] = None

    def reset(self) -> None:
        self._next = None

    def accept(self, ts: float) -> bool:
        if self._next is None or ts - self._next > 2 * self.period:
            self._next = ts + self.period
            return True
        if ts + self.tol >= self._next:
            self._next += self.period
            return True
        return False


class CameraStream:
    """Non-blocking OpenCV capture on a daemon thread; always exposes the latest frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0
        self.capture_fps = 0.0
        self.error: Optional[str] = None

    def start(self) -> "CameraStream":
        import cv2

        candidate_indices = list(dict.fromkeys([self.index, 0, 1, 2]))
        if sys.platform.startswith("win"):
            attempts = [(idx, cv2.CAP_DSHOW) for idx in candidate_indices] + [
                (idx, cv2.CAP_ANY) for idx in list(dict.fromkeys([self.index, 0, 1]))
            ]
        else:
            attempts = [(idx, cv2.CAP_ANY) for idx in candidate_indices]

        cap = None
        warm_frame = None
        for test_idx, backend in attempts:
            c = cv2.VideoCapture(test_idx, backend)
            if not c.isOpened():
                c.release()
                continue
            c.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            c.set(cv2.CAP_PROP_FPS, self.fps)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok = False
            frame = None
            for _ in range(3):
                ok, frame = c.read()
                if ok and frame is not None and frame.size > 0:
                    break
                time.sleep(0.03)

            if ok and frame is not None and frame.size > 0:
                cap = c
                warm_frame = frame
                self.index = test_idx
                break
            c.release()

        if cap is None:
            tried = ", ".join(str(i) for i in candidate_indices)
            self.error = f"Cannot open or read frames from camera (tried indices: {tried})"
            print(f"[CameraStream] {self.error}")
            return self

        self._cap = cap
        self._frame = warm_frame
        self._ts = time.perf_counter()
        self._seq = 1 if warm_frame is not None else 0
        self._running = True
        self.error = None
        self._thread = threading.Thread(target=self._loop, name="camera-producer", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        last = None
        fail_count = 0
        while self._running:
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            now = time.perf_counter()
            if not ok or frame is None or frame.size == 0:
                fail_count += 1
                self.error = f"Camera {self.index} read failed ({fail_count}x)"
                time.sleep(0.015)
                continue
            fail_count = 0
            with self._lock:
                self._frame, self._ts = frame, now
                self._seq += 1
                self.error = None
            if last is not None:
                inst = 1.0 / max(now - last, 1e-6)
                self.capture_fps = inst if self.capture_fps == 0 else 0.9 * self.capture_fps + 0.1 * inst
            last = now

    def read(self) -> tuple[int, Optional[np.ndarray], float]:
        """Return ``(sequence_id, frame_bgr, timestamp)`` of the most recent frame."""
        with self._lock:
            return self._seq, self._frame.copy() if self._frame is not None else None, self._ts

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
