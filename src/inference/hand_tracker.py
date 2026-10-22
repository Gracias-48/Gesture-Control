"""Live MediaPipe Hands wrapper + high-contrast Raycast-style skeleton overlay."""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.data.preprocessing import to_centered_coords
from src.models.stgcn import HAND_EDGES

FINGERTIPS = (4, 8, 12, 16, 20)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


INK = _bgr("#07080a")
BONE = _bgr("#f9f9f9")
CORAL = _bgr("#ff6363")
GREEN = _bgr("#30d158")
AMBER = _bgr("#ffd60a")


class HandTracker:
    """MediaPipe Hands in video mode (single hand). ``process`` returns two views of the landmarks:

    * ``centered`` — aspect-corrected, centre-origin coords (model input space)
    * ``normalized`` — raw MediaPipe [0, 1] coords (for drawing)
    """

    def __init__(self, model_complexity: int = 0, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        norm = np.array([(p.x, p.y, p.z) for p in res.multi_hand_landmarks[0].landmark], np.float32)
        return to_centered_coords(norm, w, h), norm

    def close(self) -> None:
        self._hands.close()


def draw_skeleton(frame: np.ndarray, lm_norm: Optional[np.ndarray], accent: str = "coral") -> np.ndarray:
    """Draw bones (white on ink outline) and joints (accent dots) in-place; returns frame."""
    import cv2

    if lm_norm is None:
        return frame
    h, w = frame.shape[:2]
    pts = np.stack([lm_norm[:, 0] * w, lm_norm[:, 1] * h], axis=1).astype(int)
    joint = {"coral": CORAL, "green": GREEN, "amber": AMBER}.get(accent, CORAL)
    for a, b in HAND_EDGES:
        p, q = tuple(pts[a]), tuple(pts[b])
        cv2.line(frame, p, q, INK, 6, cv2.LINE_AA)
        cv2.line(frame, p, q, BONE, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 6 if i in FINGERTIPS else (7 if i == 0 else 4)
        cv2.circle(frame, (x, y), r + 2, INK, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, joint, -1, cv2.LINE_AA)
        if i in FINGERTIPS or i == 0:
            cv2.circle(frame, (x, y), r + 2, BONE, 1, cv2.LINE_AA)
    return frame
