"""Live MediaPipe Hands wrapper + HUD-style corner-bracket hitbox & directional gesture overlay.

COMP6826001 - Deep Learning Final Project
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from src.data.preprocessing import to_centered_coords
from src.models.stgcn import HAND_EDGES

FINGERTIPS = (4, 8, 12, 16, 20)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Color Palette (BGR)
INK = _bgr("#07080a")
BONE = _bgr("#f9f9f9")
CORAL = _bgr("#ff6363")
GREEN = _bgr("#30d158")
AMBER = _bgr("#ffd60a")
CYAN = _bgr("#56c2ff")
CARD_BG = (17, 18, 19)
BORDER_GRAY = (45, 48, 50)
TEXT_WHITE = (245, 245, 245)


class HandTracker:
    """MediaPipe Hands single-hand tracker with continuous swipe delta tracking."""

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        swipe_window: int = 8,
        swipe_pixel_threshold: float = 28.0,
    ):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.swipe_window = swipe_window
        self.swipe_threshold = swipe_pixel_threshold
        # Rolling history of (centroid_x, centroid_y) in display coordinates
        self.pos_history: deque = deque(maxlen=swipe_window)
        self.last_motion_state: str = "WAITING FOR HAND..."

    def process(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Processes frame through MediaPipe Hands. Returns (centered_coords, normalized_coords)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.pos_history.clear()
            self.last_motion_state = "WAITING FOR HAND..."
            return None, None
        norm = np.array([(p.x, p.y, p.z) for p in res.multi_hand_landmarks[0].landmark], np.float32)
        return to_centered_coords(norm, w, h), norm

    def update_motion_state(
        self,
        lm_norm: Optional[np.ndarray],
        frame_w: int,
        frame_h: int,
        mirrored: bool = True,
    ) -> tuple[str, Optional[tuple[int, int, int, int]], float]:
        """Compute bounding box (with ~20px padding), rolling delta X, and classify motion state."""
        if lm_norm is None:
            self.pos_history.clear()
            self.last_motion_state = "WAITING FOR HAND..."
            return self.last_motion_state, None, 0.0

        # Display pixel coordinates (mirrored if webcam feed is mirrored)
        if mirrored:
            xs = (1.0 - lm_norm[:, 0]) * frame_w
        else:
            xs = lm_norm[:, 0] * frame_w
        ys = lm_norm[:, 1] * frame_h

        # Dynamic bounding box with 20px padding
        pad = 20
        x_min = max(0, int(np.min(xs)) - pad)
        y_min = max(0, int(np.min(ys)) - pad)
        x_max = min(frame_w - 1, int(np.max(xs)) + pad)
        y_max = min(frame_h - 1, int(np.max(ys)) + pad)
        bbox = (x_min, y_min, x_max, y_max)

        # Centroid (wrist landmark 0 or mean of all points)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        self.pos_history.append((cx, cy))

        delta_x = 0.0
        if len(self.pos_history) >= 3:
            oldest_cx = self.pos_history[0][0]
            delta_x = cx - oldest_cx

        # Classify motion state based on rolling horizontal delta
        if delta_x < -self.swipe_threshold:
            state = "<< SWIPE LEFT"
        elif delta_x > self.swipe_threshold:
            state = "SWIPE RIGHT >>"
        else:
            state = "STAY / HOLD"

        self.last_motion_state = state
        return state, bbox, delta_x

    def close(self) -> None:
        self._hands.close()


def draw_hud_hand_overlay(
    frame: np.ndarray,
    lm_norm: Optional[np.ndarray],
    motion_state: str,
    bbox: Optional[tuple[int, int, int, int]],
    accent: str = "cyan",
    mirrored: bool = True,
    candidate_label: Optional[str] = None,
) -> np.ndarray:
    """Render skeletal bones, joints, corner-bracket hitbox, and floating HUD motion badge directly on frame."""
    h, w = frame.shape[:2]
    accent_color = {
        "cyan": CYAN,
        "green": GREEN,
        "amber": AMBER,
        "coral": CORAL,
    }.get(accent, CYAN)

    if lm_norm is None or bbox is None:
        # Floating badge when waiting for hand
        badge_text = "WAITING FOR HAND..."
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        bx = (w - tw) // 2
        by = 32
        cv2.rectangle(frame, (bx - 14, by - th - 7), (bx + tw + 14, by + 7), CARD_BG, -1)
        cv2.rectangle(frame, (bx - 14, by - th - 7), (bx + tw + 14, by + 7), BORDER_GRAY, 1)
        cv2.circle(frame, (bx - 4, by - th // 2), 3, (140, 140, 140), -1)
        cv2.putText(frame, badge_text, (bx + 6, by - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
        return frame

    x_min, y_min, x_max, y_max = bbox

    # 1. Landmark Points in display pixel coordinates
    if mirrored:
        pts = np.stack([(1.0 - lm_norm[:, 0]) * w, lm_norm[:, 1] * h], axis=1).astype(int)
    else:
        pts = np.stack([lm_norm[:, 0] * w, lm_norm[:, 1] * h], axis=1).astype(int)

    # 2. Draw Hand Bones (Outer high-contrast shadow + inner bone)
    for a, b in HAND_EDGES:
        p, q = tuple(pts[a]), tuple(pts[b])
        cv2.line(frame, p, q, INK, 5, cv2.LINE_AA)
        cv2.line(frame, p, q, BONE, 2, cv2.LINE_AA)

    # 3. Draw Joints
    for i, (x, y) in enumerate(pts):
        r = 6 if i in FINGERTIPS else (7 if i == 0 else 4)
        cv2.circle(frame, (x, y), r + 2, INK, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, accent_color, -1, cv2.LINE_AA)
        if i in FINGERTIPS or i == 0:
            cv2.circle(frame, (x, y), r + 2, BONE, 1, cv2.LINE_AA)

    # 4. Subtle Hairline Bounding Box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (35, 38, 42), 1)

    # 5. Cyber-HUD Corner Brackets (at all 4 corners)
    bracket_len = max(14, min(24, (x_max - x_min) // 5))
    thk = 3

    # Top-Left Corner Bracket
    cv2.line(frame, (x_min, y_min), (x_min + bracket_len, y_min), accent_color, thk, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_min), (x_min, y_min + bracket_len), accent_color, thk, cv2.LINE_AA)

    # Top-Right Corner Bracket
    cv2.line(frame, (x_max, y_min), (x_max - bracket_len, y_min), accent_color, thk, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max, y_min + bracket_len), accent_color, thk, cv2.LINE_AA)

    # Bottom-Left Corner Bracket
    cv2.line(frame, (x_min, y_max), (x_min + bracket_len, y_max), accent_color, thk, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_max), (x_min, y_max - bracket_len), accent_color, thk, cv2.LINE_AA)

    # Bottom-Right Corner Bracket
    cv2.line(frame, (x_max, y_max), (x_max - bracket_len, y_max), accent_color, thk, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_max), (x_max - bracket_len, y_max), accent_color, thk, cv2.LINE_AA)

    # 6. Directional Motion Badge Directly Above Hand Hitbox
    if "SWIPE LEFT" in motion_state:
        state_color = CYAN
        badge_border = CYAN
    elif "SWIPE RIGHT" in motion_state:
        state_color = GREEN
        badge_border = GREEN
    elif "STAY" in motion_state:
        state_color = AMBER
        badge_border = BORDER_GRAY
    else:
        state_color = (180, 180, 180)
        badge_border = BORDER_GRAY

    badge_label = motion_state
    if candidate_label:
        badge_label = f"{motion_state}  |  {candidate_label}"

    (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    bx1 = max(10, min(w - tw - 30, x_min))
    by1 = max(th + 14, y_min - 10)
    bx2 = min(w - 10, bx1 + tw + 22)

    # Badge background capsule
    cv2.rectangle(frame, (bx1, by1 - th - 8), (bx2, by1 + 5), CARD_BG, -1)
    cv2.rectangle(frame, (bx1, by1 - th - 8), (bx2, by1 + 5), badge_border, 1)
    # Status indicator LED dot
    cv2.circle(frame, (bx1 + 10, by1 - th // 2 - 2), 4, state_color, -1, cv2.LINE_AA)
    # Badge text
    cv2.putText(frame, badge_label, (bx1 + 20, by1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_WHITE, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame: np.ndarray, lm_norm: Optional[np.ndarray], accent: str = "coral") -> np.ndarray:
    """Backwards-compatible skeleton renderer."""
    return draw_hud_hand_overlay(frame, lm_norm, "STAY / HOLD", None, accent=accent, mirrored=False)
