"""Virtual Mouse Controller & Always-On-Top Floating Screen ("Gesture Controller").

Inspired by Viral-Doshi/Gesture-Controlled-Virtual-Mouse, integrated seamlessly with
the Touchless Gesture Control HUD pipeline:
- MediaPipe landmark-based Virtual Mouse (V-Gesture cursor movement, Left/Right/Double Click,
  Fist drag & drop, Thumb-Index pinch volume/zoom/scroll)
- Native Always-On-Top floating OpenCV window ("Gesture Controller") docked at top-right
- Classic green-connections / red-dots MediaPipe hand skeleton renderer
"""
from __future__ import annotations

import math
import sys
import time
from enum import IntEnum
from typing import Optional, Tuple

import cv2
import numpy as np

from src.models.stgcn import HAND_EDGES


class Gest(IntEnum):
    """Binary-encoded hand gesture states matching Viral-Doshi/Gesture-Controlled-Virtual-Mouse."""

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31

    # Composite / contextual gestures
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36


class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


GESTURE_LABELS: dict[int, str] = {
    Gest.FIST: "FIST: DRAG & DROP",
    Gest.PALM: "PALM: NEUTRAL",
    Gest.LAST4: "PALM: NEUTRAL",
    Gest.V_GEST: "V-GEST: MOVE CURSOR",
    Gest.MID: "LEFT CLICK",
    Gest.INDEX: "RIGHT CLICK",
    Gest.TWO_FINGER_CLOSED: "DOUBLE CLICK",
    Gest.PINCH_MAJOR: "PINCH: VOLUME / ZOOM",
    Gest.PINCH_MINOR: "PINCH: SCROLL",
}


def gesture_to_label(gesture: int) -> str:
    """Return human-readable label for a Gest enum value."""
    return GESTURE_LABELS.get(int(gesture), f"GESTURE ({int(gesture)})")


def _get_screen_size() -> tuple[int, int]:
    """Return primary monitor (width, height) safely across platforms."""
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass
    try:
        import pyautogui

        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


class HandRecog:
    """Stateful single-hand finger-bitmask & Virtual Mouse gesture recognizer."""

    def __init__(self, hand_label: int = HLabel.MAJOR):
        self.finger: int = 0
        self.ori_gesture: int = Gest.PALM
        self.prev_gesture: int = Gest.PALM
        self.frame_count: int = 0
        self.hand_result: Optional[np.ndarray] = None
        self.hand_label: int = int(hand_label)

    def update_hand_result(
        self,
        lm_norm: Optional[np.ndarray],
        hand_label: Optional[int] = None,
    ) -> None:
        """Update with `(21, 3)` normalized MediaPipe landmarks (`[0..1]`)."""
        self.hand_result = lm_norm
        if hand_label is not None:
            self.hand_label = int(hand_label)

    def get_signed_dist(self, points: list[int]) -> float:
        """Signed Euclidean distance in (x, y) where sign reflects vertical ordering."""
        if self.hand_result is None:
            return 0.0
        p0, p1 = self.hand_result[points[0]], self.hand_result[points[1]]
        sign = -1.0 if p0[1] < p1[1] else 1.0
        dist = math.hypot(float(p0[0] - p1[0]), float(p0[1] - p1[1]))
        return dist * sign

    def get_dist(self, points: list[int]) -> float:
        """2D Euclidean distance between two normalized landmarks."""
        if self.hand_result is None:
            return 0.0
        p0, p1 = self.hand_result[points[0]], self.hand_result[points[1]]
        return math.hypot(float(p0[0] - p1[0]), float(p0[1] - p1[1]))

    def get_dz(self, points: list[int]) -> float:
        """Absolute depth difference between two normalized landmarks."""
        if self.hand_result is None:
            return 0.0
        return abs(float(self.hand_result[points[0]][2] - self.hand_result[points[1]][2]))

    def set_finger_state(self) -> None:
        """Compute bitmask of extended fingers: Index(8), Mid(4), Ring(2), Pinky(1), Thumb(16)."""
        if self.hand_result is None:
            self.finger = 0
            return

        # Tip, PIP, MCP, Wrist chains for the 4 non-thumb fingers
        chains = [
            (8, 6, 5, 0),    # Index
            (12, 10, 9, 0),  # Middle
            (16, 14, 13, 0), # Ring
            (20, 18, 17, 0), # Pinky
        ]
        self.finger = 0
        for tip, pip, mcp, wrist in chains:
            self.finger = self.finger << 1
            dist1 = self.get_signed_dist([tip, mcp])
            dist2 = self.get_signed_dist([mcp, wrist])
            ratio = (dist1 / dist2) if abs(dist2) > 1e-6 else 0.0
            tip_above_pip = float(self.hand_result[tip][1]) < float(self.hand_result[pip][1])
            if ratio > 0.5 or (ratio > 0.35 and tip_above_pip):
                self.finger |= 1

        # Thumb extension check (distance from thumb tip 4 to pinky MCP 17 vs index MCP 5 to pinky MCP 17)
        palm_width = self.get_dist([5, 17])
        thumb_span = self.get_dist([4, 17])
        if palm_width > 1e-6 and (thumb_span / palm_width) > 1.25:
            self.finger |= int(Gest.THUMB)

    def get_gesture(self) -> int:
        """Classify current gesture with temporal debounce."""
        if self.hand_result is None:
            self.prev_gesture = Gest.PALM
            self.ori_gesture = Gest.PALM
            self.frame_count = 0
            return Gest.PALM

        self.set_finger_state()
        fingers_only = self.finger & int(Gest.LAST4)
        current_gesture = self.finger

        # Pinch detection: Thumb tip (4) close to Index tip (8)
        if fingers_only in (int(Gest.LAST3), int(Gest.LAST4)) and self.get_dist([8, 4]) < 0.055:
            current_gesture = (
                int(Gest.PINCH_MAJOR) if self.hand_label == HLabel.MAJOR else int(Gest.PINCH_MINOR)
            )
        elif fingers_only == int(Gest.FIRST2):
            # Index + Middle up -> V_GEST vs TWO_FINGER_CLOSED vs MID
            ref_dist = self.get_dist([5, 9])
            tip_dist = self.get_dist([8, 12])
            ratio = (tip_dist / ref_dist) if ref_dist > 1e-6 else 0.0
            if ratio > 1.7:
                current_gesture = int(Gest.V_GEST)
            elif self.get_dz([8, 12]) < 0.1:
                current_gesture = int(Gest.TWO_FINGER_CLOSED)
            else:
                current_gesture = int(Gest.MID)
        elif fingers_only == int(Gest.MID):
            current_gesture = int(Gest.MID)
        elif fingers_only == int(Gest.INDEX):
            current_gesture = int(Gest.INDEX)
        elif fingers_only == int(Gest.FIST):
            current_gesture = int(Gest.FIST)
        elif fingers_only == int(Gest.LAST4):
            current_gesture = int(Gest.PALM)

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        # Fast transition for single/double clicks when V-gesture already armed; 4-frame debounce otherwise
        debounce = 2 if current_gesture in (int(Gest.MID), int(Gest.INDEX), int(Gest.TWO_FINGER_CLOSED)) else 4
        if self.frame_count >= debounce:
            self.ori_gesture = current_gesture

        return int(self.ori_gesture)


class VirtualMouseController:
    """Executes smooth cursor navigation, clicks, drag-and-drop, scroll, and volume/zoom."""

    def __init__(
        self,
        failsafe: bool = True,
        smoothing: float = 0.35,
        active_margin: float = 0.18,
    ):
        self.failsafe = failsafe
        self.smoothing = smoothing
        self.active_margin = active_margin
        self.screen_w, self.screen_h = _get_screen_size()

        self.tx_old: float = 0.0
        self.ty_old: float = 0.0
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.last_cursor_pos: tuple[int, int] = (self.screen_w // 2, self.screen_h // 2)

        self.flag: bool = False
        self.grabflag: bool = False
        self.pinchmajorflag: bool = False
        self.pinchminorflag: bool = False
        self.pinchstartxcoord: float = 0.0
        self.pinchstartycoord: float = 0.0
        self.pinchdirectionflag: Optional[bool] = None
        self.prevpinchlv: float = 0.0
        self.pinchlv: float = 0.0
        self.framecount: int = 0
        self.pinch_threshold: float = 0.3
        self.last_volume_time: float = 0.0

        self._pg = None
        self.available: bool = False
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import pyautogui

            pyautogui.FAILSAFE = self.failsafe
            pyautogui.PAUSE = 0.0
            self._pg = pyautogui
            w, h = pyautogui.size()
            self.screen_w, self.screen_h = int(w), int(h)
            self.available = True
        except Exception:
            self._pg = None
            self.available = False

    def get_position(self, lm_norm: np.ndarray, mirrored: bool = True) -> tuple[int, int]:
        """Map Middle MCP landmark (9) to smoothed full-screen pixel coordinates."""
        raw_x = float(lm_norm[9][0])
        if mirrored:
            raw_x = 1.0 - raw_x
        raw_y = float(lm_norm[9][1])

        # Remap active inner camera rectangle [margin, 1 - margin] -> [0, 1] so all corners are reachable
        m = self.active_margin
        span = max(1.0 - 2.0 * m, 0.1)
        norm_x = max(0.0, min(1.0, (raw_x - m) / span))
        norm_y = max(0.0, min(1.0, (raw_y - m) / span))

        target_x = norm_x * (self.screen_w - 1)
        target_y = norm_y * (self.screen_h - 1)

        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x, self.smooth_y = target_x, target_y
            self.tx_old, self.ty_old = target_x, target_y

        delta_sq = (target_x - self.tx_old) ** 2 + (target_y - self.ty_old) ** 2
        self.tx_old, self.ty_old = target_x, target_y

        # Micro-tremor deadzone (<= 5 px^2)
        if delta_sq <= 6.0:
            return self.last_cursor_pos

        # Velocity-adaptive ratio damping + EMA smoothing
        ratio = 0.55 if delta_sq > 900.0 else (0.35 if delta_sq > 100.0 else 0.18)
        alpha = max(0.12, min(0.75, self.smoothing * ratio * 2.5))
        self.smooth_x = self.smooth_x + alpha * (target_x - self.smooth_x)
        self.smooth_y = self.smooth_y + alpha * (target_y - self.smooth_y)

        # Keep slightly inside exact 0/max corners unless user intentionally slams for failsafe
        cx = int(max(2, min(self.screen_w - 3, round(self.smooth_x))))
        cy = int(max(2, min(self.screen_h - 3, round(self.smooth_y))))
        self.last_cursor_pos = (cx, cy)
        return cx, cy

    def _pinch_init(self, lm_norm: np.ndarray, mirrored: bool) -> None:
        rx = (1.0 - float(lm_norm[8][0])) if mirrored else float(lm_norm[8][0])
        ry = float(lm_norm[8][1])
        self.pinchstartxcoord = rx
        self.pinchstartycoord = ry
        self.pinchlv = 0.0
        self.prevpinchlv = 0.0
        self.framecount = 0
        self.pinchdirectionflag = None

    def _get_pinch_lv(self, lm_norm: np.ndarray, mirrored: bool, vertical: bool) -> float:
        if vertical:
            ry = float(lm_norm[8][1])
            return round((self.pinchstartycoord - ry) * 10.0, 1)
        rx = (1.0 - float(lm_norm[8][0])) if mirrored else float(lm_norm[8][0])
        return round((rx - self.pinchstartxcoord) * 10.0, 1)

    def _release_grab_if_needed(self, armed: bool) -> Optional[str]:
        if self.grabflag:
            self.grabflag = False
            if armed and self.available and self._pg is not None:
                self._pg.mouseUp(button="left")
            return "Drag Release"
        return None

    def handle_controls(
        self,
        gesture: int,
        lm_norm: Optional[np.ndarray],
        armed: bool = True,
        mirrored: bool = True,
    ) -> Optional[str]:
        """Dispatch OS mouse/scroll/volume actions for the current gesture.

        Always computes cursor position so the HUD can display coordinates even when disarmed;
        only injects OS events via PyAutoGUI when ``armed=True``.
        """
        if lm_norm is None:
            self.flag = False
            self.pinchmajorflag = False
            self.pinchminorflag = False
            return self._release_grab_if_needed(armed)

        cx, cy = self.get_position(lm_norm, mirrored=mirrored)

        if gesture != int(Gest.PALM):
            pass  # active hand present
        if gesture != int(Gest.PINCH_MAJOR):
            self.pinchmajorflag = False
        if gesture != int(Gest.PINCH_MINOR):
            self.pinchminorflag = False

        event_msg: Optional[str] = None

        try:
            if gesture == int(Gest.V_GEST):
                self.flag = True
                rel = self._release_grab_if_needed(armed)
                if rel:
                    event_msg = rel
                if armed and self.available and self._pg is not None:
                    self._pg.moveTo(cx, cy, _pause=False)

            elif gesture == int(Gest.FIST):
                if not self.grabflag:
                    self.grabflag = True
                    event_msg = "Drag Start"
                    if armed and self.available and self._pg is not None:
                        self._pg.mouseDown(button="left")
                if armed and self.available and self._pg is not None:
                    self._pg.moveTo(cx, cy, _pause=False)

            elif gesture == int(Gest.MID) and self.flag:
                self.flag = False
                event_msg = "Left Click"
                if armed and self.available and self._pg is not None:
                    self._pg.click(button="left")

            elif gesture == int(Gest.INDEX) and self.flag:
                self.flag = False
                event_msg = "Right Click"
                if armed and self.available and self._pg is not None:
                    self._pg.click(button="right")

            elif gesture == int(Gest.TWO_FINGER_CLOSED) and self.flag:
                self.flag = False
                event_msg = "Double Click"
                if armed and self.available and self._pg is not None:
                    self._pg.doubleClick()

            elif gesture == int(Gest.PINCH_MAJOR):
                if not self.pinchmajorflag:
                    self._pinch_init(lm_norm, mirrored)
                    self.pinchmajorflag = True
                lv_x = self._get_pinch_lv(lm_norm, mirrored, vertical=False)
                lv_y = self._get_pinch_lv(lm_norm, mirrored, vertical=True)
                if abs(lv_y) > abs(lv_x) and abs(lv_y) > self.pinch_threshold:
                    now = time.perf_counter()
                    if now - self.last_volume_time > 0.18:
                        self.last_volume_time = now
                        key = "volumeup" if lv_y > 0 else "volumedown"
                        event_msg = "Volume Up" if lv_y > 0 else "Volume Down"
                        if armed and self.available and self._pg is not None:
                            self._pg.press(key)
                        self.pinchstartycoord = float(lm_norm[8][1])
                elif abs(lv_x) > self.pinch_threshold:
                    now = time.perf_counter()
                    if now - self.last_volume_time > 0.25:
                        self.last_volume_time = now
                        event_msg = "Scroll" if lv_x > 0 else "Scroll Back"
                        if armed and self.available and self._pg is not None:
                            self._pg.scroll(120 if lv_x > 0 else -120)
                        rx = (1.0 - float(lm_norm[8][0])) if mirrored else float(lm_norm[8][0])
                        self.pinchstartxcoord = rx

            elif gesture == int(Gest.PINCH_MINOR):
                if not self.pinchminorflag:
                    self._pinch_init(lm_norm, mirrored)
                    self.pinchminorflag = True
                lv_y = self._get_pinch_lv(lm_norm, mirrored, vertical=True)
                if abs(lv_y) > self.pinch_threshold:
                    now = time.perf_counter()
                    if now - self.last_volume_time > 0.12:
                        self.last_volume_time = now
                        clicks = 120 if lv_y > 0 else -120
                        event_msg = "Scroll Up" if lv_y > 0 else "Scroll Down"
                        if armed and self.available and self._pg is not None:
                            self._pg.scroll(clicks)
                        self.pinchstartycoord = float(lm_norm[8][1])

            else:
                # PALM or other neutral posture
                self.flag = False
                rel = self._release_grab_if_needed(armed)
                if rel:
                    event_msg = rel
        except Exception as exc:
            if type(exc).__name__ == "FailSafeException":
                self.grabflag = False
                self.flag = False
                return "FailSafe Abort"
            raise

        return event_msg


def draw_classic_mediapipe_overlay(
    frame: np.ndarray,
    lm_norm: Optional[np.ndarray],
    mirrored: bool = True,
    gesture_label: Optional[str] = None,
    cursor_xy: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Render classic MediaPipe green bone connections and red landmark dots (matching screenshot)."""
    h, w = frame.shape[:2]

    if lm_norm is not None:
        if mirrored:
            pts = np.stack([(1.0 - lm_norm[:, 0]) * w, lm_norm[:, 1] * h], axis=1).astype(int)
        else:
            pts = np.stack([lm_norm[:, 0] * w, lm_norm[:, 1] * h], axis=1).astype(int)

        # Green bone lines
        for a, b in HAND_EDGES:
            p, q = tuple(pts[a]), tuple(pts[b])
            cv2.line(frame, p, q, (0, 220, 0), 2, cv2.LINE_AA)

        # Red joint landmarks
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Compact status pill in top-left corner so user sees live gesture state on floating screen
    if gesture_label:
        label_txt = gesture_label
        if cursor_xy is not None and "CURSOR" in gesture_label:
            label_txt = f"{gesture_label} ({cursor_xy[0]}, {cursor_xy[1]})"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (8, 8), (tw + 22, th + 20), (15, 16, 18), -1)
        cv2.rectangle(frame, (8, 8), (tw + 22, th + 20), (0, 220, 0) if lm_norm is not None else (80, 80, 80), 1)
        cv2.putText(frame, label_txt, (15, th + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    return frame


class FloatingScreenWindow:
    """Manages an Always-On-Top native OpenCV window ("Gesture Controller") docked at top-right."""

    def __init__(
        self,
        window_name: str = "Gesture Controller",
        width: int = 440,
        height: int = 330,
        margin_right: int = 24,
        margin_top: int = 36,
    ):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.margin_right = margin_right
        self.margin_top = margin_top
        self._is_open = False
        self._positioned = False
        self._frame_count = 0
        self.last_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        """Whether the native OpenCV floating window is currently open."""
        return self._is_open

    def _apply_win32_topmost_and_pos(self, x: int, y: int) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, self.window_name)
            if hwnd:
                HWND_TOPMOST = -1
                SWP_SHOWWINDOW = 0x0040
                SWP_NOACTIVATE = 0x0010
                # Set size & top-right position + always on top without stealing focus
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    x,
                    y,
                    self.width,
                    self.height,
                    SWP_SHOWWINDOW | SWP_NOACTIVATE,
                )
        except Exception:
            pass

    def show(self, frame_bgr: np.ndarray) -> bool:
        """Render frame in the floating window. Returns False if user closed the window."""
        if frame_bgr is None or frame_bgr.size == 0:
            return True

        try:
            if not self._is_open:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.width, self.height)
                screen_w, _ = _get_screen_size()
                x = max(0, screen_w - self.width - self.margin_right)
                y = max(0, self.margin_top)
                cv2.moveWindow(self.window_name, x, y)
                try:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
                except Exception:
                    pass
                self._is_open = True
                self._positioned = False
                self._frame_count = 1
                self.last_error = None
                cv2.imshow(self.window_name, frame_bgr)
                cv2.waitKey(1)
                self._apply_win32_topmost_and_pos(x, y)
                self._positioned = True
                return True

            self._frame_count += 1
            cv2.imshow(self.window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.close()
                return False

            # Detect user clicking the native 'X' button (after brief Win32 show warmup; -1 means unsupported)
            if self._frame_count >= 10:
                try:
                    vis = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                    if 0.0 <= vis < 1.0:
                        self._is_open = False
                        self._positioned = False
                        self._frame_count = 0
                        return False
                except Exception:
                    pass
            return True
        except Exception as exc:
            # Headless environment or OpenCV GUI error
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._is_open = False
            self._positioned = False
            self._frame_count = 0
            return False

    def close(self) -> None:
        """Safely destroy the floating OpenCV window."""
        if self._is_open:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except Exception:
                pass
        self._is_open = False
        self._positioned = False
        self._frame_count = 0

