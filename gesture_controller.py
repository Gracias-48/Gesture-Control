"""Standalone Always-On-Top Gesture Controller desktop launcher.

Runs the 30 FPS MediaPipe hand tracker and Virtual Mouse controller inside a
floating desktop camera window ("Gesture Controller") docked at top-right,
requiring no Streamlit web server to operate.

Usage:
    python gesture_controller.py
    python gesture_controller.py --no-os
    python gesture_controller.py --style hud --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.camera import CameraStream
from src.inference.hand_tracker import HandTracker, draw_hud_hand_overlay
from src.inference.virtual_mouse import (
    FloatingScreenWindow,
    HLabel,
    HandRecog,
    VirtualMouseController,
    draw_classic_mediapipe_overlay,
    gesture_to_label,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Always-On-Top Gesture Controller")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--no-os", action="store_true", help="Disable live OS mouse/keyboard actions (preview only)")
    parser.add_argument(
        "--style",
        choices=("classic", "hud"),
        default=None,
        help="Skeleton overlay style: classic (red dots / green bones) or hud",
    )
    parser.add_argument("--width", type=int, default=440, help="Floating window width in pixels")
    parser.add_argument("--height", type=int, default=330, help="Floating window height in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inf_cfg = cfg.get("inference", {})
    mp_cfg = inf_cfg.get("mediapipe", {})

    cam_idx = args.camera if args.camera is not None else int(inf_cfg.get("camera_index", 0))
    cap_w = int(inf_cfg.get("capture_width", 640))
    cap_h = int(inf_cfg.get("capture_height", 480))
    cap_fps = int(inf_cfg.get("capture_fps", 30))
    mirror_display = bool(inf_cfg.get("mirror_display", True))
    overlay_style = args.style or str(inf_cfg.get("overlay_style", "classic"))

    camera = CameraStream(index=cam_idx, width=cap_w, height=cap_h, fps=cap_fps).start()
    if camera.error:
        print(f"[GestureController] Camera error: {camera.error}")
        sys.exit(1)

    tracker = HandTracker(
        model_complexity=int(mp_cfg.get("model_complexity", 0)),
        min_detection_confidence=float(mp_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )
    hand_recog = HandRecog(hand_label=HLabel.MAJOR)
    vm_controller = VirtualMouseController(
        failsafe=bool(cfg.get("actions", {}).get("failsafe", True)),
    )
    floating_window = FloatingScreenWindow(
        window_name="Gesture Controller",
        width=args.width,
        height=args.height,
    )

    armed = not args.no_os
    print(f"[GestureController] Running standalone floating screen (OS Control: {'ARMED' if armed else 'PREVIEW'})")
    print("[GestureController] Press 'q', ESC, or click X on the window to exit.")

    last_seq = -1
    try:
        while True:
            seq, frame, _ = camera.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq

            _, lm_norm = tracker.process(frame)
            h, w = frame.shape[:2]

            hand_recog.update_hand_result(lm_norm, hand_label=tracker.last_hand_label)
            gesture = hand_recog.get_gesture()
            gesture_label = gesture_to_label(gesture)
            action_event = vm_controller.handle_controls(
                gesture=gesture,
                lm_norm=lm_norm,
                armed=armed,
                mirrored=mirror_display,
            )
            if action_event:
                print(f"[GestureController] {action_event}")

            display_frame = cv2.flip(frame, 1) if mirror_display else frame.copy()
            if overlay_style == "classic":
                draw_classic_mediapipe_overlay(
                    display_frame,
                    lm_norm,
                    mirrored=mirror_display,
                    gesture_label=gesture_label,
                    cursor_xy=vm_controller.last_cursor_pos,
                )
            else:
                motion_state, bbox, _ = tracker.update_motion_state(lm_norm, w, h, mirrored=mirror_display)
                draw_hud_hand_overlay(
                    display_frame,
                    lm_norm,
                    motion_state=motion_state,
                    bbox=bbox,
                    accent=(255, 224, 64),
                    mirrored=mirror_display,
                    candidate_label=gesture_label,
                )

            if not floating_window.show(display_frame):
                break
    except KeyboardInterrupt:
        pass
    finally:
        floating_window.close()
        tracker.close()
        camera.stop()
        print("[GestureController] Stopped.")


if __name__ == "__main__":
    main()
