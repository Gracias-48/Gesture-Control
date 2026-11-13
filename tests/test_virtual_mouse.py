"""Unit tests for Viral-Doshi Virtual Mouse gestures, controller, and classic overlay."""
from __future__ import annotations

import numpy as np

from src.inference.virtual_mouse import (
    Gest,
    HLabel,
    HandRecog,
    VirtualMouseController,
    draw_classic_mediapipe_overlay,
    gesture_to_label,
)


def _make_base_palm() -> np.ndarray:
    """Create an upright open palm (21, 3) normalized landmark array."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lm[0] = [0.50, 0.80, 0.0]
    # Thumb (1..4) extended to the left
    lm[1] = [0.42, 0.75, 0.0]
    lm[2] = [0.36, 0.68, 0.0]
    lm[3] = [0.31, 0.60, 0.0]
    lm[4] = [0.27, 0.52, 0.0]
    # Index (5..8) extended upward
    lm[5] = [0.42, 0.56, 0.0]
    lm[6] = [0.41, 0.44, 0.0]
    lm[7] = [0.40, 0.34, 0.0]
    lm[8] = [0.39, 0.24, 0.0]
    # Middle (9..12) extended upward
    lm[9] = [0.49, 0.54, 0.0]
    lm[10] = [0.49, 0.42, 0.0]
    lm[11] = [0.49, 0.32, 0.0]
    lm[12] = [0.49, 0.22, 0.0]
    # Ring (13..16) extended upward
    lm[13] = [0.56, 0.56, 0.0]
    lm[14] = [0.57, 0.44, 0.0]
    lm[15] = [0.58, 0.35, 0.0]
    lm[16] = [0.59, 0.26, 0.0]
    # Pinky (17..20) extended upward
    lm[17] = [0.63, 0.60, 0.0]
    lm[18] = [0.65, 0.50, 0.0]
    lm[19] = [0.66, 0.42, 0.0]
    lm[20] = [0.67, 0.34, 0.0]
    return lm


def _fold_finger(lm: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> None:
    """Bend a finger down toward the palm/wrist."""
    x = lm[mcp, 0]
    y_mcp = lm[mcp, 1]
    lm[pip] = [x, y_mcp + 0.02, 0.0]
    lm[dip] = [x, y_mcp + 0.08, 0.0]
    lm[tip] = [x, y_mcp + 0.14, 0.0]


def _stabilize_gesture(recog: HandRecog, lm: np.ndarray, frames: int = 8) -> int:
    gesture = Gest.PALM
    for _ in range(frames):
        recog.update_hand_result(lm)
        gesture = recog.get_gesture()
    return gesture


def test_hand_recog_palm_and_fist():
    recog = HandRecog(hand_label=HLabel.MAJOR)

    palm = _make_base_palm()
    assert _stabilize_gesture(recog, palm) == Gest.PALM
    assert isinstance(gesture_to_label(Gest.PALM), str)

    fist = palm.copy()
    fist[4] = [0.48, 0.65, 0.0]
    _fold_finger(fist, 5, 6, 7, 8)
    _fold_finger(fist, 9, 10, 11, 12)
    _fold_finger(fist, 13, 14, 15, 16)
    _fold_finger(fist, 17, 18, 19, 20)
    assert _stabilize_gesture(recog, fist) == Gest.FIST


def test_hand_recog_v_gest_and_two_finger_closed():
    recog = HandRecog(hand_label=HLabel.MAJOR)

    v_pose = _make_base_palm()
    # Fold ring & pinky, keep index & middle separated in a clear V
    v_pose[4] = [0.46, 0.64, 0.0]
    _fold_finger(v_pose, 13, 14, 15, 16)
    _fold_finger(v_pose, 17, 18, 19, 20)
    v_pose[8] = [0.35, 0.24, 0.0]
    v_pose[12] = [0.56, 0.22, 0.0]
    assert _stabilize_gesture(recog, v_pose) == Gest.V_GEST

    closed_pose = v_pose.copy()
    # Move index & middle tips close together horizontally
    closed_pose[5] = [0.47, 0.55, 0.0]
    closed_pose[6] = [0.475, 0.43, 0.0]
    closed_pose[7] = [0.48, 0.33, 0.0]
    closed_pose[8] = [0.482, 0.23, 0.0]
    closed_pose[10] = [0.49, 0.42, 0.0]
    closed_pose[11] = [0.49, 0.32, 0.0]
    closed_pose[12] = [0.492, 0.23, 0.0]
    assert _stabilize_gesture(recog, closed_pose) == Gest.TWO_FINGER_CLOSED


def test_hand_recog_pinch_major():
    recog = HandRecog(hand_label=HLabel.MAJOR)
    pinch = _make_base_palm()
    # Bring thumb tip (4) and index tip (8) together
    pinch[3] = [0.39, 0.45, 0.0]
    pinch[4] = [0.41, 0.40, 0.0]
    pinch[7] = [0.41, 0.41, 0.0]
    pinch[8] = [0.412, 0.402, 0.0]
    assert _stabilize_gesture(recog, pinch) == Gest.PINCH_MAJOR


def test_virtual_mouse_controller_preview_and_overlay():
    ctrl = VirtualMouseController(failsafe=True)
    palm = _make_base_palm()

    x, y = ctrl.get_position(palm, mirrored=True)
    assert isinstance(x, int) and isinstance(y, int)
    assert x >= 0 and y >= 0

    # V_GEST moves cursor in preview mode without clicking
    event = ctrl.handle_controls(Gest.V_GEST, palm, armed=False, mirrored=True)
    assert event is None
    assert ctrl.last_cursor_pos is not None

    # Overlay rendering produces valid BGR uint8 image with drawn skeleton pixels
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_classic_mediapipe_overlay(
        blank,
        palm,
        mirrored=True,
        gesture_label=gesture_to_label(Gest.V_GEST),
        cursor_xy=ctrl.last_cursor_pos,
    )
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8
    assert np.any(out > 0)
