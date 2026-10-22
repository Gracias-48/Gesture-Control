"""Regression tests for the HUD camera viewport renderer."""
import base64
import re

import cv2
import numpy as np

from app.styles import render_camera_frame


def test_render_camera_frame_embeds_decodable_jpeg():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 0, 0)  # blue half in BGR

    html = render_camera_frame(frame)

    assert html.startswith("<img")
    assert "\n" not in html  # must stay a single-line raw-HTML block for st.markdown
    m = re.search(r'src="data:image/jpeg;base64,([^"]+)"', html)
    assert m, "frame must be embedded as a data URI, not a media-file reference"

    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(m.group(1)), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape
    assert decoded[24, 8, 0] > 200  # left half still blue
    assert decoded[24, 56, 0] < 50  # right half still black
