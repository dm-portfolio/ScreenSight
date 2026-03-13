import numpy as np

from ai_vision import FrameStats, build_assistant_text, compute_brightness, compute_motion_score
from ai_vision import compute_brightness, compute_motion_score


def test_brightness_range():
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    white = np.full((20, 20, 3), 255, dtype=np.uint8)

    assert compute_brightness(black) == 0.0
    assert compute_brightness(white) == 1.0


def test_motion_score_detects_change():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    b = np.full((20, 20, 3), 255, dtype=np.uint8)

    assert compute_motion_score(a, None) == 0.0
    assert compute_motion_score(b, a) > 0.9


def test_assistant_text_responds_to_question():
    stats = FrameStats(brightness=0.8, edge_density=0.2, motion_score=0.05, ocr_text=None)
    reply = build_assistant_text(stats, user_text="what do you see")

    assert "bright" in reply
