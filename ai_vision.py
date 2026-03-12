from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameStats:
    brightness: float
    edge_density: float
    motion_score: float
    ocr_text: str | None


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode JPEG frame")
    return image


def compute_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


def compute_edge_density(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def compute_motion_score(frame_bgr: np.ndarray, prev_frame_bgr: np.ndarray | None) -> float:
    if prev_frame_bgr is None:
        return 0.0

    cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)

    if cur_gray.shape != prev_gray.shape:
        prev_gray = cv2.resize(prev_gray, (cur_gray.shape[1], cur_gray.shape[0]))

    diff = cv2.absdiff(cur_gray, prev_gray)
    return float(np.mean(diff) / 255.0)


def optional_ocr(frame_bgr: np.ndarray) -> str | None:
    import importlib.util

    has_tesseract = importlib.util.find_spec("pytesseract") is not None
    if not has_tesseract:
        return None

    import pytesseract

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb).strip()
    return text or None


def analyze_frame(
    frame_bgr: np.ndarray,
    prev_frame_bgr: np.ndarray | None = None,
    enable_ocr: bool = False,
) -> FrameStats:
    ocr_text = optional_ocr(frame_bgr) if enable_ocr else None

    return FrameStats(
        brightness=compute_brightness(frame_bgr),
        edge_density=compute_edge_density(frame_bgr),
        motion_score=compute_motion_score(frame_bgr, prev_frame_bgr),
        ocr_text=ocr_text,
    )


def build_assistant_text(stats: FrameStats, user_text: str | None = None) -> str:
    brightness = "bright" if stats.brightness >= 0.6 else "dim"
    movement = "a lot of motion" if stats.motion_score >= 0.2 else "little motion"
    detail = "high detail" if stats.edge_density >= 0.12 else "low detail"

    answer = f"I see a {brightness} scene with {detail} and {movement}."
    if stats.ocr_text:
        answer += f" Visible text: {stats.ocr_text[:120]}"

    if user_text:
        prompt = user_text.strip().lower()
        if "what" in prompt and "see" in prompt:
            return answer
        if "motion" in prompt or "moving" in prompt:
            return f"Current motion score is {stats.motion_score:.2f}, indicating {movement}."
        if "bright" in prompt or "dark" in prompt:
            return f"Brightness is {stats.brightness:.2f}, so the scene appears {brightness}."
        if "text" in prompt or "read" in prompt:
            return stats.ocr_text or "I do not detect readable text right now."

    return answer


def to_payload(stats: FrameStats) -> dict[str, Any]:
    return {
        "brightness": round(stats.brightness, 4),
        "edge_density": round(stats.edge_density, 4),
        "motion_score": round(stats.motion_score, 4),
        "ocr_text": stats.ocr_text,
    }
