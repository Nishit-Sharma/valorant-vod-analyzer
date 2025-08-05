import cv2
import numpy as np
import pytesseract
import re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ScoreboardInfo:
    """Parsed information from Valorant scoreboard overlay."""
    round_number: Optional[int]
    time_seconds: Optional[int]
    bomb_planted: bool


class ScoreboardDetector:
    """Detect round/time/bomb state from the top-center scoreboard overlay.

    This uses OCR via pytesseract. Accuracy improves if you have the `tessdata_best`
    trained data installed. For production you may want to switch to specific digit
    templates or a tiny digit classifier, but OCR works well enough to mark round
    boundaries.
    """

    # Regex patterns for OCR output
    _ROUND_RE = re.compile(r"ROUND\s*(\d+)", re.IGNORECASE)
    _TIME_RE = re.compile(r"(\d)[:](\d{2})")

    def __init__(self, frame_shape: Tuple[int, int, int]):
        h, w, _ = frame_shape
        # Empirically the scoreboard occupies ~40% width by 8% height around the
        # horizontal centre.
        sb_h = int(h * 0.08)
        sb_y = 0
        sb_w = int(w * 0.4)
        sb_x = int((w - sb_w) / 2)
        self.roi = (sb_x, sb_y, sb_w, sb_h)  # (x, y, w, h)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Increase contrast and threshold for clearer OCR
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def parse(self, frame: np.ndarray) -> ScoreboardInfo:
        x, y, w, h = self.roi
        crop = frame[y : y + h, x : x + w]
        proc = self._preprocess(crop)

        text = pytesseract.image_to_string(proc, config="--psm 7")

        round_number = None
        time_seconds = None
        bomb_planted = False

        # Round number
        m = self._ROUND_RE.search(text)
        if m:
            try:
                round_number = int(m.group(1))
            except ValueError:
                pass

        # Time
        m = self._TIME_RE.search(text)
        if m:
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            time_seconds = minutes * 60 + seconds

        # Bomb planted icon: detect by dominant pink colour in crop centre (simple)
        # You can replace this with a template for the spike icon for higher accuracy.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Pink/magenta mask
        lower = np.array([150, 100, 100])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        bomb_pixels = cv2.countNonZero(mask)
        bomb_planted = bomb_pixels > (0.01 * w * h)  # >1% of ROI is pink/magenta

        return ScoreboardInfo(round_number, time_seconds, bomb_planted)
