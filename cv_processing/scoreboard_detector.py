import cv2
import numpy as np
import pytesseract
import re
import os

"""Scoreboard and replay detection utilities.

This module provides:
- ScoreboardDetector: OCR-based extraction of round number and timer, plus spike-planted detection via
  template or HSV fallback.
- ReplayDetector: Broadcast replay segment detection to avoid double-counting events during replays.
"""

# Configure Tesseract on Windows if an explicit path is provided
_TESS_CMD = os.environ.get("TESSERACT_CMD")
if _TESS_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESS_CMD

# --- Spike planted icon template(s) ---
SPIKE_TEMPLATE = None
SPIKE_TM_THRESHOLD = 0.88
_spike_template_candidates = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "game_states", "spike_planted_ui.png")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "game_states", "spike_planted.png")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "ui_elements", "spike_planted_ui.png")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "ui_elements", "spike_planted.png")),
]
for _p in _spike_template_candidates:
    if os.path.exists(_p):
        tmp = cv2.imread(_p, cv2.IMREAD_GRAYSCALE)
        if tmp is not None:
            SPIKE_TEMPLATE = tmp
            break
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
    _TIME_RE = re.compile(r"(\d{1,2})[:](\d{2})")

    def __init__(self, frame_shape: Tuple[int, int, int]):
        h, w, _ = frame_shape
        # Empirically the scoreboard occupies ~40% width by 10% height around the
        # horizontal centre. We widen a bit for OCR robustness across broadcasts.
        sb_h = int(h * 0.10)
        sb_y = 0
        sb_w = int(w * 0.5)
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

        # Bias OCR to characters we expect in the scoreboard line
        ocr_cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:"
        text = pytesseract.image_to_string(proc, config=ocr_cfg)

        round_number: Optional[int] = None
        time_seconds: Optional[int] = None
        bomb_planted: bool = False

        # Round number
        m = self._ROUND_RE.search(text)
        if m:
            try:
                round_number = int(m.group(1))
            except ValueError:
                round_number = None

        # Time
        m = self._TIME_RE.search(text)
        if m:
            try:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                time_seconds = minutes * 60 + seconds
            except ValueError:
                time_seconds = None

        # Spike planted detection
        # (1) Template matching if spike template available
        if SPIKE_TEMPLATE is not None:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if (
                gray_crop.shape[0] >= SPIKE_TEMPLATE.shape[0]
                and gray_crop.shape[1] >= SPIKE_TEMPLATE.shape[1]
            ):
                res = cv2.matchTemplate(gray_crop, SPIKE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                bomb_planted = max_val >= SPIKE_TM_THRESHOLD

        # (2) Fallback colour heuristic
        if not bomb_planted:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower = np.array([150, 100, 100])
            upper = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            bomb_pixels = cv2.countNonZero(mask)
            bomb_planted = bomb_pixels > (0.01 * w * h)  # >1% of ROI

        return ScoreboardInfo(round_number, time_seconds, bomb_planted)


class ReplayDetector:
    """Detect broadcast replay segments to avoid double-counting events.

    Strategy:
    - Prefer template matching if a replay banner template exists under templates/.
    - Fallback to OCR looking for the word REPLAY in a top-left/overall ROI.
    """

    def __init__(self):
        self.threshold = 0.92
        self._templates = []
        # Replay banners often appear bottom-right in VCT; also include generic templates
        replay_candidates = [
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "game_states", "replay.png")),
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "ui_elements", "replay.png")),
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "templates", "ui_elements", "replay_br.png")),
        ]
        for p in replay_candidates:
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self._templates.append(img)

    def is_replay(self, frame: np.ndarray) -> bool:
        # Template approach (full frame match)
        if self._templates:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for tmpl in self._templates:
                if gray.shape[0] >= tmpl.shape[0] and gray.shape[1] >= tmpl.shape[1]:
                    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val >= self.threshold:
                        return True
        # OCR fallback: look for REPLAY/REPLAY HIGHLIGHTS in bottom-right quadrant
        h, w = frame.shape[:2]
        roi_br = frame[int(h * 0.60):h, int(w * 0.60):w]
        proc_br = ScoreboardDetector._preprocess(self, roi_br)
        txt_br = pytesseract.image_to_string(proc_br, config="--psm 7 --oem 3")
        if "REPLAY" in txt_br.upper():
            return True
        # Also scan a central strip for animated wipes with text
        roi_mid = frame[int(h * 0.35):int(h * 0.65), int(w * 0.25):int(w * 0.75)]
        proc_mid = ScoreboardDetector._preprocess(self, roi_mid)
        txt_mid = pytesseract.image_to_string(proc_mid, config="--psm 7 --oem 3")
        return "REPLAY" in txt_mid.upper()
