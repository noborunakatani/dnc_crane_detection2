#!/usr/bin/env python3
"""
read_multiplier.py

固定位置に表示される「×n」表記から右側の整数を読み取るスクリプト。

実行例:
    python read_multiplier.py --video "/mnt/data/ズーム動画.mp4" --fps 5 --roi 1200 70 80 50 --save-annotated
    python read_multiplier.py --video "/mnt/data/ズーム動画.mp4" --fps 5 --auto-roi --track --save-annotated
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------- OCR Engine Wrapper -----------------------------


class BaseOCREngine:
    """Common interface for OCR engines."""

    name: str = "base"

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        raise NotImplementedError


class PaddleEngine(BaseOCREngine):
    name = "paddle"

    def __init__(self, use_gpu: bool = False) -> None:
        from paddleocr import PaddleOCR  # type: ignore

        self.ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False, use_gpu=use_gpu)

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        # PaddleOCR expects RGB images.
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(rgb, cls=False)
        outputs: List[Tuple[str, float]] = []
        for line in result:
            if not line:
                continue
            text, confidence = line[1]
            outputs.append((str(text), float(confidence)))
        return outputs


class EasyOCREngine(BaseOCREngine):
    name = "easyocr"

    def __init__(self) -> None:
        import easyocr  # type: ignore

        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        if image.ndim == 2:
            to_ocr = image
        else:
            to_ocr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(to_ocr)
        outputs: List[Tuple[str, float]] = []
        for _, text, confidence in results:
            outputs.append((str(text), float(confidence)))
        return outputs


class TesseractEngine(BaseOCREngine):
    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract  # type: ignore

        from pytesseract import Output  # type: ignore

        self.pytesseract = pytesseract
        self.output = Output
        self.config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"

    def recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        data = self.pytesseract.image_to_data(gray, output_type=self.output.DICT, config=self.config)
        outputs: List[Tuple[str, float]] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            if text is None:
                continue
            cleaned = text.strip()
            if not cleaned:
                continue
            try:
                conf_float = float(conf)
            except (TypeError, ValueError):
                conf_float = 0.0
            if conf_float < 0:
                conf_float = 0.0
            outputs.append((cleaned, conf_float / 100.0))
        return outputs


def initialize_ocr_engine(preferred: str = "auto", allow_gpu: bool = False) -> BaseOCREngine:
    """Instantiate available OCR engine following preference order."""

    errors: Dict[str, str] = {}
    order = ["paddle", "easyocr", "tesseract"] if preferred == "auto" else [preferred]
    # Append fallbacks if not already included
    if preferred != "auto":
        for name in ["paddle", "easyocr", "tesseract"]:
            if name not in order:
                order.append(name)

    for name in order:
        try:
            if name == "paddle":
                return PaddleEngine(use_gpu=allow_gpu)
            if name == "easyocr":
                return EasyOCREngine()
            if name == "tesseract":
                return TesseractEngine()
        except Exception as exc:  # pragma: no cover - informative logging
            errors[name] = str(exc)
            continue
    raise RuntimeError(f"No OCR engine could be initialized. Errors: {errors}")


# ------------------------------ Helper Classes -------------------------------


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return ROI(x, y, w, h)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class ValueSmoother:
    """Mode voting smoother for temporal stabilization."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, window_size)
        self.buffer: Deque[str] = deque(maxlen=self.window_size)

    def update(self, value: str) -> str:
        if value:
            self.buffer.append(value)
        if not self.buffer:
            return value
        counts = Counter(self.buffer)
        max_count = max(counts.values())
        candidates = [val for val, cnt in counts.items() if cnt == max_count]
        if value and value in candidates:
            return value
        # choose the most recent candidate
        for stored in reversed(self.buffer):
            if stored in candidates:
                return stored
        return value


# ------------------------------ Utility Methods ------------------------------


def build_cross_template(size: int = 45, thickness: int = 6) -> np.ndarray:
    """Generate a synthetic '×' template for template matching."""

    size = max(15, size)
    thickness = max(1, thickness)
    template = np.zeros((size, size), dtype=np.uint8)
    cv2.line(template, (0, 0), (size - 1, size - 1), 255, thickness)
    cv2.line(template, (size - 1, 0), (0, size - 1), 255, thickness)
    template = cv2.GaussianBlur(template, (3, 3), 0)
    return template


def preprocess_for_ocr(
    roi_img: np.ndarray,
    scale_factor: float = 2.5,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    adaptive: bool = False,
) -> np.ndarray:
    """Apply contrast enhancement and binarization before OCR."""

    if roi_img.ndim == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    if adaptive:
        processed = cv2.adaptiveThreshold(
            unsharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
    else:
        _, processed = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if scale_factor != 1.0:
        processed = cv2.resize(
            processed,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    kernel = np.ones((3, 3), dtype=np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)
    return processed


def build_digit_mask(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    adaptive: bool = False,
    color_hint: Optional[Tuple[int, int, int]] = None,
    color_tolerance: int = 30,
) -> np.ndarray:
    """Create a binary mask emphasizing the digit strokes."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clahe_clip), tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if adaptive:
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            2,
        )
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if color_hint is not None:
        target = np.array(color_hint, dtype=np.int16)
        diff = np.abs(image.astype(np.int16) - target)
        channel_diff = diff.max(axis=2)
        mask = (channel_diff <= max(1, color_tolerance)).astype(np.uint8) * 255
        binary = cv2.bitwise_and(binary, mask)

    white_ratio = float(cv2.countNonZero(binary)) / float(binary.size or 1)
    if white_ratio < 0.1:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def _active_span(values: np.ndarray, min_len: int, threshold: float) -> Optional[Tuple[int, int]]:
    active_idx = np.where(values >= threshold)[0]
    if active_idx.size == 0:
        return None
    start = int(active_idx[0])
    end = int(active_idx[-1])
    if end - start + 1 < min_len:
        return None
    return start, end


def refine_digit_roi(
    frame: np.ndarray,
    digit_roi: ROI,
    clahe_clip: float,
    clahe_grid: Tuple[int, int],
    adaptive: bool,
    color_hint: Optional[Tuple[int, int, int]],
    color_tolerance: int,
    min_band_px: int = 8,
    min_x: Optional[int] = None,
) -> Tuple[ROI, np.ndarray]:
    """Tighten the digit ROI to active columns/rows and return its mask."""

    h, w = frame.shape[:2]
    digit_roi = digit_roi.clamp(w, h)
    crop = frame[digit_roi.y : digit_roi.y + digit_roi.h, digit_roi.x : digit_roi.x + digit_roi.w]
    if crop.size == 0:
        return digit_roi, np.zeros((1, 1), dtype=np.uint8)

    mask = build_digit_mask(
        crop,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        adaptive=adaptive,
        color_hint=color_hint,
        color_tolerance=color_tolerance,
    )

    if min_x is not None:
        min_col_start = max(0, min_x - digit_roi.x)
        if min_col_start > 0:
            mask[:, :min_col_start] = 0

    cols = mask.sum(axis=0) / 255.0
    rows = mask.sum(axis=1) / 255.0
    col_threshold = max(1.0, 0.12 * mask.shape[0])
    row_threshold = max(1.0, 0.10 * mask.shape[1])

    col_span = _active_span(cols, min_band_px, col_threshold)
    row_span = _active_span(rows, max(3, int(0.5 * mask.shape[0])), row_threshold)

    if col_span:
        start_x, end_x = col_span
        start_x = max(0, start_x - 1)
        end_x = min(mask.shape[1] - 1, end_x + 1)
        digit_roi = ROI(digit_roi.x + start_x, digit_roi.y, end_x - start_x + 1, digit_roi.h)
        mask = mask[:, start_x : end_x + 1]
    if row_span:
        start_y, end_y = row_span
        start_y = max(0, start_y - 1)
        end_y = min(mask.shape[0] - 1, end_y + 1)
        digit_roi = ROI(digit_roi.x, digit_roi.y + start_y, digit_roi.w, end_y - start_y + 1)
        mask = mask[start_y : end_y + 1, :]

    return digit_roi.clamp(w, h), mask


def split_digit_segments(mask: np.ndarray, expected: int) -> List[np.ndarray]:
    if mask.size == 0:
        return []
    cols = mask.sum(axis=0) / 255.0
    gap_threshold = max(1.0, 0.05 * mask.shape[0])
    segments: List[Tuple[int, int]] = []
    start = 0
    for idx in range(mask.shape[1]):
        if cols[idx] <= gap_threshold:
            if idx - start >= 3:
                segments.append((start, idx - 1))
            start = idx + 1
    if mask.shape[1] - start >= 3:
        segments.append((start, mask.shape[1] - 1))

    if not segments and expected > 0:
        width = mask.shape[1]
        chunk = max(1, width // expected)
        segments = [(i * chunk, min(width - 1, (i + 1) * chunk - 1)) for i in range(expected)]

    if expected and len(segments) != expected:
        if len(segments) > expected:
            segments = segments[:expected]
        else:
            while len(segments) < expected:
                last = segments[-1] if segments else (0, mask.shape[1] - 1)
                segments.append(last)

    extracted: List[np.ndarray] = []
    for start_col, end_col in segments:
        extracted.append(mask[:, start_col : end_col + 1])
    return extracted


def count_holes(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    inverted = cv2.bitwise_not(mask)
    num_labels, labels = cv2.connectedComponents(inverted)
    hole_labels: List[int] = []
    h, w = mask.shape
    for label in range(1, num_labels):
        component = labels == label
        if not component.any():
            continue
        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border:
            hole_labels.append(label)
    return len(hole_labels)


def disambiguate_three_eight(value: str, mask: np.ndarray) -> str:
    if not value or not value.isdigit():
        return value
    segments = split_digit_segments(mask, len(value)) if mask.size else []
    if not segments:
        return value
    chars = list(value)
    for idx, segment_mask in enumerate(segments):
        if idx >= len(chars):
            break
        char = chars[idx]
        if char not in {"3", "8"}:
            continue
        holes = count_holes(segment_mask)
        if char == "3" and holes >= 2:
            chars[idx] = "8"
        elif char == "8" and holes == 0:
            chars[idx] = "3"
    return "".join(chars)


def extract_numeric_candidate(recognitions: Sequence[Tuple[str, float]], whitelist_regex: str = r"^[0-9]{1,2}$") -> Tuple[str, float]:
    """Pick the best numeric candidate from OCR outputs."""

    best_value = ""
    best_conf = 0.0
    for text, conf in recognitions:
        cleaned = text.strip().replace(" ", "")
        cleaned = cleaned.replace("O", "0").replace("o", "0")
        cleaned = cleaned.replace("l", "1").replace("I", "1")
        cleaned = cleaned.replace("|", "1")
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]
        if not cleaned:
            continue
        if not re_match(whitelist_regex, cleaned):
            continue
        if conf >= best_conf:
            best_value = cleaned
            best_conf = conf
    return best_value, best_conf


def re_match(pattern: str, text: str) -> bool:
    import re

    return re.fullmatch(pattern, text) is not None


def annotate_frame(
    frame: np.ndarray,
    roi: ROI,
    digit_roi: ROI,
    value: str,
    confidence: float,
    output_path: Path,
) -> None:
    """Save an annotated frame showing ROI, digit band, and OCR result."""

    annotated = frame.copy()
    cv2.rectangle(annotated, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 2)
    cv2.rectangle(
        annotated,
        (digit_roi.x, digit_roi.y),
        (digit_roi.x + digit_roi.w, digit_roi.y + digit_roi.h),
        (0, 165, 255),
        2,
    )
    shown_value = value if value else "未検出"
    label = f"値:{shown_value} / 信頼度:{confidence:.2f}"
    text_pos = (digit_roi.x, max(0, digit_roi.y - 5))
    cv2.putText(
        annotated,
        label,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), annotated)


# --------------------------- ROI Calibration Logic ---------------------------


def calibrate_roi_from_frames(
    frames: Sequence[np.ndarray],
    template: Optional[np.ndarray],
    left_margin: float,
    right_extension: float,
    top_margin: float,
    bottom_extension: float,
    digit_width_factor: float,
) -> Tuple[ROI, np.ndarray]:
    """Estimate ROI via template matching against the '×' symbol."""

    if not frames:
        raise ValueError("No frames available for ROI calibration")

    h, w = frames[0].shape[:2]
    if template is None:
        base_size = max(30, min(h, w) // 12)
        template = build_cross_template(size=base_size)

    scores: List[Tuple[float, Tuple[int, int], int]] = []
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search = cv2.Canny(gray, 50, 150)
        tmpl = template
        if tmpl.ndim == 3:
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        else:
            tmpl_gray = tmpl
        tmpl_edges = cv2.Canny(tmpl_gray, 50, 150)
        res = cv2.matchTemplate(search, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        scores.append((float(max_val), (int(max_loc[0]), int(max_loc[1])), idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    if not scores:
        raise RuntimeError("Template matching failed during calibration")

    top_k = scores[: min(5, len(scores))]
    total_score = sum(score for score, _loc, _idx in top_k)
    if total_score <= 1e-6:
        raise RuntimeError("Template matching scores too small; calibration failed")

    weighted_x = sum(score * loc[0] for score, loc, _idx in top_k) / total_score
    weighted_y = sum(score * loc[1] for score, loc, _idx in top_k) / total_score
    best_w = template.shape[1]
    best_h = template.shape[0]
    roi_x = int(max(0, weighted_x - best_w * left_margin))
    roi_y = int(max(0, weighted_y - best_h * top_margin))
    roi_w = int(min(w - roi_x, best_w * (1.0 + right_extension + digit_width_factor)))
    roi_h = int(min(h - roi_y, best_h * (1.0 + bottom_extension)))

    roi = ROI(roi_x, roi_y, max(1, roi_w), max(1, roi_h))

    # Build refined template from the best-scoring frame
    best_score, best_loc, best_idx = top_k[0]
    best_frame = frames[min(max(best_idx, 0), len(frames) - 1)]
    refined_x = int(best_loc[0])
    refined_y = int(best_loc[1])
    refined_w = int(min(best_w, w - refined_x))
    refined_h = int(min(best_h, h - refined_y))
    refined_crop = best_frame[refined_y : refined_y + refined_h, refined_x : refined_x + refined_w]
    refined_template = cv2.cvtColor(refined_crop, cv2.COLOR_BGR2GRAY)
    return roi.clamp(w, h), refined_template


def track_cross(
    gray_frame: np.ndarray,
    prev_roi: ROI,
    template: np.ndarray,
    search_radius: int,
    score_threshold: float = 0.35,
) -> Optional[Tuple[int, int, float]]:
    h, w = gray_frame.shape[:2]
    tmpl_h, tmpl_w = template.shape[:2]
    roi_x = max(0, prev_roi.x - search_radius)
    roi_y = max(0, prev_roi.y - search_radius)
    roi_w = min(prev_roi.w + 2 * search_radius, w - roi_x)
    roi_h = min(prev_roi.h + 2 * search_radius, h - roi_y)
    search = gray_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    if search.size == 0 or roi_w < tmpl_w or roi_h < tmpl_h:
        return None
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if float(max_val) < score_threshold:
        return None
    new_x = roi_x + int(max_loc[0])
    new_y = roi_y + int(max_loc[1])
    return new_x, new_y, float(max_val)


# ----------------------------- Processing Pipeline ---------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read multiplier digits from video frames")
    parser.add_argument("--video", type=str, default="/mnt/data/ズーム動画.mp4", help="Path to input video")
    parser.add_argument("--frames-dir", type=str, default="", help="Directory of pre-extracted frames (optional)")
    parser.add_argument("--fps", type=float, default=5.0, help="Sampling FPS")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help=(
            "Manual ROI in pixels. Provide the top-left coordinate of the × symbol (X,Y) and set"
            " W,H so the rectangle encloses the × and the digits immediately to the right."
        ),
    )
    parser.add_argument("--auto-roi", action="store_true", help="Enable automatic ROI calibration from initial frames")
    parser.add_argument("--calibration-frames", type=int, default=20, help="Number of frames to use for auto ROI calibration")
    parser.add_argument("--calibration-stride", type=int, default=2, help="Stride between frames during calibration")
    parser.add_argument("--cross-template", type=str, default="", help="Optional template image for × symbol")
    parser.add_argument("--track", action="store_true", help="Enable small-window tracking of ROI per frame")
    parser.add_argument("--drift-radius", type=int, default=12, help="Tracking search radius in pixels")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated frames to out_frames directory")
    parser.add_argument("--output-dir", type=str, default="out_frames", help="Directory for annotated frames")
    parser.add_argument("--csv", type=str, default="results.csv", help="Output CSV path")
    parser.add_argument("--digit-band-start", type=float, default=0.35, help="Fractional start (0-1) of digit band within ROI width")
    parser.add_argument("--digit-band-end", type=float, default=1.0, help="Fractional end (0-1) of digit band within ROI width")
    parser.add_argument("--cross-band", type=float, default=0.3, help="Fractional width of cross area inside ROI")
    parser.add_argument(
        "--digit-offset",
        type=int,
        default=4,
        help="Pixel offset applied to the right edge of the × region before OCR starts (guards against × being read)",
    )
    parser.add_argument(
        "--min-digit-width",
        type=int,
        default=56,
        help="Minimum width in pixels reserved for the digit band (ensures room for up to two digits)",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Window size for mode voting smoother")
    parser.add_argument("--ocr-engine", type=str, default="auto", choices=["auto", "paddle", "easyocr", "tesseract"], help="OCR engine selection")
    parser.add_argument("--ocr-gpu", action="store_true", help="Allow GPU usage for OCR if supported")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Use adaptive threshold instead of Otsu")
    parser.add_argument("--digit-regex", type=str, default=r"^[0-9]{1,2}$", help="Regex to validate OCR outputs")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument(
        "--clahe-grid",
        nargs=2,
        type=int,
        default=(8, 8),
        metavar=("GRID_W", "GRID_H"),
        help="CLAHE grid size (tile width and height)",
    )
    parser.add_argument("--scale", type=float, default=2.5, help="Scale factor prior to OCR")
    parser.add_argument("--summary", action="store_true", help="Print extended summary at end")
    parser.add_argument(
        "--digit-color",
        nargs=3,
        type=int,
        metavar=("B", "G", "R"),
        help="Optional BGR color hint for digits (improves masking when digits have fixed color)",
    )
    parser.add_argument(
        "--color-tolerance",
        type=int,
        default=35,
        help="Tolerance for color-based masking when --digit-color is set",
    )
    return parser.parse_args()


@dataclass
class FrameRecord:
    timestamp: float
    frame_index: int
    value: str
    confidence: float
    bbox: ROI
    annotated_name: str = ""


def load_template_image(path: str) -> np.ndarray:
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Failed to read template image: {path}")
    return template


def compute_digit_roi(
    base_roi: ROI,
    width: int,
    height: int,
    start_frac: float,
    end_frac: float,
    min_start: float = 0.0,
    offset_px: int = 0,
    min_width_px: int = 1,
) -> ROI:
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    end_frac = float(np.clip(end_frac, 0.0, 1.0))
    min_start = float(np.clip(min_start, 0.0, 1.0))
    offset_px = max(0, int(offset_px))
    min_width_px = max(1, int(min_width_px))

    if start_frac < min_start:
        start_frac = min_start
    if end_frac <= start_frac:
        end_frac = min(1.0, start_frac + 0.5)

    cross_right = base_roi.x + int(round(base_roi.w * min_start))
    x_from_frac = base_roi.x + int(round(base_roi.w * start_frac))
    x = max(x_from_frac, cross_right + offset_px)
    x = min(x, width - 1)
    max_x_candidate = base_roi.x + int(round(base_roi.w * end_frac))
    max_x_candidate = max(max_x_candidate, x + min_width_px)
    max_x = min(max_x_candidate, width)
    w = max(1, max_x - x)
    return ROI(x, base_roi.y, w, base_roi.h).clamp(width, height)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.save_annotated:
        output_dir.mkdir(parents=True, exist_ok=True)

    ocr_engine = initialize_ocr_engine(args.ocr_engine, allow_gpu=args.ocr_gpu)
    print(f"Using OCR engine: {ocr_engine.name}")
    if args.ocr_gpu and ocr_engine.name in {"tesseract", "easyocr"}:
        print(
            "Note: GPU acceleration is not used by the selected OCR engine; the --ocr-gpu flag is ignored."
        )

    frames_for_calibration: List[np.ndarray] = []
    calibration_template: Optional[np.ndarray] = None
    if args.cross_template:
        calibration_template = load_template_image(args.cross_template)

    manual_roi: Optional[ROI] = None
    if args.roi:
        manual_roi = ROI(*args.roi)
        print(
            "Manual ROI interpreted as: top-left at (X,Y) covering width W and height H around the ×,"
            " with the OCR digit band taken from {:.0f}% to {:.0f}% of that width toward the right.".format(
                args.digit_band_start * 100.0,
                args.digit_band_end * 100.0,
            )
        )
        print(
            "    ┌─────────── digit band ───────────┐\n"
            "    │    × area    │     digits      → │\n"
            "    └──────────────────────────────────┘\n"
            "    ↑(X,Y)        ↑ ROI width covers both regions"
        )
        print(
            "ヒント: X,Y はフレーム内の × マーク左上に合わせ、W は × と右隣の数字が十分入る幅、"
            "H は上下の余裕をもたせる高さに設定してください。"
        )
        print(
            "      → スクリプトは ROI の右側帯域だけを切り出して倍率 (数字) を読み取ります。× の右端から"
            f" {args.digit_offset}px 以上離れた位置から OCR を開始し、{args.min_digit_width}px 分の幅を必ず確保します。"
        )

    video_path = args.video
    frames_dir = args.frames_dir

    all_records: List[FrameRecord] = []
    smoother = ValueSmoother(window_size=args.smooth_window)
    calibrated_roi: Optional[ROI] = manual_roi
    clahe_grid = tuple(int(v) for v in args.clahe_grid)
    digit_color = tuple(int(v) for v in args.digit_color) if args.digit_color else None

    if frames_dir:
        frame_paths = sorted(
            [p for p in Path(frames_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}],
            key=lambda p: p.name,
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in directory: {frames_dir}")

        height = width = None
        roi = manual_roi
        cross_template = calibration_template
        if args.auto_roi and roi is None:
            for idx, path in enumerate(frame_paths[: args.calibration_frames]):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                frames_for_calibration.append(img)
            if not frames_for_calibration:
                raise RuntimeError("Unable to load frames for calibration")
            roi, cross_template = calibrate_roi_from_frames(
                frames_for_calibration,
                calibration_template,
                left_margin=0.15,
                right_extension=0.2,
                top_margin=0.1,
                bottom_extension=0.3,
                digit_width_factor=1.7,
            )
        elif roi is None:
            raise ValueError("ROI must be provided when not using auto calibration")

        cross_band_width = None

        for idx, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if height is None or width is None:
                height, width = frame.shape[:2]
                roi = roi.clamp(width, height)  # type: ignore
                cross_band_width = max(1, int(roi.w * args.cross_band))
                if cross_template is None and args.track:
                    gray_first = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cross_template = gray_first[
                        roi.y : roi.y + roi.h,
                        roi.x : roi.x + cross_band_width,
                    ].copy()
            frame_index = idx
            timestamp = idx / args.fps if args.fps > 0 else float(idx)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if args.track and cross_template is not None and cross_template.size > 0 and cross_band_width:
                tracked = track_cross(gray_frame, roi, cross_template, args.drift_radius)
                if tracked is not None:
                    new_x, new_y, score = tracked
                    roi = ROI(new_x, new_y, roi.w, roi.h).clamp(width, height)
                    if score > 0.6:
                        cross_template = gray_frame[
                            roi.y : roi.y + roi.h,
                            roi.x : roi.x + cross_band_width,
                        ].copy()

            digit_roi_base = compute_digit_roi(
                roi,
                width,
                height,
                args.digit_band_start,
                args.digit_band_end,
                min_start=args.cross_band,
                offset_px=args.digit_offset,
                min_width_px=args.min_digit_width,
            )
            digit_roi, digit_mask = refine_digit_roi(
                frame,
                digit_roi_base,
                clahe_clip=args.clahe_clip,
                clahe_grid=clahe_grid,
                adaptive=args.adaptive_threshold,
                color_hint=digit_color,
                color_tolerance=args.color_tolerance,
                min_x=digit_roi_base.x,
            )
            digit_crop = frame[digit_roi.y : digit_roi.y + digit_roi.h, digit_roi.x : digit_roi.x + digit_roi.w]
            processed = preprocess_for_ocr(
                digit_crop,
                scale_factor=args.scale,
                clahe_clip=args.clahe_clip,
                clahe_grid=clahe_grid,
                adaptive=args.adaptive_threshold,
            )
            ocr_ready = processed if ocr_engine.name == "tesseract" else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            recognitions = ocr_engine.recognize(ocr_ready)
            value_raw, conf = extract_numeric_candidate(recognitions, whitelist_regex=args.digit_regex)
            value_raw = disambiguate_three_eight(value_raw, digit_mask)
            value = smoother.update(value_raw)

            image_name = f"frame_{frame_index:06d}.png"
            record = FrameRecord(
                timestamp=timestamp,
                frame_index=frame_index,
                value=value,
                confidence=conf,
                bbox=digit_roi,
                annotated_name=image_name if args.save_annotated else "",
            )
            all_records.append(record)

            if args.save_annotated:
                out_path = output_dir / image_name
                annotate_frame(frame, roi, digit_roi, value, conf, out_path)

        calibrated_roi = roi if args.auto_roi else manual_roi

    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 1e-3:
            video_fps = args.fps if args.fps > 0 else 30.0
        frame_interval = max(1, int(round(video_fps / args.fps))) if args.fps > 0 else 1

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

        # Auto calibration
        roi = manual_roi
        cross_template = calibration_template
        if args.auto_roi and roi is None:
            cap_calib = cv2.VideoCapture(video_path)
            if not cap_calib.isOpened():
                raise FileNotFoundError(f"Cannot open video for calibration: {video_path}")
            calib_idx = 0
            while calib_idx < args.calibration_frames:
                ret, frame = cap_calib.read()
                if not ret:
                    break
                if calib_idx % max(1, args.calibration_stride) == 0:
                    frames_for_calibration.append(frame.copy())
                calib_idx += 1
            cap_calib.release()
            if not frames_for_calibration:
                raise RuntimeError("Failed to gather frames for ROI calibration")
            roi, cross_template = calibrate_roi_from_frames(
                frames_for_calibration,
                calibration_template,
                left_margin=0.15,
                right_extension=0.25,
                top_margin=0.1,
                bottom_extension=0.3,
                digit_width_factor=1.8,
            )
        elif roi is None:
            raise ValueError("ROI must be provided when not using auto calibration")

        calibrated_roi = roi
        frame_idx = 0

        cross_band_width = max(1, int(roi.w * args.cross_band))
        cross_template_ready = cross_template

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / video_fps if video_fps > 0 else frame_idx
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if height is None or width is None:
                height, width = frame.shape[:2]
                roi = roi.clamp(width, height)  # type: ignore

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if args.track:
                if cross_template_ready is None or cross_template_ready.size == 0:
                    cross_crop = gray_frame[roi.y : roi.y + roi.h, roi.x : roi.x + cross_band_width]
                    cross_template_ready = cross_crop.copy()
                else:
                    tracked = track_cross(gray_frame, roi, cross_template_ready, args.drift_radius)
                    if tracked is not None:
                        new_x, new_y, score = tracked
                        roi = ROI(new_x, new_y, roi.w, roi.h).clamp(width, height)
                        if score > 0.6:
                            cross_template_ready = gray_frame[
                                roi.y : roi.y + roi.h,
                                roi.x : roi.x + cross_band_width,
                            ].copy()

            digit_roi_base = compute_digit_roi(
                roi,
                width,
                height,
                args.digit_band_start,
                args.digit_band_end,
                min_start=args.cross_band,
                offset_px=args.digit_offset,
                min_width_px=args.min_digit_width,
            )
            digit_roi, digit_mask = refine_digit_roi(
                frame,
                digit_roi_base,
                clahe_clip=args.clahe_clip,
                clahe_grid=clahe_grid,
                adaptive=args.adaptive_threshold,
                color_hint=digit_color,
                color_tolerance=args.color_tolerance,
                min_x=digit_roi_base.x,
            )
            digit_crop = frame[digit_roi.y : digit_roi.y + digit_roi.h, digit_roi.x : digit_roi.x + digit_roi.w]
            processed = preprocess_for_ocr(
                digit_crop,
                scale_factor=args.scale,
                clahe_clip=args.clahe_clip,
                clahe_grid=clahe_grid,
                adaptive=args.adaptive_threshold,
            )
            ocr_ready = processed if ocr_engine.name == "tesseract" else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            recognitions = ocr_engine.recognize(ocr_ready)
            value_raw, conf = extract_numeric_candidate(recognitions, whitelist_regex=args.digit_regex)
            value_raw = disambiguate_three_eight(value_raw, digit_mask)
            value = smoother.update(value_raw)

            image_name = f"frame_{frame_idx:06d}.png"
            record = FrameRecord(
                timestamp=timestamp,
                frame_index=frame_idx,
                value=value,
                confidence=conf,
                bbox=digit_roi,
                annotated_name=image_name if args.save_annotated else "",
            )
            all_records.append(record)

            if args.save_annotated:
                out_path = output_dir / image_name
                annotate_frame(frame, roi, digit_roi, value, conf, out_path)
            frame_idx += 1

        cap.release()

    # ---------------------------- Output Generation ----------------------------

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mode_comment = "auto_roi" if args.auto_roi else "manual_roi"
    calibrated_roi_tuple = calibrated_roi.to_tuple() if calibrated_roi else None

    with csv_path.open("w", newline="") as f:
        f.write(f"# mode: {mode_comment}\n")
        if calibrated_roi_tuple:
            f.write(
                "# calibrated_roi: {0},{1},{2},{3}\n".format(
                    calibrated_roi_tuple[0],
                    calibrated_roi_tuple[1],
                    calibrated_roi_tuple[2],
                    calibrated_roi_tuple[3],
                )
            )
        if args.save_annotated:
            f.write(
                "# annotated_frame_pattern: frame_{frame_index:06d}.png (matches frame_index column when annotations saved)\n"
            )
        f.write(
            "# columns: timestamp_sec(seconds from start), frame_index(zero-based), value(OCR result), "
            "confidence(0-1 score), bbox_x/bbox_y/bbox_w/bbox_h(pixel ROI of digits), "
            "annotated_image(saved filename or blank)\n"
        )
        f.write("timestamp_sec,frame_index,value,confidence,bbox_x,bbox_y,bbox_w,bbox_h,annotated_image\n")
        writer = csv.writer(f)
        for record in all_records:
            writer.writerow(
                [
                    f"{record.timestamp:.3f}",
                    record.frame_index,
                    record.value,
                    f"{record.confidence:.4f}",
                    record.bbox.x,
                    record.bbox.y,
                    record.bbox.w,
                    record.bbox.h,
                    record.annotated_name,
                ]
            )

    # ------------------------------ Summary Print -----------------------------

    detections = [rec for rec in all_records if rec.value]
    detection_count = len(detections)
    avg_conf = sum(rec.confidence for rec in detections) / detection_count if detection_count else 0.0
    missing_count = len(all_records) - detection_count
    print("Processed frames:", len(all_records))
    print("Detected values:", detection_count)
    print("Average confidence:", f"{avg_conf:.3f}")
    print("Missing detections:", missing_count)
    print(f"CSV saved to: {csv_path}")
    if args.save_annotated:
        print(f"Annotated frames saved to: {output_dir}")
        print("  → ファイル名は frame_{frame_index:06d}.png 形式で、results.csv の frame_index 列と対応します。")
        print("  → 黄色枠は × を含む全体の ROI、オレンジ枠は数字領域、ラベルは『値:● / 信頼度:●』です。")

    if args.summary:
        unique_values = sorted(set(rec.value for rec in all_records if rec.value))
        print("Unique detected values:", unique_values)
        if all_records:
            first = all_records[0]
            last = all_records[-1]
            duration = last.timestamp - first.timestamp if last.timestamp >= first.timestamp else 0.0
            print(f"Duration covered: {duration:.2f} sec")
        print("results.csv columns:")
        print("  timestamp_sec - Seconds from start of the video/frame sequence.")
        print("      → 処理開始からの経過秒数。")
        print("  frame_index   - Zero-based index of the processed frame.")
        print("      → 0 始まりのフレーム番号。")
        print("  value         - Final stabilized digit string (empty if OCR failed).")
        print("      → 平滑化後の最終的な認識値（失敗時は空文字）。")
        print("  confidence    - Confidence score reported by the OCR engine (0-1 range).")
        print("      → OCR エンジンが返す信頼度 (0〜1)。")
        print("  bbox_x/y/w/h  - Location and size of the digit crop within the full frame in pixels.")
        print("      → 元フレーム内での数字領域の位置と大きさ (ピクセル単位)。")
        print("  annotated_image - Saved filename of the annotated frame (blank when not saved).")
        print("      → out_frames 内の画像ファイル名。保存していない場合は空欄。")


if __name__ == "__main__":
    main()
