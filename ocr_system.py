"""Comprehensive OCR solution for reading tiny on-screen text like "×1".

This module provides utilities to capture the screen, interactively choose a
region-of-interest (ROI), preprocess low-resolution text, and run multiple OCR
engines (Tesseract, EasyOCR, PaddleOCR).  It is designed for scenarios where
small white text such as monitoring overlays must be recognised reliably.

The module offers:

* High-quality preprocessing tuned for tiny characters (scaling, sharpening,
  contrast/brightness optimisation, denoising, binarisation).
* Screenshot capture using ``mss`` with optional ROI cropping and persistence.
* Tkinter-based GUI ROI selection with drag-to-select interactions.
* Unified interface to run several OCR backends with confidence reporting and
  graceful error handling when optional dependencies are missing.
* Real-time OCR streaming loop suitable for monitoring applications where text
  changes over time.

Example usage from the command line::

    python ocr_system.py --engine tesseract --select-roi --realtime

The example above launches an ROI picker, then continuously recognises text in
that region using Tesseract, printing both the recognised text and the
confidence score.  All functionality is available programmatically via the
``OCRSystem`` class as well.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk


def _optional_import(module_name: str):
    """Return the imported module if available, otherwise ``None``.

    The function avoids raising ``ImportError`` so the caller can provide a user
    friendly message when an optional dependency is missing.
    """

    if importlib.util.find_spec(module_name) is None:  # type: ignore[attr-defined]
        return None
    return importlib.import_module(module_name)


def _optional_attr(module, attr: str):
    """Fetch ``attr`` from ``module`` if present, otherwise return ``None``."""

    if module is None:
        return None
    return getattr(module, attr, None)


@dataclass
class OCRResult:
    """Stores the outcome of running an OCR engine."""

    engine: str
    text: str = ""
    confidence: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        """Represent the result as a JSON-serialisable dictionary."""

        payload = {
            "engine": self.engine,
            "text": self.text,
            "confidence": f"{self.confidence:.2f}",
        }
        if self.error:
            payload["error"] = self.error
        return payload


class ROISelector:
    """GUI helper to draw a rectangular ROI over a screenshot."""

    def __init__(self, screenshot: Image.Image):
        tkinter = _optional_import("tkinter")
        if tkinter is None:
            raise RuntimeError("tkinter is required for GUI ROI selection")

        self._tk = tkinter.Tk()
        self._tk.title("Select ROI - drag to draw a rectangle")
        self._start: Optional[Tuple[int, int]] = None
        self._end: Optional[Tuple[int, int]] = None
        self._finished = threading.Event()

        self._canvas = tkinter.Canvas(
            self._tk,
            width=screenshot.width,
            height=screenshot.height,
            cursor="cross",
        )
        self._canvas.pack(fill="both", expand=True)

        self._photo = ImageTk.PhotoImage(screenshot)
        self._canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._rect = self._canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event):
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)
        self._canvas.coords(self._rect, event.x, event.y, event.x, event.y)

    def _on_drag(self, event):
        if self._start is None:
            return
        self._end = (event.x, event.y)
        self._canvas.coords(self._rect, self._start[0], self._start[1], event.x, event.y)

    def _on_release(self, event):
        if self._start is None:
            return
        self._end = (event.x, event.y)
        self._finished.set()
        self._tk.quit()

    def select(self, timeout: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        """Run the GUI loop and return ``(x, y, width, height)`` if selected."""

        self._finished.clear()
        threading.Thread(target=self._tk.mainloop, daemon=True).start()
        self._finished.wait(timeout=timeout)
        if not self._finished.is_set() or self._start is None or self._end is None:
            self._tk.destroy()
            return None
        x0, y0 = self._start
        x1, y1 = self._end
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        self._tk.destroy()
        return left, top, right - left, bottom - top


class OCRSystem:
    """High-level controller for multi-engine OCR with preprocessing."""

    def __init__(self) -> None:
        self._pytesseract = _optional_import("pytesseract")
        self._easyocr_module = _optional_import("easyocr")
        self._paddleocr_module = _optional_import("paddleocr")

        self._easyocr_reader = None
        self._paddleocr_reader = None

    @staticmethod
    def capture_screen(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture the whole screen or a rectangular ROI using ``mss``."""

        with mss.mss() as sct:
            if region is None:
                monitor = sct.monitors[1]
            else:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}

            frame = sct.grab(monitor)
            img = np.array(frame)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

    @staticmethod
    def save_screenshot(path: Path, region: Optional[Tuple[int, int, int, int]] = None) -> Path:
        """Capture the screen/ROI and store it as an image file."""

        image = OCRSystem.capture_screen(region)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
        return path

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        scale_factor: float = 3.0,
        sharpen: bool = True,
        adjust_contrast: bool = True,
        denoise: bool = True,
        adaptive_threshold: bool = True,
    ) -> np.ndarray:
        """Apply preprocessing steps to enhance tiny text."""

        if image.size == 0:
            raise ValueError("Empty image supplied for preprocessing")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if scale_factor != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA,
            )

        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)

        if adjust_contrast:
            # Improve contrast for light text on dark backgrounds.
            gray = cv2.equalizeHist(gray)
            gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=-15)

        if sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

        if adaptive_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )

        return gray

    def select_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Launch the ROI selection GUI and return the chosen rectangle."""

        screenshot = self.capture_screen()
        rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        roi_selector = ROISelector(Image.fromarray(rgb))
        return roi_selector.select()

    def _ensure_easyocr(self):
        if self._easyocr_reader is not None:
            return self._easyocr_reader
        if self._easyocr_module is None:
            return None
        reader_class = _optional_attr(self._easyocr_module, "Reader")
        if reader_class is None:
            return None
        logging.info("Initialising EasyOCR reader (CPU mode, language=en)")
        self._easyocr_reader = reader_class(["en"], gpu=False)
        return self._easyocr_reader

    def _ensure_paddleocr(self):
        if self._paddleocr_reader is not None:
            return self._paddleocr_reader
        if self._paddleocr_module is None:
            return None
        paddle_class = _optional_attr(self._paddleocr_module, "PaddleOCR")
        if paddle_class is None:
            return None
        logging.info("Initialising PaddleOCR reader (CPU mode, language=en)")
        self._paddleocr_reader = paddle_class(lang="en", use_gpu=False, show_log=False)
        return self._paddleocr_reader

    def recognize_with_tesseract(self, image: np.ndarray) -> OCRResult:
        if self._pytesseract is None:
            return OCRResult(engine="tesseract", error="pytesseract is not installed")

        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789xX×"
        try:
            data = self._pytesseract.image_to_data(image, config=config, output_type=self._pytesseract.Output.DICT)
        except Exception as exc:  # noqa: BLE001
            return OCRResult(engine="tesseract", error=str(exc))

        text_candidates = [text for text in data.get("text", []) if text.strip()]
        confidences = [float(conf) for conf in data.get("conf", []) if conf not in ("-1", "-1.0")]
        text = " ".join(text_candidates).strip()
        confidence = np.mean(confidences) if confidences else 0.0
        return OCRResult(engine="tesseract", text=text, confidence=confidence)

    def recognize_with_easyocr(self, image: np.ndarray) -> OCRResult:
        reader = self._ensure_easyocr()
        if reader is None:
            return OCRResult(engine="easyocr", error="easyocr is not installed")
        try:
            result = reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001
            return OCRResult(engine="easyocr", error=str(exc))

        if not result:
            return OCRResult(engine="easyocr", text="", confidence=0.0)
        # Choose the detection with highest confidence.
        best = max(result, key=lambda item: item[2])
        return OCRResult(engine="easyocr", text=best[1], confidence=float(best[2]))

    def recognize_with_paddleocr(self, image: np.ndarray) -> OCRResult:
        reader = self._ensure_paddleocr()
        if reader is None:
            return OCRResult(engine="paddleocr", error="paddleocr is not installed")
        try:
            result = reader.ocr(image, cls=False)
        except Exception as exc:  # noqa: BLE001
            return OCRResult(engine="paddleocr", error=str(exc))

        if not result:
            return OCRResult(engine="paddleocr", text="", confidence=0.0)
        # PaddleOCR returns [[[box], (text, confidence)], ...]
        best = max(result, key=lambda item: item[1][1])
        return OCRResult(engine="paddleocr", text=best[1][0], confidence=float(best[1][1]))

    def recognize(
        self,
        image: np.ndarray,
        engines: Iterable[str] = ("tesseract", "easyocr", "paddleocr"),
    ) -> Dict[str, OCRResult]:
        """Run selected OCR engines on the provided (preprocessed) image."""

        results: Dict[str, OCRResult] = {}
        for engine in engines:
            engine_lower = engine.lower()
            if engine_lower == "tesseract":
                results[engine] = self.recognize_with_tesseract(image)
            elif engine_lower == "easyocr":
                results[engine] = self.recognize_with_easyocr(image)
            elif engine_lower == "paddleocr":
                results[engine] = self.recognize_with_paddleocr(image)
            else:
                results[engine] = OCRResult(engine=engine, error="Unknown OCR engine")
        return results

    def recognize_screen(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        engines: Iterable[str] = ("tesseract", "easyocr", "paddleocr"),
        preprocess_kwargs: Optional[Dict[str, object]] = None,
    ) -> Dict[str, OCRResult]:
        """Capture the screen/ROI, preprocess, and run OCR."""

        raw_image = self.capture_screen(region)
        processed = self.preprocess_image(
            raw_image,
            **(preprocess_kwargs or {}),
        )
        return self.recognize(processed, engines)

    def start_realtime(
        self,
        region: Tuple[int, int, int, int],
        callback: Callable[[Dict[str, OCRResult]], None],
        engines: Iterable[str] = ("tesseract", "easyocr", "paddleocr"),
        interval: float = 0.5,
        stop_event: Optional[threading.Event] = None,
        preprocess_kwargs: Optional[Dict[str, object]] = None,
    ) -> threading.Thread:
        """Start a background thread that streams OCR results in real-time."""

        if stop_event is None:
            stop_event = threading.Event()

        def _worker():
            logging.info("Starting real-time OCR loop with interval %.2fs", interval)
            while not stop_event.is_set():
                try:
                    results = self.recognize_screen(region, engines, preprocess_kwargs)
                except Exception as exc:  # noqa: BLE001
                    logging.exception("Real-time OCR loop failed: %s", exc)
                    break
                callback(results)
                stop_event.wait(interval)
            logging.info("Real-time OCR loop terminated")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen OCR utility for tiny text")
    parser.add_argument(
        "--engine",
        choices=["tesseract", "easyocr", "paddleocr", "all"],
        default="all",
        help="OCR engine to run",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Launch GUI to select the ROI instead of capturing the entire screen",
    )
    parser.add_argument(
        "--roi",
        type=str,
        help="Specify ROI as left,top,width,height (overrides --select-roi)",
    )
    parser.add_argument(
        "--save-screenshot",
        type=Path,
        help="Save the captured ROI/screen to the given image path",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Continuously perform OCR until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between OCR evaluations in real-time mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def _parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    components = value.split(",")
    if len(components) != 4:
        raise ValueError("ROI must have four comma-separated integers")
    return tuple(int(c.strip()) for c in components)  # type: ignore[return-value]


def _results_to_string(results: Dict[str, OCRResult]) -> str:
    parts = []
    for key, result in results.items():
        if result.error:
            parts.append(f"{key}: ERROR - {result.error}")
        else:
            parts.append(f"{key}: '{result.text}' (confidence={result.confidence:.2f})")
    return " | ".join(parts)


def main() -> None:
    args = _parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    system = OCRSystem()

    engines: Iterable[str]
    if args.engine == "all":
        engines = ("tesseract", "easyocr", "paddleocr")
    else:
        engines = (args.engine,)

    roi = _parse_roi(args.roi)
    if roi is None and args.select_roi:
        logging.info("Selecting ROI via GUI")
        roi = system.select_roi()
        if roi is None:
            logging.error("ROI selection cancelled")
            return

    if args.save_screenshot:
        path = system.save_screenshot(args.save_screenshot, roi)
        logging.info("Saved screenshot to %s", path)

    def print_results(results: Dict[str, OCRResult]):
        logging.info(_results_to_string(results))

    if args.realtime:
        if roi is None:
            logging.error("Real-time mode requires an ROI; use --select-roi or --roi")
            return

        stop_event = threading.Event()

        def _keyboard_listener():
            keyboard_module = _optional_import("keyboard")
            if keyboard_module is None:
                logging.warning(
                    "keyboard package not available; terminate with Ctrl+C",
                )
                while not stop_event.is_set():
                    time.sleep(0.2)
            else:
                logging.info("Press ESC to stop real-time OCR")
                try:
                    keyboard_module.wait("esc")
                except Exception as exc:  # noqa: BLE001
                    logging.warning("keyboard wait failed: %s", exc)
                finally:
                    stop_event.set()

        threading.Thread(target=_keyboard_listener, daemon=True).start()

        thread = system.start_realtime(
            region=roi,
            callback=print_results,
            engines=engines,
            interval=args.interval,
            stop_event=stop_event,
        )
        try:
            while thread.is_alive():
                thread.join(timeout=0.5)
        except KeyboardInterrupt:
            stop_event.set()
            thread.join()
    else:
        results = system.recognize_screen(region=roi, engines=engines)
        print_results(results)


if __name__ == "__main__":
    main()

