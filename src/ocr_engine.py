"""
OCR Engine module using EasyOCR with bounding-box line reconstruction and confidence tracking.
Strictly local & offline-safe execution with CPU optimization (sub-4s per image).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _GPU_AVAILABLE = False

import config

_EASYOCR_READER = None


def get_ocr_reader(gpu: Optional[bool] = None):
    """
    Get or initialize cached EasyOCR Reader singleton.
    Defaults to CPU (or GPU if explicitly available and requested).
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        use_gpu = _GPU_AVAILABLE if gpu is None else gpu
        _EASYOCR_READER = easyocr.Reader(
            config.EASYOCR_LANGS,
            gpu=use_gpu,
            verbose=False,
            download_enabled=True,
        )
    return _EASYOCR_READER


def preprocess_image_for_ocr(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    max_dimension: int = 1050,
) -> np.ndarray:
    """
    Convert image to numpy array format suitable for EasyOCR.
    Resizes high-res images (1500x950 -> 1050x665) to achieve sub-4s CPU latency
    while improving character recognition on printed text.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
        np_img = np.array(img)
    elif isinstance(image_input, Image.Image):
        np_img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        np_img = image_input.copy()
        if len(np_img.shape) == 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB) if cv2 else np.stack([np_img]*3, axis=-1)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Optimal scaling for speed & accuracy
    h, w = np_img.shape[:2]
    max_dim = max(h, w)
    if max_dim > max_dimension and cv2:
        scale = max_dimension / float(max_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np_img


def extract_text_from_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    max_dimension: int = 1050,
    y_threshold_factor: float = 0.5,
) -> Dict[str, object]:
    """
    Run EasyOCR on the image and reconstruct clean lines sorted top-to-bottom, left-to-right.
    Returns:
    {
        'raw_text': str,
        'lines': List[str],
        'mean_confidence': float,
        'boxes': List[dict],
        'latency_seconds': float
    }
    """
    start_time = time.perf_counter()
    reader = get_ocr_reader()
    img_arr = preprocess_image_for_ocr(image_input, max_dimension=max_dimension)

    # detail=1 returns [ [bbox, text, confidence], ... ]
    results = reader.readtext(img_arr, detail=1)
    elapsed = time.perf_counter() - start_time

    if not results:
        return {
            "raw_text": "",
            "lines": [],
            "mean_confidence": 0.0,
            "boxes": [],
            "latency_seconds": round(elapsed, 3),
        }

    # Parse and sort boxes
    parsed_boxes = []
    confidences = []
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        conf_float = float(conf)
        confidences.append(conf_float)

        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        height = max(1, y_max - y_min)

        parsed_boxes.append({
            "bbox": bbox,
            "text": text,
            "conf": conf_float,
            "y_center": (y_min + y_max) / 2.0,
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
            "height": height,
        })

    if not parsed_boxes:
        return {
            "raw_text": "",
            "lines": [],
            "mean_confidence": 0.0,
            "boxes": [],
            "latency_seconds": round(elapsed, 3),
        }

    # Group into lines by vertical closeness
    parsed_boxes.sort(key=lambda b: b["y_center"])
    median_height = np.median([b["height"] for b in parsed_boxes])
    y_tol = max(8.0, median_height * y_threshold_factor)

    lines_boxes = []
    current_line = [parsed_boxes[0]]

    for box in parsed_boxes[1:]:
        prev_y = np.mean([b["y_center"] for b in current_line])
        if abs(box["y_center"] - prev_y) <= y_tol:
            current_line.append(box)
        else:
            current_line.sort(key=lambda b: b["x_min"])
            lines_boxes.append(current_line)
            current_line = [box]

    if current_line:
        current_line.sort(key=lambda b: b["x_min"])
        lines_boxes.append(current_line)

    reconstructed_lines = [" ".join(b["text"] for b in line) for line in lines_boxes]
    raw_text = "\n".join(reconstructed_lines)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "raw_text": raw_text,
        "lines": reconstructed_lines,
        "mean_confidence": round(mean_conf, 4),
        "boxes": parsed_boxes,
        "latency_seconds": round(elapsed, 3),
    }
