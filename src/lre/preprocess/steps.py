"""Image preprocessing steps (report 4.4, Figure 2). Each step is a pure function on a grayscale page."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------------------------------------------------------- 1. ingest


def load_pages(path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Grayscale uint8 pages from an image file or a PDF (one array per PDF page)."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pdf2image is required for PDF input") from exc
        try:
            images = convert_from_path(str(p), dpi=dpi, grayscale=True)
        except Exception as exc:  # poppler missing or unreadable PDF
            raise RuntimeError(f"cannot rasterise {p}: {exc}. PDF input needs poppler "
                               "(macOS: brew install poppler; Ubuntu: apt install poppler-utils)") from exc
        return [np.asarray(im.convert("L")) for im in images]
    if p.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"unsupported page format: {p.suffix}")
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {p}")
    return [img]


# ----------------------------------------------------------------------------- 2. illumination + contrast


def correct_illumination(gray: np.ndarray, kernel: int = 51, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Divide by a background estimate (grey closing removes the dark ink), then CLAHE."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(flat)


# ----------------------------------------------------------------------------- 3. denoise


def denoise(gray: np.ndarray, h: int = 10) -> np.ndarray:
    """Non-local means: removes paper texture without thinning strokes."""
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


# ----------------------------------------------------------------------------- 4. deskew


def estimate_skew(gray: np.ndarray, max_deg: float = 10.0, work_width: int = 1600) -> float:
    """Skew of the page in degrees (cv2 convention: positive = counter-clockwise).

    Probabilistic Hough on the edge map finds the long ruling lines; the median of
    their deviations from horizontal / vertical is the skew. Returns 0.0 when no
    long lines are found.
    """
    scale = work_width / gray.shape[1]
    small = cv2.resize(gray, (work_width, int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    min_len = int(0.25 * min(small.shape))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=120, minLineLength=min_len, maxLineGap=15)
    if lines is None:
        return 0.0
    devs = []
    # Support both (N,1,4) and (N,4) shapes from HoughLinesP
    line_vals = lines[:, 0] if lines.ndim == 3 else lines
    for x1, y1, x2, y2 in line_vals:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = (ang + 90) % 180 - 90                      # (-90, 90]
        if abs(ang) <= max_deg:                          # near-horizontal
            devs.append(ang)
        elif abs(abs(ang) - 90) <= max_deg:              # near-vertical
            devs.append(ang - 90 if ang > 0 else ang + 90)
    if not devs:
        return 0.0
    # image y points down, so a line tilted up to the right has a negative angle
    return float(-np.median(devs))


def rotate(gray: np.ndarray, deg: float, border: int | None = None) -> np.ndarray:
    """Rotate about the centre keeping the canvas size (undo a skew with ``rotate(img, -skew)``)."""
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    if border is None:
        border = int(np.median(gray[:: max(1, h // 50), :: max(1, w // 50)]))
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


# ----------------------------------------------------------------------------- 5. binarise


def binarise(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    """Sauvola thresholding. Returns a uint8 mask with ink = 255."""
    t = threshold_sauvola(gray, window_size=window, k=k)
    return np.where(gray < t, 255, 0).astype(np.uint8)
