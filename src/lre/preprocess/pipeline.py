"""Preprocessing chain from scanned page to deskewed grayscale and binary images (report Figure 2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PreCfg
from . import steps


@dataclass
class PreResult:
    gray: np.ndarray          # deskewed, illumination-corrected, denoised grayscale (for the recogniser)
    binary: np.ndarray        # deskewed Sauvola mask, ink = 255 (for line detection only)
    skew_deg: float           # estimated skew that was removed
    timings: dict = field(default_factory=dict)


def preprocess(img: np.ndarray, cfg: PreCfg, denoise: bool = True) -> PreResult:
    import time
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    gray = img if img.ndim == 2 else np.asarray(img.mean(axis=2), dtype=np.uint8)
    gray = steps.correct_illumination(gray, cfg.background_kernel, cfg.clahe_clip, cfg.clahe_tiles)
    t["illumination"] = time.perf_counter() - t0
    if denoise:
        t0 = time.perf_counter()
        gray = steps.denoise(gray, cfg.nlmeans_h)
        t["denoise"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    skew = steps.estimate_skew(gray, cfg.deskew_max_deg)
    gray = steps.rotate(gray, -skew)
    t["deskew"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    binary = steps.binarise(gray, cfg.sauvola_window, cfg.sauvola_k)
    t["binarise"] = time.perf_counter() - t0
    return PreResult(gray=gray, binary=binary, skew_deg=skew, timings=t)
