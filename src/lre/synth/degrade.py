"""Make a clean rendered page look like an old scanned register page (report D.6 step 3).

The geometric part is one affine map, ``M = R(skew) . S(scale, shift)``, applied
about the page centre with the canvas size kept. Ground truth stores boxes in the
*deskewed frame* (``S`` applied, ``R`` not), which is exactly what a perfect
deskew recovers, plus ``M`` itself for anything needed in the scanned frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import SynthCfg


@dataclass
class ScanTransform:
    skew_deg: float
    scale: float
    tx: float
    ty: float
    width: int
    height: int

    def scale_shift(self) -> np.ndarray:
        """3x3: form frame -> deskewed frame (scale about the centre, then shift)."""
        cx, cy = self.width / 2, self.height / 2
        return np.array([[self.scale, 0, cx - self.scale * cx + self.tx],
                         [0, self.scale, cy - self.scale * cy + self.ty],
                         [0, 0, 1]], dtype=np.float64)

    def rotation(self) -> np.ndarray:
        """3x3: deskewed frame -> scanned frame (rotation about the centre)."""
        r = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), self.skew_deg, 1.0)
        return np.vstack([r, [0, 0, 1]])

    def full(self) -> np.ndarray:
        return self.rotation() @ self.scale_shift()

    def box_to_deskewed(self, box) -> list[int]:
        x, y, w, h = box
        m = self.scale_shift()
        x0, y0 = m[0, 0] * x + m[0, 2], m[1, 1] * y + m[1, 2]
        return [int(round(x0)), int(round(y0)), int(round(w * self.scale)), int(round(h * self.scale))]

    def box_inside_scan(self, box) -> bool:
        """True if all four corners of a form-frame box land inside the scanned image."""
        x, y, w, h = box
        pts = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], dtype=np.float64).T
        sx, sy = (self.full() @ pts)[:2]
        return bool((sx >= 0).all() and (sx < self.width).all() and (sy >= 0).all() and (sy < self.height).all())

    def point_to_deskewed(self, x: float, y: float) -> tuple[float, float]:
        m = self.scale_shift()
        return m[0, 0] * x + m[0, 2], m[1, 1] * y + m[1, 2]


def sample_transform(rng: np.random.Generator, cfg: SynthCfg, width: int, height: int) -> ScanTransform:
    # Most scans are nearly straight: centred on 0, capped at the configured maximum.
    skew = float(np.clip(rng.normal(0, cfg.max_skew_deg / 3), -cfg.max_skew_deg, cfg.max_skew_deg))
    return ScanTransform(
        skew_deg=skew,
        scale=float(rng.uniform(0.985, 1.015)),
        tx=float(rng.uniform(-0.008, 0.008) * width),
        ty=float(rng.uniform(-0.008, 0.008) * height),
        width=width, height=height,
    )


def _lowfreq(rng: np.random.Generator, h: int, w: int, cells: int = 12) -> np.ndarray:
    small = rng.normal(0, 1, size=(cells, max(2, int(cells * w / h)))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def degrade(img: np.ndarray, paper: tuple[int, int, int], tf: ScanTransform, cfg: SynthCfg,
            rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """img: HxWx3 uint8 RGB clean page. Returns degraded RGB page and the parameters used."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    paper_f = np.array(paper, dtype=np.float32)
    info: dict = {}

    if rng.random() < cfg.fade_prob:  # faded ink in large patches
        amount = float(rng.uniform(0.2, 0.55))
        mask = np.clip(_lowfreq(rng, h, w, 6) * 0.5 + 0.2, 0, 1) * amount
        out = out + (paper_f - out) * mask[..., None]
        info["fade"] = round(amount, 3)

    if rng.random() < cfg.bleed_prob:  # bleed-through of the reverse side
        alpha = float(rng.uniform(0.05, 0.13))
        ink = (paper_f.mean() - out[:, ::-1].mean(axis=2)).clip(0, None)
        ink = cv2.GaussianBlur(ink, (0, 0), 2.5)
        shift = int(rng.integers(-40, 40))
        ink = np.roll(ink, shift, axis=0)
        out = out - alpha * ink[..., None] * np.array([0.9, 1.0, 1.1], dtype=np.float32)
        info["bleed"] = round(alpha, 3)

    if rng.random() < cfg.stain_prob:  # tea / water stains
        n = int(rng.integers(1, 5))
        stain = np.zeros((h, w), np.float32)
        for _ in range(n):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(w // 30, w // 8)), int(rng.integers(h // 30, h // 8)))
            cv2.ellipse(stain, c, axes, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.3, 1.0)), -1)
        stain = cv2.GaussianBlur(stain, (0, 0), w / 200) * float(rng.uniform(0.08, 0.2))
        tint = np.array([0.55, 0.75, 1.0], dtype=np.float32)  # browner: blue absorbed most
        out = out * (1 - stain[..., None] * tint)
        info["stains"] = n

    # scan geometry: skew, small scale, shift
    m = tf.full()[:2]
    out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=tuple(float(v) for v in paper))

    # paper texture: uneven tone plus grain
    tone = 1 + 0.035 * _lowfreq(rng, h, w, 10)
    out = out * tone[..., None] + rng.normal(0, 3.0, size=(h, w, 1)).astype(np.float32)

    if rng.random() < cfg.blur_prob:
        sigma = float(rng.uniform(0.5, 1.2))
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        info["blur"] = round(sigma, 2)

    out = out.clip(0, 255).astype(np.uint8)
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    info["jpeg_quality"] = q
    return out, info
