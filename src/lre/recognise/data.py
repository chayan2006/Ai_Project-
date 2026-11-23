"""Line-image datasets for the CRNN.

`build_lines` cuts every ground-truth text line out of SynJama pages after the same
preprocessing used at inference, normalises it to 64 px height and packs all lines of
a split into one memory-mapped uint8 file (64 x total width) plus a JSON index, so a
100 k-line split loads instantly and costs no RAM until a batch is read.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import Config
from .charset import encode, normalise_label

HEIGHT = 64
MAX_WIDTH = 1024


def line_to_input(crop: np.ndarray, height: int = HEIGHT, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Resize a grayscale line crop to ``height`` keeping the aspect ratio (squash if wider than max)."""
    h, w = crop.shape[:2]
    new_w = int(np.clip(round(w * height / max(1, h)), 8, max_width))
    return cv2.resize(crop, (new_w, height), interpolation=cv2.INTER_AREA)


def _page_lines(gt_path: str, data_dir: str, cfg: Config, pad: int = 6,
                margin: int = 12) -> list[tuple[np.ndarray, str, str]]:
    # NL-means is local (21 px search window), so denoising each line crop with a margin gives the
    # same lines as denoising the whole page (0-4 grey levels apart, from rotating first) at a third
    # of the cost: the full page is ~90 % of preprocessing time and lines cover a small part of it.
    from ..preprocess import steps
    from ..preprocess.pipeline import preprocess
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    gray = cv2.imread(str(Path(data_dir) / gt["image"]), cv2.IMREAD_GRAYSCALE)
    pre = preprocess(gray, cfg.preprocess, denoise=False)
    H, W = pre.gray.shape
    out = []
    for h in gt["holdings"]:
        for cell in h["cells"]:
            for ln in cell["lines"]:
                if ln.get("clipped") or not normalise_label(ln["text"]):
                    continue
                x, y, w, hh = ln["bbox"]
                y0, y1, x0, x1 = max(0, y - pad), min(H, y + hh + pad), max(0, x - pad), min(W, x + w + pad)
                if y1 <= y0 or x1 <= x0:
                    continue
                Y0, X0 = max(0, y0 - margin), max(0, x0 - margin)
                big = steps.denoise(np.ascontiguousarray(pre.gray[Y0:min(H, y1 + margin), X0:min(W, x1 + margin)]),
                                    cfg.preprocess.nlmeans_h)
                crop = big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
                out.append((line_to_input(crop), normalise_label(ln["text"]), cell["field"]))
    return out


def build_lines(split_dir: str | Path, out_dir: str | Path, cfg: Config, workers: int = 6,
                limit: int | None = None) -> dict:
    from tqdm import tqdm
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gts = [str(p) for p in sorted((split_dir / "gt").glob("*.json"))[:limit]]
    raw_path = out_dir / "lines.u8"
    index, offset = [], 0
    with open(raw_path, "wb") as fh, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_page_lines, g, str(split_dir), cfg) for g in gts]
        for g, f in tqdm(zip(gts, futs), total=len(futs), desc=f"lines {split_dir.name}", unit="page"):
            for img, text, field in f.result():
                fh.write(np.ascontiguousarray(img.T).tobytes())   # column-major: a line is a contiguous slice
                index.append({"offset": offset, "width": img.shape[1], "text": text, "field": field,
                              "page": Path(g).stem})
                offset += img.shape[1]
    meta = {"height": HEIGHT, "total_width": offset, "lines": len(index), "source": str(split_dir)}
    (out_dir / "index.json").write_text(json.dumps({"meta": meta, "lines": index}, ensure_ascii=False))
    return meta


class LineStore:
    """Read-only view over a packed split."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        d = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        self.meta, self.lines = d["meta"], d["lines"]
        self.raw = np.memmap(folder / "lines.u8", dtype=np.uint8, mode="r",
                             shape=(self.meta["total_width"], self.meta["height"]))

    def __len__(self) -> int:
        return len(self.lines)

    def image(self, i: int) -> np.ndarray:
        rec = self.lines[i]
        return np.asarray(self.raw[rec["offset"]:rec["offset"] + rec["width"]]).T


# ----------------------------------------------------------------------------- augmentation and batching


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Report Table 6: small affine (rotate 3 deg, shear 0.3), erosion / dilation, blur, ink fading, JPEG."""
    h, w = img.shape
    if rng.random() < 0.7:
        ang = np.deg2rad(rng.uniform(-3, 3))
        sh = rng.uniform(-0.3, 0.3)
        m = np.array([[np.cos(ang), -np.sin(ang) + sh, 0], [np.sin(ang), np.cos(ang), 0]], np.float32)
        m[:, 2] = np.array([w / 2, h / 2]) - m[:, :2] @ np.array([w / 2, h / 2])
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    r = rng.random()
    if r < 0.15:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))       # thicker ink (ink is dark)
    elif r < 0.3:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))      # thinner ink
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:                                    # ink fading
        paper = float(np.percentile(img, 95))
        img = (img + (paper - img.astype(np.float32)) * rng.uniform(0.1, 0.45)).clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


@dataclass
class Batch:
    images: torch.Tensor          # B x 1 x 64 x W, ink bright in [0, 1]
    targets: torch.Tensor         # concatenated label ids
    target_lengths: torch.Tensor
    input_lengths: torch.Tensor   # T = width / 4 per item
    texts: list[str]
    fields: list[str]


def collate(images: list[np.ndarray], texts: list[str], fields: list[str]) -> Batch:
    width = max(8, int(np.ceil(max(im.shape[1] for im in images) / 4) * 4))
    arr = np.zeros((len(images), 1, HEIGHT, width), np.float32)
    for i, im in enumerate(images):
        arr[i, 0, :, :im.shape[1]] = 1.0 - im.astype(np.float32) / 255.0
    labels = [encode(t) for t in texts]
    return Batch(torch.from_numpy(arr), torch.tensor([c for lab in labels for c in lab], dtype=torch.long),
                 torch.tensor([len(lab) for lab in labels], dtype=torch.long),
                 torch.tensor([max(1, im.shape[1] // 4) for im in images], dtype=torch.long), texts, fields)


def bucketed_batches(store: LineStore, batch_size: int, rng: np.random.Generator,
                     shuffle: bool = True, limit: int | None = None):
    """Index batches of similar width (less padding); order of batches is shuffled."""
    order = np.arange(len(store)) if limit is None else rng.permutation(len(store))[:limit]
    widths = np.array([store.lines[i]["width"] for i in order])
    if shuffle:
        order = order[np.argsort(widths + rng.uniform(0, 64, len(widths)), kind="stable")]
    else:
        order = order[np.argsort(widths, kind="stable")]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches
