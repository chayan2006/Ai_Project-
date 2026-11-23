"""U-Net-lite ruling-line segmenter (report 5.3.1).

Four encoder levels (16, 32, 64, 128 channels), about 0.5 M parameters. Input:
512 x 512 tiles of the deskewed page at half resolution. Output: two masks,
horizontal and vertical rules. Loss: BCE + Dice. Trained on SynJama pages, whose
line masks are known exactly, with augmentations that fade and break the rules.
It runs only when the morphological grid fails its sanity check.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 0.5
TILE = 512


# ----------------------------------------------------------------------------- model


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetLite(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, widths=(16, 32, 64, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.enc.append(_block(c, w))
            c = w
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return bce + dice.mean()


# ----------------------------------------------------------------------------- data


def masks_from_gt(gt: dict, shape: tuple[int, int], scale: float = SCALE) -> np.ndarray:
    """2 x H x W uint8 masks (horizontal, vertical rules) in the deskewed frame at ``scale``."""
    h, w = shape
    m = np.zeros((2, h, w), np.uint8)
    for ln in gt["h_lines"]:
        y = int(round(ln["y"] * scale))
        t = max(2, int(round(ln["w"] * scale)) + 1)
        cv2.line(m[0], (int(ln["x0"] * scale), y), (int(ln["x1"] * scale), y), 1, t)
    for ln in gt["v_lines"]:
        x = int(round(ln["x"] * scale))
        t = max(2, int(round(ln["w"] * scale)) + 1)
        cv2.line(m[1], (x, int(ln["y0"] * scale)), (x, int(ln["y1"] * scale)), 1, t)
    return m


def page_input(gray_deskewed: np.ndarray, scale: float = SCALE) -> np.ndarray:
    h, w = gray_deskewed.shape
    return cv2.resize(gray_deskewed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def fade_and_break(img: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
                   n_fade=(5, 40), fade_size=(30, 300), n_break=(5, 60), gap=(6, 40)) -> np.ndarray:
    """Augmentation / stress test: lighten random stretches of the rules and cut gaps into them."""
    out = img.astype(np.float32)
    paper = float(np.percentile(img, 90))
    lines = (masks.max(axis=0) > 0)
    dil = cv2.dilate(lines.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    h, w = img.shape
    region = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(*n_fade))):         # faded stretches
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        rw, rh = int(rng.integers(*fade_size)), int(rng.integers(*fade_size))
        region[y:y + rh, x:x + rw] = max(region[y:y + rh, x:x + rw].max(initial=0), rng.uniform(0.5, 0.9))
    for _ in range(int(rng.integers(*n_break))):        # breaks
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(*gap))
        region[y:y + s, x:x + s] = 1.0
    a = region * dil
    out = out + (paper - out) * a
    return out.clip(0, 255).astype(np.uint8)


class TileSampler:
    """Random 512 x 512 tiles from cached half-resolution pages, biased towards the table area."""

    def __init__(self, pages: list[tuple[np.ndarray, np.ndarray]], seed: int = 0, augment: bool = True):
        self.pages = pages
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n):
            img, m = self.pages[int(self.rng.integers(len(self.pages)))]
            h, w = img.shape
            y0 = int(self.rng.integers(0, max(1, h - TILE)))
            x0 = int(self.rng.integers(0, max(1, w - TILE)))
            ti = img[y0:y0 + TILE, x0:x0 + TILE]
            tm = m[:, y0:y0 + TILE, x0:x0 + TILE]
            if self.augment:
                if self.rng.random() < 0.7:
                    ti = fade_and_break(ti, tm, self.rng)
                ti = np.clip(ti.astype(np.float32) * self.rng.uniform(0.8, 1.2) + self.rng.normal(0, 8, ti.shape),
                             0, 255).astype(np.uint8)
            xs.append(ti)
            ys.append(tm)
        x = torch.from_numpy(np.stack(xs)[:, None].astype(np.float32) / 255.0)
        y = torch.from_numpy(np.stack(ys).astype(np.float32))
        return 1.0 - x, y                               # ink bright, paper dark


def load_pages(data_dir: str | Path, limit: int, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    from ..preprocess.pipeline import preprocess
    out = []
    for p in sorted(Path(data_dir, "gt").glob("*.json"))[:limit]:
        gt = json.loads(p.read_text(encoding="utf-8"))
        gray = cv2.imread(str(Path(data_dir) / gt["image"]), cv2.IMREAD_GRAYSCALE)
        pre = preprocess(gray, cfg.preprocess, denoise=False)
        img = page_input(pre.gray)
        out.append((img, masks_from_gt(gt, img.shape)))
    return out


# ----------------------------------------------------------------------------- training / inference


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def mask_iou(model: nn.Module, sampler: TileSampler, n_batches: int, dev) -> list[float]:
    model.eval()
    inter = np.zeros(2)
    union = np.zeros(2)
    for _ in range(n_batches):
        x, y = sampler.batch(8)
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        inter += (p * y).sum(dim=(0, 2, 3)).numpy()
        union += ((p + y) > 0).float().sum(dim=(0, 2, 3)).numpy()
    model.train()
    return (inter / np.maximum(union, 1)).tolist()


def train(train_dir: str | Path, val_dir: str | Path, cfg, out: str | Path, steps: int = 1500,
          n_train_pages: int = 120, n_val_pages: int = 20, batch: int = 8, lr: float = 1e-3,
          seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed)
    dev = device()
    t0 = time.time()
    tr = TileSampler(load_pages(train_dir, n_train_pages, cfg), seed=seed, augment=True)
    va = TileSampler(load_pages(val_dir, n_val_pages, cfg), seed=seed + 1, augment=True)
    log(f"loaded {len(tr.pages)} train / {len(va.pages)} val pages in {time.time() - t0:.0f}s; device {dev}")
    model = UNetLite().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    history = []
    for step in range(1, steps + 1):
        x, y = tr.batch(batch)
        loss = dice_bce(model(x.to(dev)), y.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps:
            iou = mask_iou(model, va, 6, dev)
            history.append({"step": step, "loss": round(float(loss), 4), "val_iou_h": round(iou[0], 4),
                            "val_iou_v": round(iou[1], 4)})
            log(f"step {step}: loss {float(loss):.4f}  val IoU h {iou[0]:.3f} v {iou[1]:.3f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "widths": (16, 32, 64, 128), "scale": SCALE}, out)
    card = {"model": "UNetLite", "params": n_params(model), "train_data": str(train_dir), "val_data": str(val_dir),
            "train_pages": len(tr.pages), "steps": steps, "batch": batch, "lr": lr, "seed": seed,
            "augmentation": "fade/break rules, brightness, noise", "history": history,
            "val_iou": history[-1] if history else None, "minutes": round((time.time() - t0) / 60, 1),
            "licence": "trained only on synthetic SynJama pages"}
    out.with_suffix(".card.json").write_text(json.dumps(card, indent=2))
    return card


_MODEL_CACHE: dict = {}


def load_model(path: str | Path) -> Optional[nn.Module]:
    p = Path(path)
    if not p.exists():
        return None
    if str(p) not in _MODEL_CACHE:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        m = UNetLite(widths=tuple(ckpt["widths"]))
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _MODEL_CACHE[str(p)] = m
    return _MODEL_CACHE[str(p)]


@torch.no_grad()
def predict_masks(model: nn.Module, gray_deskewed: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution uint8 masks (255 = rule) for horizontal and vertical rules, by overlapping tiles."""
    img = page_input(gray_deskewed)
    h, w = img.shape
    ph, pw = int(np.ceil(h / 256) * 256), int(np.ceil(w / 256) * 256)
    pad = np.full((max(ph, TILE), max(pw, TILE)), 255, np.uint8)
    pad[:h, :w] = img
    prob = np.zeros((2, *pad.shape), np.float32)
    cnt = np.zeros(pad.shape, np.float32)
    step = TILE // 2
    for y0 in range(0, pad.shape[0] - TILE + 1, step):
        for x0 in range(0, pad.shape[1] - TILE + 1, step):
            t = 1.0 - torch.from_numpy(pad[y0:y0 + TILE, x0:x0 + TILE][None, None].astype(np.float32) / 255.0)
            prob[:, y0:y0 + TILE, x0:x0 + TILE] += torch.sigmoid(model(t))[0].numpy()
            cnt[y0:y0 + TILE, x0:x0 + TILE] += 1
    prob = prob[:, :h, :w] / np.maximum(cnt[:h, :w], 1)
    full = [cv2.resize(prob[i], (gray_deskewed.shape[1], gray_deskewed.shape[0]), interpolation=cv2.INTER_LINEAR)
            for i in range(2)]
    return tuple(((m > threshold) * 255).astype(np.uint8) for m in full)
