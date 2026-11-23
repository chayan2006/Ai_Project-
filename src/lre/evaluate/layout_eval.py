"""Phase 2 acceptance check: column count on >= 95 % of pages, mean cell IoU >= 0.90 (report D.3)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..layout.cells import body_rows, find_cells, split_lines
from ..preprocess.pipeline import preprocess


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def stress_rules(pre, gt: dict, cfg: Config, seed: int) -> None:
    """Fade and break the ruling lines of a page in place (faded registers, report 5.3.1)."""
    from ..layout.unet import fade_and_break, masks_from_gt
    from ..preprocess.steps import binarise
    rng = np.random.default_rng(seed)
    masks = masks_from_gt(gt, pre.gray.shape, scale=1.0)
    pre.gray = fade_and_break(pre.gray, masks, rng, n_fade=(40, 120), fade_size=(150, 900),
                              n_break=(150, 400), gap=(20, 90))
    pre.binary = binarise(pre.gray, cfg.preprocess.sauvola_window, cfg.preprocess.sauvola_k)


def evaluate_page(gt: dict, data_dir: Path, cfg: Config, unet=None, stress_seed: int | None = None) -> dict:
    img = cv2.imread(str(data_dir / gt["image"]), cv2.IMREAD_GRAYSCALE)
    pre = preprocess(img, cfg.preprocess)
    if stress_seed is not None:
        stress_rules(pre, gt, cfg, stress_seed)
    grid, cells = find_cells(pre, cfg.layout, gt["page_id"], unet=unet)
    res = {"page_id": gt["page_id"], "skew_err": abs(pre.skew_deg - gt["skew_deg"]),
           "n_columns": grid.n_columns, "columns_ok": grid.n_columns == cfg.layout.n_columns,
           "grid_ok": grid.ok, "reason": grid.reason, "method": grid.method, "gt_rows": len(gt["holdings"]),
           "found_rows": len({c.row for c in cells}), "cell_ious": [], "line_count_ok": []}
    if not grid.ok:
        res["cell_ious"] = [0.0] * sum(len(h["cells"]) for h in gt["holdings"])
        return res
    # unpadded detected cell boxes for every body row, by column
    det = {c: [(grid.x_lines[c], y0, grid.x_lines[c + 1] - grid.x_lines[c], y1 - y0) for y0, y1 in body_rows(grid)]
           for c in range(grid.n_columns)}
    for h in gt["holdings"]:
        for cell in h["cells"]:
            col = cell["col"]
            best = max(det[col], key=lambda b: iou(b, cell["bbox"]))
            res["cell_ious"].append(iou(best, cell["bbox"]))
            # text-line split on the matched cell, cells with at least one GT line only
            if cell["lines"]:
                x, y, w, hh = best
                p = cfg.layout.cell_padding
                crop_bin = pre.binary[y + p:y + hh - p, x + p:x + w - p]
                spans = split_lines(pre.gray[y + p:y + hh - p, x + p:x + w - p], crop_bin)
                res["line_count_ok"].append(len(spans) == len(cell["lines"]))
    return res


def evaluate_layout(data_dir: str | Path, cfg: Config, limit: int | None = None, unet=None,
                    stress: bool = False) -> dict:
    data_dir = Path(data_dir)
    gts = sorted((data_dir / "gt").glob("*.json"))[:limit]
    if not gts:
        raise FileNotFoundError(f"no ground truth in {data_dir / 'gt'}")
    pages = [evaluate_page(json.loads(p.read_text(encoding="utf-8")), data_dir, cfg, unet,
                           stress_seed=i if stress else None)
             for i, p in enumerate(tqdm(gts, desc="layout eval", unit="page"))]
    all_iou = [v for p in pages for v in p["cell_ious"]]
    all_lines = [v for p in pages for v in p["line_count_ok"]]
    col_acc = float(np.mean([p["columns_ok"] for p in pages]))
    mean_iou = float(np.mean(all_iou)) if all_iou else 0.0
    return {
        "pages": len(pages),
        "acceptance_passed": col_acc >= 0.95 and mean_iou >= 0.90,
        "column_count_accuracy": col_acc,
        "grid_ok_rate": float(np.mean([p["grid_ok"] for p in pages])),
        "mean_cell_iou": mean_iou,
        "cells_iou_ge_0.9": float(np.mean([v >= 0.9 for v in all_iou])) if all_iou else 0.0,
        "row_count_accuracy": float(np.mean([p["found_rows"] == p["gt_rows"] for p in pages])),
        "line_split_accuracy": float(np.mean(all_lines)) if all_lines else 0.0,
        "mean_skew_err_deg": float(np.mean([p["skew_err"] for p in pages])),
        "max_skew_err_deg": float(np.max([p["skew_err"] for p in pages])),
        "unet_fallback_used": sum(p["method"] == "unet" for p in pages),
        "stress": stress,
        "failures": [{"page_id": p["page_id"], "reason": p["reason"]} for p in pages if not p["grid_ok"]],
    }
