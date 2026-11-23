"""Cells and text-line crops (report 4.4 step 7)."""
from __future__ import annotations

import cv2
import numpy as np

from ..config import LayoutCfg
from ..preprocess.pipeline import PreResult
from ..schema import COLUMN_FIELDS, Cell
from .grid import Grid, detect_grid


def body_rows(grid: Grid) -> list[tuple[int, int]]:
    """(y0, y1) of every row below the header (y_lines[0] = table top, y_lines[1] = header bottom)."""
    return list(zip(grid.y_lines[1:-1], grid.y_lines[2:]))


def find_cells(pre: PreResult, cfg: LayoutCfg, page_id: str = "", grid: Grid | None = None,
               min_ink: float = 0.002, unet=None) -> tuple[Grid, list[Cell]]:
    """Grid cells of every body row that contains handwriting; the column index gives the field type.

    If the morphological grid fails its sanity check and a U-Net-lite model is given, the
    rules are re-detected from the network's line masks (report 5.3.1).
    """
    grid = grid or detect_grid(pre.binary, cfg, pre.gray)
    if not grid.ok and unet is not None:
        from .grid import grid_from_masks
        from .unet import predict_masks
        h_mask, v_mask = predict_masks(unet, pre.gray)
        fallback = grid_from_masks(h_mask, v_mask, cfg, pre.gray, method="unet")
        if fallback.ok:
            grid = fallback
        else:
            grid.reason += f"; U-Net fallback: {fallback.reason}"
    if not grid.ok:
        return grid, []
    # ink without the ruling lines, used to skip empty rows
    rules = cv2.bitwise_or(grid.line_masks["h"], grid.line_masks["v"])
    ink = cv2.bitwise_and(pre.binary, cv2.bitwise_not(rules))
    cells: list[Cell] = []
    row_idx = 0
    for (y0, y1) in body_rows(grid):
        row_ink = ink[y0:y1, grid.x_lines[0]:grid.x_lines[-1]]
        if row_ink.size == 0 or (row_ink > 0).mean() < min_ink:
            continue
        for c in range(grid.n_columns):
            x0, x1 = grid.x_lines[c], grid.x_lines[c + 1]
            p = cfg.cell_padding
            bbox = (x0 + p, y0 + p, max(1, x1 - x0 - 2 * p), max(1, y1 - y0 - 2 * p))
            bx, by, bw, bh = bbox
            cells.append(Cell(page_id=page_id, row=row_idx, col=c, field_type=COLUMN_FIELDS[c].value,
                              bbox=bbox, crop=pre.gray[by:by + bh, bx:bx + bw]))
        row_idx += 1
    return grid, cells


def split_lines(crop_gray: np.ndarray, binary: np.ndarray | None = None, min_gap: int = 6,
                min_height: int = 12, ink_frac: float = 0.002) -> list[tuple[int, int]]:
    """Split a cell into text lines by its horizontal projection profile. Returns (y0, y1) spans."""
    if binary is None:
        _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    if h == 0 or w == 0:
        return []
    # drop remnants of ruling lines touching the cell border
    m = binary.copy()
    m[:3, :] = 0
    m[-3:, :] = 0
    m[:, :3] = 0
    m[:, -3:] = 0
    profile = (m > 0).sum(axis=1) / max(1, w)
    on = profile > ink_frac
    spans, start, gap = [], None, 0
    for y, v in enumerate(on):
        if v:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                end = y - gap + 1
                if end - start >= min_height:
                    spans.append((start, end))
                start, gap = None, 0
    if start is not None and h - start >= min_height:
        spans.append((start, h))
    return spans


def normalise_line(crop: np.ndarray, height: int = 64, buckets: tuple[int, ...] = (128, 256, 384, 512),
                   pad_value: int = 255) -> np.ndarray:
    """Resize to a fixed height keeping aspect ratio, then right-pad to the nearest width bucket."""
    h, w = crop.shape[:2]
    new_w = max(1, int(round(w * height / max(1, h))))
    max_w = buckets[-1]
    if new_w > max_w:
        new_w = max_w
    resized = cv2.resize(crop, (new_w, height), interpolation=cv2.INTER_AREA)
    target = next(b for b in buckets if b >= new_w)
    out = np.full((height, target), pad_value, dtype=crop.dtype)
    out[:, :new_w] = resized
    return out
