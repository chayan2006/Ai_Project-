"""Ruling-line grid detection by morphology (report 4.4 step 6).

Opening the ink mask with a long horizontal kernel (width/30 x 1) keeps only
horizontal rules; a tall vertical kernel (1 x height/40) keeps only vertical
rules. Devanagari headlines (shirorekha) and letter stems survive the opening
only as short pieces, so line pieces are grouped by position and a group counts
as a ruling line only if it spans a large part of the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import LayoutCfg


@dataclass
class Grid:
    x_lines: list[int]                 # vertical rules, left to right
    y_lines: list[int]                 # horizontal rules, top to bottom (table top, header bottom, rows..., bottom)
    ok: bool
    reason: str = ""
    line_masks: dict = field(default_factory=dict, repr=False)
    method: str = "morphology"          # or "unet" when the fallback produced the grid

    @property
    def n_columns(self) -> int:
        return max(0, len(self.x_lines) - 1)


def _line_groups(mask: np.ndarray, axis: str, min_piece: int, tol: int) -> list[tuple[float, int, int, int]]:
    """Group line pieces by their centre coordinate.

    Returns (position, covered_length, start, end) per group, where position is the
    length-weighted centre and covered_length the summed length of the pieces.
    """
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if axis == "h" and w >= min_piece:
            pieces.append((cents[i][1], w, x, x + w))
        elif axis == "v" and h >= min_piece:
            pieces.append((cents[i][0], h, y, y + h))
    pieces.sort()
    groups: list[list] = []
    for pos, length, a, b in pieces:
        if groups and pos - groups[-1][-1][0] <= tol:
            groups[-1].append((pos, length, a, b))
        else:
            groups.append([(pos, length, a, b)])
    out = []
    for g in groups:
        total = sum(p[1] for p in g)
        centre = sum(p[0] * p[1] for p in g) / total
        out.append((centre, total, min(p[2] for p in g), max(p[3] for p in g)))
    return out


def detect_grid(binary: np.ndarray, cfg: LayoutCfg, gray: np.ndarray | None = None,
                min_cover: float = 0.5) -> Grid:
    h, w = binary.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // cfg.h_kernel_frac), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // cfg.v_kernel_frac)))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    return grid_from_masks(h_mask, v_mask, cfg, gray, min_cover)


def grid_from_masks(h_mask: np.ndarray, v_mask: np.ndarray, cfg: LayoutCfg, gray: np.ndarray | None = None,
                    min_cover: float = 0.5, method: str = "morphology") -> Grid:
    """Group rule pieces from line masks (morphology or U-Net) into a checked grid."""
    h, w = h_mask.shape
    # close small gaps along each line so faded rules stay in one piece
    h_mask = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    v_mask = cv2.dilate(v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    if method == "unet":  # the network bridges gaps, but pieces can still be short: join along the line
        h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 40, 1)))
        v_mask = cv2.morphologyEx(v_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40)))

    v_groups = _line_groups(v_mask, "v", h // cfg.v_kernel_frac, tol=max(8, w // 300))
    if not v_groups:
        return Grid([], [], False, "no vertical rules", {"h": h_mask, "v": v_mask})
    longest_v = max(g[1] for g in v_groups)
    v_keep = [g for g in v_groups if g[1] >= min_cover * longest_v]

    h_groups = _line_groups(h_mask, "h", w // cfg.h_kernel_frac, tol=max(8, h // 300))
    table_w = (v_keep[-1][0] - v_keep[0][0]) if len(v_keep) >= 2 else w
    h_keep = [g for g in h_groups if g[1] >= min_cover * table_w]

    if gray is not None and method == "morphology":   # faded rules are exactly what the U-Net recovers
        v_keep = _drop_faint(v_keep, gray, "v")
        h_keep = _drop_faint(h_keep, gray, "h")
    min_gap = max(8, cfg.line_height // 4)
    xs = _dedupe([int(round(g[0])) for g in v_keep], [_darkness(g, gray, "v") for g in v_keep], min_gap)
    ys = _dedupe([int(round(g[0])) for g in h_keep], [_darkness(g, gray, "h") for g in h_keep], min_gap)

    grid = Grid(xs, ys, True, "", {"h": h_mask, "v": v_mask}, method=method)
    return check_grid(grid, cfg)


def _darkness(group, gray: np.ndarray | None, axis: str) -> float:
    """Median darkness (255 - grey) along a candidate rule; real rules are dark, bleed-through ghosts faint."""
    if gray is None:
        return 0.0
    pos, _, a, b = group
    p = int(round(pos))
    if axis == "h":
        band = gray[max(0, p - 3):p + 4, a:b]
        prof = band.min(axis=0) if band.size else np.array([255])
    else:
        band = gray[a:b, max(0, p - 3):p + 4]
        prof = band.min(axis=1) if band.size else np.array([255])
    return float(255 - np.median(prof))


def _drop_faint(groups: list, gray: np.ndarray, axis: str, rel: float = 0.5) -> list:
    if not groups:
        return groups
    dark = [_darkness(g, gray, axis) for g in groups]
    ref = float(np.percentile(dark, 90))
    return [g for g, d in zip(groups, dark) if d >= rel * ref]


def _dedupe(positions: list[int], darkness: list[float], min_gap: int) -> list[int]:
    """Of two rules closer than ``min_gap`` px keep the darker one (a ghost beside a real rule)."""
    out: list[tuple[int, float]] = []
    for p, d in sorted(zip(positions, darkness)):
        if out and p - out[-1][0] < min_gap:
            if d > out[-1][1]:
                out[-1] = (p, d)
        else:
            out.append((p, d))
    return [p for p, _ in out]


def check_grid(grid: Grid, cfg: LayoutCfg) -> Grid:
    """Sanity checks that trigger the U-Net fallback (report 5.3.1): column count and row heights."""
    if grid.n_columns != cfg.n_columns:
        grid.ok, grid.reason = False, f"found {grid.n_columns} columns, expected {cfg.n_columns}"
        return grid
    if len(grid.y_lines) < 3:
        grid.ok, grid.reason = False, f"found {len(grid.y_lines)} horizontal rules, need at least 3"
        return grid
    # body rows, header excluded; the last strip above the printed table bottom can be any height
    heights = np.diff(grid.y_lines[1:-1]) if len(grid.y_lines) > 3 else np.array([])
    if len(heights) and heights.min() < cfg.line_height * 0.6:
        grid.ok, grid.reason = False, f"row of height {int(heights.min())} px is implausibly thin"
        return grid
    return grid
