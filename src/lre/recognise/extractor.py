"""Proposed extractor: grid cells -> text lines -> CRNN -> field-aware decoding -> holdings.

Output dicts have the same shape as the template baseline's (row_span, raw_texts, cell_boxes,
column_conf, needs_review), so `lre run`, the reviewer and the metrics treat both alike.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..config import Config
from ..extract.holding import EXTRACTED_COLUMNS, FIELD_OF_COLUMN, REQUIRED_COLUMNS, build_holding
from ..layout.cells import find_cells, split_lines
from ..preprocess.pipeline import preprocess
from ..schema import to_dict
from .crnn import CRNN
from .data import collate, line_to_input
from .decode import decode_line

# Until per-field calibration exists (report 5.3.8), a field whose raw confidence is below this goes to review.
UNCALIBRATED_THRESHOLD = 0.90


def _line_crops(gray: np.ndarray, binary: np.ndarray, bbox, pad: int = 6) -> list[np.ndarray]:
    """Text lines of one cell, each trimmed horizontally to its ink."""
    x, y, w, h = bbox
    cg, cb = gray[y:y + h, x:x + w], binary[y:y + h, x:x + w]
    out = []
    for y0, y1 in split_lines(cg, cb):
        cols = np.nonzero((cb[y0:y1] > 0).sum(axis=0) > 1)[0]
        if len(cols) == 0:
            continue
        a, b = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
        out.append(line_to_input(cg[max(0, y0 - 3):min(h, y1 + 3), a:b]))
    return out


@torch.no_grad()
def recognise_lines(model: CRNN, images: list[np.ndarray], dev, batch_size: int = 32) -> list[np.ndarray]:
    """Log-prob matrices (T x C, numpy) per line, batched by similar width."""
    order = np.argsort([im.shape[1] for im in images])
    out: list[Optional[np.ndarray]] = [None] * len(images)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        b = collate([images[j] for j in idx], [""] * len(idx), [""] * len(idx))
        lp = model(b.images.to(dev)).float().cpu().numpy()          # T x B x C
        for k, j in enumerate(idx):
            out[j] = lp[: int(b.input_lengths[k]), k, :]
    return out  # type: ignore[return-value]


def extract_page(gray: np.ndarray, page_id: str, village_code: str, cfg: Config, model: CRNN, dev,
                 lexicon: Optional[set[str]] = None, unet=None, pre=None,
                 threshold: float = UNCALIBRATED_THRESHOLD, owner_parser=None) -> list[dict]:
    pre = pre if pre is not None else preprocess(gray, cfg.preprocess)
    grid, cells = find_cells(pre, cfg.layout, page_id, unet=unet)
    if not grid.ok:
        return []                                   # report E.11: page routed to manual entry
    rows: dict[int, dict[int, object]] = {}
    for c in cells:
        rows.setdefault(c.row, {})[c.col] = c
    jobs = []                                       # (row, col, line image)
    for r, cols in rows.items():
        for col in EXTRACTED_COLUMNS:
            for img in _line_crops(pre.gray, pre.binary, cols[col].bbox):
                jobs.append((r, col, img))
    lps = recognise_lines(model, [j[2] for j in jobs], dev) if jobs else []
    decoded: dict[tuple[int, int], list] = {}
    for (r, col, _), lp in zip(jobs, lps):
        decoded.setdefault((r, col), []).append(decode_line(lp, FIELD_OF_COLUMN[col], lexicon,
                                                            cfg.decoding.beam_width, cfg.decoding.lexicon_weight))
    out = []
    for r in sorted(rows):
        texts, raws, confs, boxes, bad_grammar = {}, {}, {}, {}, []
        for col in EXTRACTED_COLUMNS:
            lines = decoded.get((r, col), [])
            texts[col] = "\n".join(d.text for d in lines)
            raws[str(col)] = "\n".join(d.raw_text for d in lines)
            confs[col] = min((d.conf_raw for d in lines), default=None)
            boxes[str(col)] = list(rows[r][col].bbox)
            if any(not d.grammar_ok for d in lines):
                bad_grammar.append(col)
        kw = {"owner_parser": owner_parser} if owner_parser is not None else {}
        h = build_holding(texts, f"{page_id}-R{len(out) + 1:02d}", village_code, page_id, len(out), confs, **kw)
        missing = [c for c in REQUIRED_COLUMNS if not texts.get(c)]
        low = [c for c, v in confs.items() if v is not None and v < threshold]
        any_cell = next(iter(rows[r].values()))
        y0 = any_cell.bbox[1] - cfg.layout.cell_padding
        y1 = any_cell.bbox[1] + any_cell.bbox[3] + cfg.layout.cell_padding
        d = to_dict(h)
        d.update({"row_span": [int(y0), int(y1)], "raw_texts": raws, "cell_boxes": boxes,
                  "column_conf": {str(c): confs[c] for c in confs},
                  "needs_review": bool(missing or low or bad_grammar), "low_conf_columns": low,
                  "missing_columns": missing, "grammar_failed_columns": bad_grammar,
                  "texts": {str(c): texts[c] for c in texts}, "grid_method": grid.method})
        out.append(d)
    return out
