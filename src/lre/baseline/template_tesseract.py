"""Template-based rule extraction: the baseline named in Objective 3 (report 5.2).

(i)   fixed column boundaries measured once on a blank form;
(ii)  rows found by horizontal projection;
(iii) Tesseract 5 (LSTM, hin+eng, --psm 6) on each cell;
(iv)  regular expressions to parse shares, areas and khasra numbers;
(v)   the proposed system's rule catalogue with exact equality (validate/engine.py, profile "exact");
(vi)  review whenever a Tesseract word confidence is below 60.

It shares the preprocessing with the proposed system, so any gain can be
attributed to the learned components.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..extract.holding import EXTRACTED_COLUMNS, REQUIRED_COLUMNS, build_holding
from ..preprocess.pipeline import preprocess
from ..schema import to_dict

REVIEW_CONF = 60.0
TESS_CONFIG = "--oem 1 --psm 6"


@dataclass
class Template:
    x_lines: list[int]
    table_top: int
    header_bottom: int
    table_bottom: int
    width: int
    height: int


def template_path(cfg: Config) -> Path:
    return Path(cfg.paths.models_dir) / "baseline_template.json"


def measure_template(cfg: Config) -> Template:
    """Render the printed form with no handwriting and measure its rules once."""
    from PIL import Image

    from ..layout.grid import detect_grid
    from ..synth.render import PageInk, draw_form, make_form
    s = cfg.synth
    form = make_form(int(s.page_width_in * s.dpi), int(s.page_height_in * s.dpi))
    ink = PageInk(ink=(30, 30, 60), red_ink=(150, 30, 30), paper=(244, 238, 222), rule_color=(70, 56, 40),
                  heavy_stroke=False, seed=0)
    page = Image.new("RGB", (form.width, form.height), ink.paper)
    draw_form(page, form, [], str(Path(cfg.paths.fonts_dir) / s.fonts.printed), "", ink, np.random.default_rng(0))
    pre = preprocess(np.asarray(page.convert("L")), cfg.preprocess, denoise=False)
    grid = detect_grid(pre.binary, cfg.layout, pre.gray)
    if grid.n_columns != cfg.layout.n_columns or len(grid.y_lines) != 3:
        raise RuntimeError(f"blank form measurement failed: {grid}")
    return Template(x_lines=grid.x_lines, table_top=grid.y_lines[0], header_bottom=grid.y_lines[1],
                    table_bottom=grid.y_lines[2], width=form.width, height=form.height)


def load_template(cfg: Config) -> Template:
    p = template_path(cfg)
    if p.exists():
        return Template(**json.loads(p.read_text()))
    tpl = measure_template(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tpl.__dict__, indent=2))
    return tpl


def find_rows(binary: np.ndarray, tpl: Template, window: int = 7, min_thickness: float = 2.4,
              min_height: int = 40) -> list[tuple[int, int]]:
    """Rows between ruling lines found on the horizontal ink projection inside the template's table."""
    x0, x1 = tpl.x_lines[0], tpl.x_lines[-1]
    prof = (binary[:, x0:x1] > 0).sum(axis=1) / max(1, x1 - x0)
    smooth = np.convolve(prof, np.ones(window), mode="same")     # tolerant to a little residual skew
    is_rule = smooth >= min_thickness
    ys, start = [], None
    for y, v in enumerate(is_rule):
        if v and start is None:
            start = y
        elif not v and start is not None:
            ys.append((start + y) // 2)
            start = None
    ys = [y for y in ys if y >= tpl.header_bottom - 60]
    return [(a, b) for a, b in zip(ys, ys[1:]) if b - a >= min_height]


def ocr_cell(crop: np.ndarray, tessdata: str) -> tuple[str, list[float]]:
    """Tesseract on one cell: text (one line per recognised line) and word confidences (0-100)."""
    import pytesseract
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "", []
    d = pytesseract.image_to_data(crop, lang="hin+eng", config=f"--tessdata-dir {tessdata} {TESS_CONFIG}",
                                  output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list[str]] = {}
    confs = []
    for i, word in enumerate(d["text"]):
        if not word.strip():
            continue
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(word)
        confs.append(float(d["conf"][i]))
    return "\n".join(" ".join(ws) for _, ws in sorted(lines.items())), confs


def extract_page(gray: np.ndarray, page_id: str, village_code: str, cfg: Config, tpl: Template,
                 tessdata: str, pad: int = 6, pre=None) -> list[dict]:
    pre = pre if pre is not None else preprocess(gray, cfg.preprocess)
    has_ink = lambda a: a.size and (a > 0).mean() > 0.002  # noqa: E731
    out = []
    for r, (y0, y1) in enumerate(find_rows(pre.binary, tpl)):
        if not has_ink(pre.binary[y0 + pad:y1 - pad, tpl.x_lines[0]:tpl.x_lines[-1]]):
            continue
        texts, confs, min_word, boxes = {}, {}, {}, {}
        for c in EXTRACTED_COLUMNS:
            xa, xb = tpl.x_lines[c] + pad, tpl.x_lines[c + 1] - pad
            text, wc = ocr_cell(pre.gray[y0 + pad:y1 - pad, xa:xb], tessdata)
            boxes[str(c)] = [int(xa), int(y0 + pad), int(xb - xa), int(y1 - y0 - 2 * pad)]
            texts[c] = text
            confs[c] = (min(wc) / 100.0) if wc else None
            min_word[c] = min(wc) if wc else None
        h = build_holding(texts, f"{page_id}-B{len(out) + 1:02d}", village_code, page_id, len(out), confs)
        missing = [c for c in REQUIRED_COLUMNS if not texts.get(c)]
        low = [c for c, m in min_word.items() if m is not None and m < REVIEW_CONF]
        d = to_dict(h)
        d.update({"row_span": [int(y0), int(y1)], "raw_texts": {str(c): texts[c] for c in texts},
                  "cell_boxes": boxes, "column_conf": {str(c): confs[c] for c in confs},
                  "needs_review": bool(missing or low), "low_conf_columns": low, "missing_columns": missing})
        out.append(d)
    return out


def _page_job(gt_path: str, data_dir: str, cfg: Config, tpl: Template, tessdata: str) -> dict:
    import cv2
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    gray = cv2.imread(str(Path(data_dir) / gt["image"]), cv2.IMREAD_GRAYSCALE)
    return {"page_id": gt["page_id"], "village_code": gt["village_code"],
            "holdings": extract_page(gray, gt["page_id"], gt["village_code"], cfg, tpl, tessdata)}


def tessdata_dir(cfg: Config) -> str:
    d = Path(cfg.paths.models_dir) / "tessdata"
    if not (d / "hin.traineddata").exists():
        raise FileNotFoundError(f"{d}/hin.traineddata missing: run `lre tessdata` first")
    return str(d.resolve())


def run_baseline(data_dir: str | Path, cfg: Config, workers: int = 1, limit: Optional[int] = None) -> list[dict]:
    from tqdm import tqdm
    tpl = load_template(cfg)
    tess = tessdata_dir(cfg)
    gts = [str(p) for p in sorted(Path(data_dir, "gt").glob("*.json"))[:limit]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_page_job, g, str(data_dir), cfg, tpl, tess) for g in gts]
            return [f.result() for f in tqdm(futs, desc="baseline", unit="page")]
    return [_page_job(g, str(data_dir), cfg, tpl, tess) for g in tqdm(gts, desc="baseline", unit="page")]
