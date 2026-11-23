"""SynJama: synthetic Jamabandi register pages, ground truth and a perturbed digital register.

Output layout for one split::

    <out>/pages/<page_id>.png        degraded page image
    <out>/gt/<page_id>.json          page ground truth (holdings, cells, text lines, rules, transform)
    <out>/register.csv               digital register for every village in the split
    <out>/labels.csv                 every injected change (contradiction / intra / benign / renumber)
    <out>/holdings_gt.jsonl          true holdings as written on the pages
    <out>/manifest.json              counts, seeds, config hash
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..io import save_register, write_json
from ..schema import Holding, to_dict
from .degrade import degrade, sample_transform
from .perturb import Label, apply_page_error, build_register
from .render import (Form, HandStyle, HoldingLayout, PageInk, check_complex_script_support,
                     layout_holding, load_font, make_form, paginate, render_page)
from .sample import page_texts, remarks_raw, sample_village

GENERATOR_VERSION = "synjama-0.1"
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "sample": 3}

INKS = [(28, 36, 92), (22, 26, 48), (34, 48, 120), (40, 40, 60), (18, 30, 80)]
REDS = [(150, 32, 30), (170, 40, 40), (130, 30, 45)]
PAPERS = [(244, 238, 222), (238, 230, 208), (246, 242, 230), (232, 222, 196), (240, 236, 226)]
RULES = [(70, 56, 40), (60, 50, 45), (85, 70, 50)]


@dataclass
class PageJob:
    split: str
    page_id: str
    village_code: str
    village_name: str
    hadbast: str
    register_year: str
    digit_style: str
    style: HandStyle
    holdings: list[Holding]
    layouts: list[HoldingLayout]
    seed: int


def _form(cfg: Config) -> Form:
    s = cfg.synth
    return make_form(int(s.page_width_in * s.dpi), int(s.page_height_in * s.dpi))


def plan_split(split: str, n_pages: int, cfg: Config, seed: int, village_start: int):
    """Sample villages until ``n_pages`` pages are laid out. Returns jobs, register rows, labels, villages."""
    s = cfg.synth
    rng = np.random.default_rng([seed, SPLIT_IDS.get(split, 9)])
    font_names = s.fonts.test if split == "test" else s.fonts.train
    fonts_dir = Path(cfg.paths.fonts_dir)
    for f in [*font_names, s.fonts.printed]:
        load_font(str(fonts_dir / f), 20)  # fail early with a clear message
    form = _form(cfg)

    jobs: list[PageJob] = []
    all_rows, all_labels, villages = [], [], []
    v = village_start
    while len(jobs) < n_pages:
        code = f"V{v:03d}"
        v += 1
        village = sample_village(rng, s, code)
        villages.append(village)
        style = HandStyle(font_path=str(fonts_dir / font_names[int(rng.integers(len(font_names)))]),
                          font_px=int(rng.integers(s.font_px[0], s.font_px[1] + 1)),
                          slant_deg=float(rng.uniform(*s.slant_deg)))

        page_versions: dict[str, Holding] = {}
        labels: list[Label] = []
        for h in village.holdings:
            ph = copy.deepcopy(h)
            if rng.random() < s.page_error_rate:
                lab = apply_page_error(ph, rng)
                if lab:
                    labels.append(lab)
            page_versions[h.khewat_no] = ph

        texts = [page_texts(page_versions[h.khewat_no], village, rng) for h in village.holdings]
        for t, h in zip(texts, village.holdings):
            page_versions[h.khewat_no].remarks.raw = remarks_raw(t)
        layouts = [layout_holding(t, form, style) for t in texts]
        pages = paginate(layouts, form)[: n_pages - len(jobs)]

        rows, reg_labels = build_register(village, page_versions, s, rng)
        labels.extend(reg_labels)

        id_of: dict[str, str] = {}
        for p_idx, idxs in enumerate(pages):
            page_id = f"{code}-P{p_idx + 1:04d}"
            hs = []
            for r, i in enumerate(idxs):
                ph = page_versions[village.holdings[i].khewat_no]
                ph.holding_id = f"{page_id}-R{r + 1:02d}"
                ph.page_id, ph.row_index = page_id, r
                id_of[ph.khewat_no] = ph.holding_id
                hs.append(ph)
            jobs.append(PageJob(split=split, page_id=page_id, village_code=code, village_name=village.name,
                                hadbast=village.hadbast, register_year=village.register_year,
                                digit_style=village.digit_style, style=style, holdings=hs,
                                layouts=[layouts[i] for i in idxs], seed=int(rng.integers(2**62))))
        for lab in labels:
            lab.holding_id = id_of.get(lab.khewat_no, "")
        all_rows.extend(rows)
        all_labels.extend(labels)
    return jobs, all_rows, all_labels, villages


def render_job(job: PageJob, cfg: Config, out_dir: str) -> dict:
    s = cfg.synth
    rng = np.random.default_rng(job.seed)
    form = _form(cfg)
    ink = PageInk(ink=INKS[int(rng.integers(len(INKS)))], red_ink=REDS[int(rng.integers(len(REDS)))],
                  paper=PAPERS[int(rng.integers(len(PAPERS)))], rule_color=RULES[int(rng.integers(len(RULES)))],
                  heavy_stroke=bool(rng.random() < 0.3), seed=int(rng.integers(2**62)))
    title = (f"जमाबंदी साल {job.register_year}     गाँव {job.village_name}     "
             f"हदबस्त नं. {job.hadbast}     (synthetic)")
    reds = [bool(h.remarks.mutation_no) for h in job.holdings]
    page, cells, row_spans, rules = render_page(job.layouts, reds, form, job.style, ink,
                                                str(Path(cfg.paths.fonts_dir) / s.fonts.printed), title)
    tf = sample_transform(rng, s, form.width, form.height)
    img, deg_info = degrade(np.asarray(page), ink.paper, tf, s, rng)

    out = Path(out_dir)
    ext = "jpg" if s.image_format == "jpg" else "png"
    img_path = out / "pages" / f"{job.page_id}.{ext}"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params)

    holdings = []
    for h, hc, (y0, y1) in zip(job.holdings, cells, row_spans):
        d = to_dict(h)
        d["row_bbox"] = tf.box_to_deskewed((form.x_edges[0], y0, form.x_edges[-1] - form.x_edges[0], y1 - y0))
        d["cells"] = [{"col": c.col, "field": c.field, "bbox": tf.box_to_deskewed(c.bbox),
                       "lines": [{"text": ln.text, "bbox": tf.box_to_deskewed(ln.bbox),
                                  "clipped": not tf.box_inside_scan(ln.bbox)} for ln in c.lines]}
                      for c in hc]
        holdings.append(d)
    pt = tf.point_to_deskewed
    gt = {
        "page_id": job.page_id, "split": job.split, "village_code": job.village_code,
        "village_name": job.village_name, "hadbast": job.hadbast, "register_year": job.register_year,
        "image": f"pages/{img_path.name}", "width": form.width, "height": form.height, "dpi": s.dpi,
        "frame": "deskewed",  # boxes below: scanned image rotated back by skew_deg about its centre
        "skew_deg": tf.skew_deg, "transform": {"scale": tf.scale, "tx": tf.tx, "ty": tf.ty,
                                                "matrix_form_to_scan": tf.full()[:2].round(6).tolist()},
        "font": Path(job.style.font_path).name, "font_px": job.style.font_px,
        "slant_deg": round(job.style.slant_deg, 2), "digit_style": job.digit_style,
        "ink": list(ink.ink), "paper": list(ink.paper), "degradation": deg_info, "seed": job.seed,
        "columns": [{"index": c, "x0": round(pt(form.x_edges[c], 0)[0]), "x1": round(pt(form.x_edges[c + 1], 0)[0])}
                    for c in range(12)],
        "header": {"y0": round(pt(0, form.table_top)[1]), "y1": round(pt(0, form.header_bottom)[1])},
        "table_bottom": round(pt(0, form.table_bottom)[1]),
        "h_lines": [{"y": round(pt(0, l["y"])[1]), "x0": round(pt(l["x0"], 0)[0]), "x1": round(pt(l["x1"], 0)[0]),
                     "w": l["w"]} for l in rules["h_lines"]],
        "v_lines": [{"x": round(pt(l["x"], 0)[0]), "y0": round(pt(0, l["y0"])[1]), "y1": round(pt(0, l["y1"])[1]),
                     "w": l["w"]} for l in rules["v_lines"]],
        "holdings": holdings,
    }
    write_json(out / "gt" / f"{job.page_id}.json", gt)
    return {"page_id": job.page_id, "holdings": len(job.holdings),
            "lines": sum(len(c["lines"]) for h in holdings for c in h["cells"])}


def generate_split(out_dir: str | Path, split: str, n_pages: int, cfg: Config, seed: int = 0,
                   village_start: int = 0, workers: int = 1) -> dict:
    check_complex_script_support()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs, rows, labels, villages = plan_split(split, n_pages, cfg, seed, village_start)

    results = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(render_job, j, cfg, str(out)) for j in jobs]
            for f in tqdm(futs, desc=f"render {split}", unit="page"):
                results.append(f.result())
    else:
        for j in tqdm(jobs, desc=f"render {split}", unit="page"):
            results.append(render_job(j, cfg, str(out)))

    save_register(rows, out / "register.csv")
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        fields = list(Label.__dataclass_fields__)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for lab in labels:
            w.writerow(lab.as_row())
    with open(out / "holdings_gt.jsonl", "w", encoding="utf-8") as fh:
        for j in jobs:
            for h in j.holdings:
                fh.write(json.dumps(to_dict(h), ensure_ascii=False) + "\n")

    on_page = [lab for lab in labels if lab.holding_id]
    manifest = {
        "generator": GENERATOR_VERSION, "split": split, "seed": seed, "config_hash": cfg.hash(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "pages": len(jobs), "villages": [v.code for v in villages],
        "holdings_on_pages": sum(r["holdings"] for r in results),
        "text_lines": sum(r["lines"] for r in results),
        "register_rows": len(rows),
        "fonts": sorted({Path(j.style.font_path).name for j in jobs}),
        "labels_on_pages": dict(Counter(f"{lab.kind}:{lab.type}" for lab in on_page)),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def generate_all(out_root: str | Path, n_pages: int, cfg: Config, seed: int = 0, workers: int = 1) -> dict:
    """Train / val / test splits by village, with held-out fonts and seeds for test (report Table 5)."""
    fr = cfg.synth.split_fractions
    counts = {k: int(round(n_pages * fr[k])) for k in ("train", "val", "test")}
    counts["train"] = n_pages - counts["val"] - counts["test"]
    out, start = {}, 0
    for split in ("train", "val", "test"):
        if counts[split] <= 0:
            continue
        m = generate_split(Path(out_root) / split, split, counts[split], cfg, seed, start, workers)
        start += len(m["villages"])
        out[split] = m
    return out
