"""`lre run`: ingest -> preprocess -> extract -> link -> validate -> route -> store (report 5.1).

Each page writes stage checkpoints to data/interim/<page_id>/, a downscaled deskewed
image for the reviewer, and one crop per extracted cell (the reviewer's zoom, and the
image half of any correction that later becomes training data).

The extractor is pluggable. Until the CRNN is trained (Phase 5) the only one is the
template + Tesseract baseline, which is validated with the rules' `exact` profile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

from . import store
from .config import Config
from .extract.holding import EXTRACTED_COLUMNS, FIELD_OF_COLUMN
from .extract.normalise import marla_to_kanal_marla
from .io import load_register, sha256_file
from .link.linker import build_index, link
from .logging_utils import get_logger
from .pipeline import save_checkpoint
from .preprocess.pipeline import preprocess
from .preprocess.steps import IMAGE_EXTS, load_pages
from .route.router import decide_route
from .schema import Holding, RegisterRow, holding_from_dict, to_dict
from .validate.engine import load_rules, validate

DISPLAY_WIDTH = 1600
EXTRACTORS = {"baseline": ("template-tesseract-5", "exact"), "crnn": ("crnn", "default")}


def _owners_text(owners) -> str:
    return "; ".join(" ".join(x for x in (o.name, o.relation or "", o.parent or "", o.share or "?") if x)
                     for o in owners)


def field_values(h: Holding) -> dict[int, str]:
    return {
        0: h.khewat_no, 1: h.khatauni_no, 3: _owners_text(h.owners), 4: h.cultivator,
        6: ", ".join(p.khasra for p in h.plots if p.khasra),
        7: ", ".join(f"{marla_to_kanal_marla(p.area_marla) if p.area_marla is not None else '?'} {p.land_type or '?'}"
                     for p in h.plots) + (f" (कुल {marla_to_kanal_marla(h.total_area_marla)})"
                                          if h.total_area_marla is not None else ""),
        11: " ".join(x for x in (h.remarks.mutation_no, h.remarks.date) if x),
    }


def register_values(row: RegisterRow | None) -> dict[int, str]:
    if row is None:
        return {}
    return {
        0: row.khewat_no, 1: row.khatauni_no, 3: _owners_text(row.owners), 4: row.cultivator,
        6: ", ".join(p.khasra for p in row.plots),
        7: ", ".join(f"{marla_to_kanal_marla(p.area_marla) if p.area_marla is not None else '?'} {p.land_type or '?'}"
                     for p in row.plots),
        11: "history: " + ", ".join(row.mutation_history[-4:]) if row.mutation_history else "",
    }


def page_files(pages: Path) -> list[Path]:
    if pages.is_file():
        return [pages]
    return sorted(p for p in pages.iterdir() if p.suffix.lower() in IMAGE_EXTS | {".pdf"})


def _extractor(name: str, village: str, cfg: Config, rows: list[RegisterRow], crnn_path: Path):
    """Returns extract(gray, page_id, pre) -> list of holding dicts."""
    if name == "baseline":
        from .baseline.template_tesseract import extract_page, load_template, tessdata_dir
        tpl, tess = load_template(cfg), tessdata_dir(cfg)
        return lambda gray, page_id, pre: extract_page(gray, page_id, village, cfg, tpl, tess, pre=pre)
    from .layout.unet import device, load_model
    from .recognise.decode import build_lexicon
    from .recognise.extractor import extract_page as crnn_extract
    from .recognise.train import load_checkpoint
    if not Path(crnn_path).exists():
        raise FileNotFoundError(f"no CRNN checkpoint at {crnn_path}: run `lre train-crnn` first")
    dev = device()
    model = load_checkpoint(crnn_path).to(dev)
    unet = load_model(Path(cfg.paths.models_dir) / "unet_v1.pt")
    lexicon = build_lexicon(rows)
    from .extract.crf_owner import load_default
    crf = load_default(cfg.paths.models_dir)
    return lambda gray, page_id, pre: crnn_extract(gray, page_id, village, cfg, model, dev, lexicon, unet, pre=pre,
                                                   owner_parser=crf)


def run_pages(pages: Path, village: str, register_csv: Path, cfg: Config, db_path: Path,
              extractor: str = "baseline", log: Callable[[str], None] | None = None,
              crnn_path: Path = Path("models/crnn_B.pt")) -> dict:
    log = log or get_logger().info
    if extractor not in EXTRACTORS:
        raise ValueError(f"unknown extractor '{extractor}'; available: {sorted(EXTRACTORS)}")
    model_version, profile = EXTRACTORS[extractor]
    if extractor == "crnn":
        model_version = f"crnn:{Path(crnn_path).name}"

    rows = load_register(register_csv, village)
    if not rows:
        raise ValueError(f"no register rows for village '{village}' in {register_csv}")
    extract = _extractor(extractor, village, cfg, rows, crnn_path)
    rules = load_rules(cfg.paths.rules, profile)
    idx = build_index(rows)
    khasra_owner = idx.khasra_owner
    conn = store.connect(db_path)
    store.add_register_rows(conn, rows)
    interim = Path(cfg.paths.interim_dir)
    summary = {"pages": 0, "duplicates": 0, "holdings": 0, "AUTO": 0, "REVIEW": 0, "CONTRADICTION": 0}

    for path in page_files(Path(pages)):
        file_sha = sha256_file(path)
        for page_no, gray in enumerate(load_pages(path)):
            sha = file_sha if page_no == 0 else f"{file_sha}:{page_no}"
            if (dup := store.page_exists(conn, sha)) is not None:
                log(f"skip {path.name}: already ingested as {dup}")
                summary["duplicates"] += 1
                continue
            stem = path.stem if path.stem.startswith(village) else f"{village}-{path.stem}"
            page_id = stem + (f"-{page_no + 1}" if page_no else "")
            pdir = interim / page_id
            pre = preprocess(gray, cfg.preprocess)
            scale = DISPLAY_WIDTH / pre.gray.shape[1]
            display = pdir / "display.jpg"
            pdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(display), cv2.resize(pre.gray, (DISPLAY_WIDTH, int(pre.gray.shape[0] * scale)),
                                                 interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
            save_checkpoint(interim, page_id, "preprocess", {"source": str(path), "sha256": sha,
                                                            "skew_deg": pre.skew_deg})
            store.add_page(conn, page_id, village, path, sha, cfg.synth.dpi, pre.skew_deg, display, scale,
                           status="processing")
            extracted = extract(gray, page_id, pre)
            save_checkpoint(interim, page_id, "extract", extracted)
            (pdir / "cells").mkdir(exist_ok=True)
            routed = []
            for d in extracted:
                h = holding_from_dict({k: v for k, v in d.items() if k in Holding.__dataclass_fields__})
                h.link = link(h, idx, cfg.linking)
                row = idx.rows.get(h.link.register_row_id) if h.link.register_row_id else None
                h.flags = validate(h, row, rules, khasra_owner)
                h.route = decide_route(bool(d.get("needs_review")), h.flags, h.link, rules)
                h.models = {"extractor": model_version, "rules_profile": profile, "config": cfg.hash()}
                values, reg = field_values(h), register_values(row)
                fields = []
                for c in EXTRACTED_COLUMNS:
                    box = d["cell_boxes"][str(c)]
                    x, y, w, hh = box
                    crop_path = pdir / "cells" / f"{h.holding_id}_{c:02d}.png"
                    cv2.imwrite(str(crop_path), pre.gray[y:y + hh, x:x + w])
                    fields.append({"field_name": FIELD_OF_COLUMN[c], "raw_text": d["raw_texts"].get(str(c), ""),
                                   "normalised_value": values.get(c, ""), "register_value": reg.get(c, ""),
                                   "confidence_raw": d["column_conf"].get(str(c)), "bbox": box,
                                   "crop_path": str(crop_path), "model_version": model_version})
                store.add_holding(conn, {
                    "holding_id": h.holding_id, "page_id": page_id, "row_index": h.row_index,
                    "village_code": village, "khewat_no": h.khewat_no, "khatauni_no": h.khatauni_no,
                    "route": h.route.value, "link": to_dict(h.link), "row_span": d["row_span"],
                    "holding_json": to_dict(h), "models": h.models}, fields, to_dict(h.flags))
                summary[h.route.value] += 1
                routed.append(to_dict(h))
            store.set_page_status(conn, page_id, "processed")
            conn.commit()
            save_checkpoint(interim, page_id, "route", routed)
            summary["pages"] += 1
            summary["holdings"] += len(extracted)
            log(f"{page_id}: {len(extracted)} holdings")
    conn.close()
    return summary
