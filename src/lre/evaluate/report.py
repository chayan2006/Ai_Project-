"""Run a system over a ground-truth folder and write metrics / tables (report 7, Appendix C)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import Config
from .metrics import FIELD_TYPES, evaluate_extraction


def load_gt_pages(data_dir: str | Path, limit: int | None = None) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(data_dir, "gt").glob("*.json"))[:limit]]


def update_metrics_file(path: Path, key: str, value: dict) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_per_field_table(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "accuracy"])
        for f in FIELD_TYPES:
            v = metrics["per_field_accuracy"].get(f)
            w.writerow([f, "" if v is None else f"{v:.4f}"])


def validate_predictions(preds: list[dict], data_dir: Path, cfg: Config, profile: str) -> None:
    """Link + rules on predicted holdings; sets ``contradiction``, ``flags``, ``link`` and ``route`` in place."""
    from ..io import load_register
    from ..schema import Holding, holding_from_dict, to_dict
    from ..route.router import decide_route
    from ..schema import Route
    from ..validate.engine import load_rules
    from ..validate.runner import validate_holdings
    rules = load_rules(cfg.paths.rules, profile)
    rows = load_register(data_dir / "register.csv")
    dicts = [d for page in preds for d in page["holdings"]]
    hs = [holding_from_dict({k: v for k, v in d.items() if k in Holding.__dataclass_fields__}) for d in dicts]
    validate_holdings(hs, rows, rules, cfg.linking)
    for d, h in zip(dicts, hs):
        route = decide_route(bool(d.get("needs_review")), h.flags, h.link, rules)
        d["contradiction"] = route == Route.CONTRADICTION
        d["needs_review"] = route != Route.AUTO
        d["route"] = route.value
        d["link"] = to_dict(h.link) if h.link else None
        d["flags"] = to_dict(h.flags)


def _score(preds: list[dict], data_dir: Path, cfg: Config, system: str, profile: str,
           limit: int | None) -> dict:
    """Validate, route and score predictions of any extractor; write metrics, table and predictions."""
    from .validation_eval import load_labels, truth_from_labels
    name = data_dir.name
    validate_predictions(preds, data_dir, cfg, profile=profile)
    pred_path = Path(cfg.paths.interim_dir) / system / f"{name}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, ensure_ascii=False))
    gt_pages = load_gt_pages(data_dir, limit)
    labels = load_labels(data_dir / "labels.csv")
    truth = truth_from_labels(labels, [h["holding_id"] for g in gt_pages for h in g["holdings"]])
    metrics, records = evaluate_extraction(gt_pages, {p["page_id"]: p for p in preds}, truth)
    metrics.update({"system": system, "rules_profile": profile, "data": str(data_dir), "pages": len(gt_pages),
                    "config_hash": cfg.hash(), "predictions": str(pred_path)})
    results = Path(cfg.paths.results_dir)
    update_metrics_file(results / f"{system}_metrics.json", name, metrics)
    write_per_field_table(results / "tables" / f"{system}_per_field_{name}.csv", metrics)
    (pred_path.parent / f"{name}_records.json").write_text(json.dumps(records))
    return metrics


def evaluate_baseline(data_dir: str | Path, cfg: Config, workers: int = 1, limit: int | None = None) -> dict:
    from ..baseline.template_tesseract import run_baseline
    data_dir = Path(data_dir)
    preds = run_baseline(data_dir, cfg, workers=workers, limit=limit)
    return _score(preds, data_dir, cfg, "baseline", "exact", limit)


def evaluate_proposed(data_dir: str | Path, cfg: Config, crnn_path: str | Path, unet_path: str | Path | None = None,
                      limit: int | None = None) -> dict:
    """Proposed system A (report Table 11): grid + CRNN + field-aware decoding + rules (default profile)."""
    import cv2
    from tqdm import tqdm

    from ..io import load_register
    from ..layout.unet import device, load_model
    from ..recognise.decode import build_lexicon
    from ..recognise.extractor import extract_page
    from ..recognise.train import load_checkpoint
    data_dir = Path(data_dir)
    dev = device()
    model = load_checkpoint(crnn_path).to(dev)
    unet = load_model(unet_path) if unet_path else None
    from ..extract.crf_owner import load_default
    crf = load_default(cfg.paths.models_dir)
    rows = load_register(data_dir / "register.csv")
    by_village: dict[str, list] = {}
    for r in rows:
        by_village.setdefault(r.village_code, []).append(r)
    lexicons = {v: build_lexicon(rs) for v, rs in by_village.items()}
    preds = []
    for gt in tqdm(load_gt_pages(data_dir, limit), desc="proposed", unit="page"):
        gray = cv2.imread(str(data_dir / gt["image"]), cv2.IMREAD_GRAYSCALE)
        hs = extract_page(gray, gt["page_id"], gt["village_code"], cfg, model, dev,
                          lexicons.get(gt["village_code"]), unet, owner_parser=crf)
        preds.append({"page_id": gt["page_id"], "village_code": gt["village_code"], "holdings": hs})
    m = _score(preds, data_dir, cfg, "proposed_a", "default", limit)
    m["crnn"] = str(crnn_path)
    m["unet"] = str(unet_path) if unet_path else None
    m["owner_parser"] = "crf" if crf is not None else "regex"
    return m
