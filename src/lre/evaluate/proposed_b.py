"""Proposed system B (report Table 11): A + GBM contradiction scorer + calibrated routing.

* the scorer is trained on flags raised on the CRNN system's own output for SynJama *train* pages;
* theta (F0.5) and the per-field isotonic calibrators / thresholds are fitted on *val*;
* both are applied once to the evaluated split.
Inputs are the prediction files written by `lre evaluate --stage proposed` (data/interim/proposed_a/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..schema import Holding, holding_from_dict
from ..route.calibrate import COLUMNS, Calibrators, column_correct
from ..validate.scorer import ContradictionScorer, flag_features
from .metrics import evaluate_extraction, match_rows
from .report import load_gt_pages, update_metrics_file, write_per_field_table
from .validation_eval import POSITIVE_KINDS, load_labels, truth_from_labels

SEVERE = ("high", "medium")


def _pairs(data_dir: Path, preds: list[dict]):
    """(pred holding dict, matched GT holding dict or None) for every predicted holding."""
    gts = {g["page_id"]: g for g in load_gt_pages(data_dir)}
    for page in preds:
        g = gts.get(page["page_id"])
        if g is None:
            continue
        spans = [(h["row_bbox"][1], h["row_bbox"][1] + h["row_bbox"][3]) for h in g["holdings"]]
        matched = match_rows(spans, [tuple(p["row_span"]) for p in page["holdings"]])
        back = {pi: gi for gi, pi in enumerate(matched) if pi is not None}
        for pi, p in enumerate(page["holdings"]):
            yield p, (g["holdings"][back[pi]] if pi in back else None)


def _holding(d: dict) -> Holding:
    return holding_from_dict({k: v for k, v in d.items() if k in Holding.__dataclass_fields__ and k != "flags"})


def flag_training_set(data_dir: Path, preds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = load_labels(data_dir / "labels.csv")
    X, y = [], []
    for p, g in _pairs(data_dir, preds):
        labs = labels.get(g["holding_id"], []) if g else []
        expected = {r for lab in labs if lab["kind"] in POSITIVE_KINDS for r in lab["rules"].split(";") if r}
        for fl in p.get("flags", []):
            if fl["severity"] in SEVERE:
                X.append(flag_features(fl, p))
                y.append(fl["rule"] in expected)
    return np.array(X), np.array(y, bool)


def fit_calibrators(data_dir: Path, preds: list[dict], alpha: float) -> tuple[Calibrators, dict]:
    samples: dict[str, tuple[list, list]] = {c: ([], []) for c in COLUMNS}
    for p, g in _pairs(data_dir, preds):
        if g is None:
            continue
        gh = holding_from_dict({k: v for k, v in g.items() if k not in ("row_bbox", "cells")})
        ph = _holding(p)
        for col in COLUMNS:
            conf = (p.get("column_conf") or {}).get(col)
            if conf is not None:
                samples[col][0].append(conf)
                samples[col][1].append(column_correct(col, gh, ph))
    cal = Calibrators()
    return cal, cal.fit(samples, alpha)


def apply_b(preds: list[dict], scorer: ContradictionScorer, cal: Calibrators) -> None:
    """Re-route predictions in place: scored contradictions, calibrated field thresholds."""
    for page in preds:
        for p in page["holdings"]:
            s = scorer.score_holding(p, SEVERE)
            contra = s >= scorer.theta
            unlinked = not p.get("link") or p["link"].get("method") == "unlinked"
            confs = p.get("column_conf") or {}
            fields_ok = all(cal.passes(col, confs.get(col)) for col in COLUMNS)
            for col, v in confs.items():
                p.setdefault("column_conf_cal", {})[col] = cal.calibrate(col, v)
            p["contradiction"] = bool(contra)
            p["needs_review"] = bool(contra or unlinked or not fields_ok or (p.get("flags") and not contra))
            p["route"] = "CONTRADICTION" if contra else ("REVIEW" if p["needs_review"] else "AUTO")


def evaluate_proposed_b(cfg: Config, train_dir: Path, val_dir: Path, eval_dir: Path) -> dict:
    interim = Path(cfg.paths.interim_dir) / "proposed_a"
    load = lambda d: json.loads((interim / f"{Path(d).name}.json").read_text(encoding="utf-8"))  # noqa: E731
    X, y = flag_training_set(train_dir, load(train_dir))
    scorer = ContradictionScorer().fit(X, y)
    val_preds = load(val_dir)
    labels = load_labels(val_dir / "labels.csv")
    scores, truth = [], []
    for p, g in _pairs(val_dir, val_preds):
        scores.append(scorer.score_holding(p, SEVERE))
        truth.append(bool(g) and any(lab["kind"] in POSITIVE_KINDS for lab in labels.get(g["holding_id"], [])))
    theta = scorer.tune_theta(scores, truth)
    cal, cal_report = fit_calibrators(val_dir, val_preds, cfg.routing.alpha)
    models = Path(cfg.paths.models_dir)
    scorer.save(models / "gbm_v1.joblib")
    cal.save(models / "calibrators_v1.joblib")

    preds = load(eval_dir)
    apply_b(preds, scorer, cal)
    gt_pages = load_gt_pages(eval_dir)
    ev_labels = load_labels(eval_dir / "labels.csv")
    t = truth_from_labels(ev_labels, [h["holding_id"] for g in gt_pages for h in g["holdings"]])
    metrics, _ = evaluate_extraction(gt_pages, {p["page_id"]: p for p in preds}, t)
    metrics.update({"system": "proposed_b", "scorer_train_flags": int(len(y)), "scorer_train_positive": int(y.sum()),
                    "theta": theta, "alpha": cfg.routing.alpha, "calibration": cal_report,
                    "data": str(eval_dir), "fitted_on": {"scorer": str(train_dir), "theta_and_calibration": str(val_dir)}})
    results = Path(cfg.paths.results_dir)
    update_metrics_file(results / "proposed_b_metrics.json", eval_dir.name, metrics)
    write_per_field_table(results / "tables" / f"proposed_b_per_field_{eval_dir.name}.csv", metrics)
    return metrics
