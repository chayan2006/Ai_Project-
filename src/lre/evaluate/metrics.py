"""Extraction metrics against SynJama / HandFilled ground truth (report Table 10, Table 14).

Predicted rows are matched to ground-truth holdings by vertical overlap on the
deskewed page. A ground-truth holding with no matching row counts as wrong on
every field and as needing review (it would have to be entered by hand).
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Optional

import numpy as np
from rapidfuzz.distance import Levenshtein

from ..extract import normalise as N
from ..schema import Holding, holding_from_dict

FIELD_TYPES = ["khewat_no", "khatauni_no", "owner_name", "parent_name", "share", "cultivator",
               "khasra", "area", "land_type", "mutation_ref"]
NUMERIC_FIELDS = ["khewat_no", "khatauni_no", "share", "khasra", "area"]


def _y_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_rows(gt_spans: list[tuple[int, int]], pred_spans: list[tuple[int, int]],
               min_iou: float = 0.5) -> list[Optional[int]]:
    """For each ground-truth row, the index of the best-overlapping predicted row (each used once)."""
    pairs = sorted(((_y_iou(g, p), gi, pi) for gi, g in enumerate(gt_spans) for pi, p in enumerate(pred_spans)),
                   reverse=True)
    out: list[Optional[int]] = [None] * len(gt_spans)
    used = set()
    for v, gi, pi in pairs:
        if v < min_iou:
            break
        if out[gi] is None and pi not in used:
            out[gi] = pi
            used.add(pi)
    return out


def _share(s: Optional[str]) -> Optional[Fraction]:
    try:
        return Fraction(s) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def compare_holding(gt: Holding, pred: Optional[Holding]) -> tuple[dict[str, list[bool]], dict[str, list[int]]]:
    """Per-field correctness lists and CER parts {'name': [dist, ref_len], 'remarks': [...]}."""
    ok: dict[str, list[bool]] = defaultdict(list)
    cer = {"name": [0, 0], "remarks": [0, 0]}
    p = pred
    ok["khewat_no"].append(p is not None and N.normalise_khewat(p.khewat_no) == N.normalise_khewat(gt.khewat_no))
    ok["khatauni_no"].append(p is not None and N.parse_khatauni(p.khatauni_no) == N.parse_khatauni(gt.khatauni_no))
    for i, go in enumerate(gt.owners):
        po = p.owners[i] if p is not None and i < len(p.owners) else None
        ok["owner_name"].append(po is not None and N.normalise_name(po.name) == N.normalise_name(go.name))
        ok["parent_name"].append(po is not None and N.normalise_name(po.parent or "") == N.normalise_name(go.parent or ""))
        ok["share"].append(po is not None and _share(po.share) == _share(go.share))
        for ref, hyp in ((go.name, po.name if po else ""), (go.parent or "", (po.parent or "") if po else "")):
            ref, hyp = N.normalise_name(ref), N.normalise_name(hyp)
            cer["name"][0] += Levenshtein.distance(ref, hyp)
            cer["name"][1] += len(ref)
    ok["cultivator"].append(p is not None and N.normalise_cultivator(p.cultivator) == N.normalise_cultivator(gt.cultivator))
    for i, gp in enumerate(gt.plots):
        pp = p.plots[i] if p is not None and i < len(p.plots) else None
        ok["khasra"].append(pp is not None and N.parse_khasra(pp.khasra or "") == N.parse_khasra(gp.khasra))
        ok["area"].append(pp is not None and pp.area_marla == gp.area_marla)
        ok["land_type"].append(pp is not None and pp.land_type == gp.land_type)
    ok["mutation_ref"].append(p is not None and (p.remarks.mutation_no or None) == (gt.remarks.mutation_no or None))
    ref = N.clean(gt.remarks.raw)
    if ref:
        cer["remarks"][0] += Levenshtein.distance(ref, N.clean(p.remarks.raw) if p else "")
        cer["remarks"][1] += len(ref)
    return ok, cer


def evaluate_extraction(gt_pages: list[dict], pred_pages: dict[str, dict],
                        truth: Optional[dict[str, bool]] = None) -> tuple[dict, list[dict]]:
    """Returns (metrics, per-holding records with the matched prediction and its review flag).

    Predictions may carry ``contradiction`` (bool); with ``truth`` (ground-truth holding id ->
    holding truly contains a contradiction) holding-level contradiction precision / recall are added.
    Flagged predicted rows that match no ground-truth holding count as false positives.
    """
    field_ok: dict[str, list[bool]] = defaultdict(list)
    auto_ok: dict[str, list[bool]] = defaultdict(list)
    cer_tot = {"name": [0, 0], "remarks": [0, 0]}
    records, n_review, n_missed, n_pred_rows = [], 0, 0, 0
    spurious_flagged = 0
    for gp in gt_pages:
        pred = pred_pages.get(gp["page_id"], {"holdings": []})
        preds = pred["holdings"]
        n_pred_rows += len(preds)
        gt_spans = [(h["row_bbox"][1], h["row_bbox"][1] + h["row_bbox"][3]) for h in gp["holdings"]]
        matched = match_rows(gt_spans, [tuple(p["row_span"]) for p in preds])
        used = {pi for pi in matched if pi is not None}
        spurious_flagged += sum(1 for i, p in enumerate(preds) if i not in used and p.get("contradiction"))
        for gh, pi in zip(gp["holdings"], matched):
            g = holding_from_dict({k: v for k, v in gh.items() if k not in ("row_bbox", "cells")})
            pd_ = preds[pi] if pi is not None else None
            p = holding_from_dict({k: v for k, v in pd_.items() if k in Holding.__dataclass_fields__}) if pd_ else None
            ok, cer = compare_holding(g, p)
            review = pd_ is None or bool(pd_.get("needs_review"))
            n_review += review
            n_missed += pd_ is None
            for f, v in ok.items():
                field_ok[f].extend(v)
                if not review:
                    auto_ok[f].extend(v)
            for k in cer_tot:
                cer_tot[k][0] += cer[k][0]
                cer_tot[k][1] += cer[k][1]
            records.append({"gt_holding_id": g.holding_id, "pred_holding_id": p.holding_id if p else None,
                            "needs_review": review, "all_fields_ok": all(all(v) for v in ok.values()),
                            "pred_contradiction": bool(pd_ and pd_.get("contradiction"))})
    per_field = {f: float(np.mean(field_ok[f])) if field_ok[f] else None for f in FIELD_TYPES}
    auto_vals = [v for f in FIELD_TYPES for v in auto_ok[f]]
    n_gt = len(records)
    metrics = {
        "holdings": n_gt,
        "predicted_rows": n_pred_rows,
        "missed_holdings": n_missed,
        "per_field_accuracy": per_field,
        "mean_field_accuracy": float(np.mean([v for v in per_field.values() if v is not None])),
        "numeric_field_accuracy": float(np.mean([per_field[f] for f in NUMERIC_FIELDS if per_field[f] is not None])),
        "name_cer": cer_tot["name"][0] / max(1, cer_tot["name"][1]),
        "remarks_cer": cer_tot["remarks"][0] / max(1, cer_tot["remarks"][1]),
        "review_share": n_review / max(1, n_gt),
        "auto_accepted_holdings": n_gt - n_review,
        "auto_accept_accuracy": float(np.mean(auto_vals)) if auto_vals else None,
        "holdings_fully_correct": float(np.mean([r["all_fields_ok"] for r in records])) if records else 0.0,
    }
    if truth is not None:
        tp = sum(1 for r in records if r["pred_contradiction"] and truth.get(r["gt_holding_id"]))
        fp = sum(1 for r in records if r["pred_contradiction"] and not truth.get(r["gt_holding_id"])) + spurious_flagged
        fn = sum(1 for r in records if not r["pred_contradiction"] and truth.get(r["gt_holding_id"]))
        metrics["contradiction"] = {
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn, "flagged": tp + fp, "positives": tp + fn}
    return metrics, records
