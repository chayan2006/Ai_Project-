"""Holding builder, baseline row finding, extraction metrics, U-Net-lite."""
import cv2
import numpy as np
import torch

from lre.baseline.template_tesseract import Template, find_rows
from lre.evaluate.metrics import compare_holding, evaluate_extraction, match_rows
from lre.extract.holding import build_holding, parse_plots
from lre.layout.unet import UNetLite, dice_bce, masks_from_gt, n_params, predict_masks
from lre.schema import Holding, Owner, Plot, Remarks


def test_build_holding_from_cell_texts():
    texts = {
        0: "४१", 1: "59-60", 2: "पत्ती कलां",
        3: "राम सिंह पुत्र हरी सिंह हिस्सा 1/2\nमोहन लाल पुत्र हरी\nसिंह हिस्सा 1/2",
        4: "खुदकाश्त", 5: "चाही", 6: "112//3\n112//4",
        7: "6-12 चाही\n4-08 बारानी\nकुल 11-00", 11: "इंतकाल नं. 2231\nदि. 12-03-2019",
    }
    h = build_holding(texts, "H1", "V000")
    assert h.khewat_no == "41" and h.khatauni_no == "59-60"
    assert [(o.name, o.parent, o.share) for o in h.owners] == [("राम सिंह", "हरी सिंह", "1/2"), ("मोहन लाल", "हरी सिंह", "1/2")]
    assert [(p.khasra, p.area_marla, p.land_type) for p in h.plots] == [("112//3", 132, "chahi"), ("112//4", 88, "barani")]
    assert h.total_area_marla == 220 and h.cultivator == "khudkasht"
    assert (h.remarks.mutation_no, h.remarks.date) == ("2231", "2019-03-12")


def test_parse_plots_keeps_invalid_units_visible():
    plots, total = parse_plots("118", "6-23 चाही")
    assert plots[0].area_raw == "6-23" and total is None


def test_unreadable_share_is_empty_not_one():
    h = build_holding({3: "राम सिंह पुत्र हरी सिंह हिस्सा ॥/2"}, "H1", "V000")
    assert h.owners[0].share == ""


def test_baseline_rows_by_projection():
    img = np.zeros((1500, 2000), np.uint8)
    tpl = Template(x_lines=list(np.linspace(100, 1900, 13).astype(int)), table_top=100, header_bottom=250,
                   table_bottom=1400, width=2000, height=1500)
    for y in (100, 250, 600, 900, 1400):
        cv2.line(img, (100, y), (1900, y), 255, 4)
    cv2.putText(img, "text text text", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)
    rows = find_rows(img, tpl)
    assert [(round(a, -1), round(b, -1)) for a, b in rows] == [(250, 600), (600, 900), (900, 1400)]


def _gt():
    return Holding("G1", "V000", "41", "57", owners=[Owner("राम सिंह", "s/o", "हरी सिंह", "1/2"),
                                                     Owner("मोहन लाल", "s/o", "हरी सिंह", "1/2")],
                   cultivator="khudkasht", plots=[Plot("112//3", 132, "chahi")],
                   remarks=Remarks(raw="इंतकाल नं. 2231", mutation_no="2231"))


def test_compare_holding():
    gt = _gt()
    ok, cer = compare_holding(gt, gt)
    assert all(all(v) for v in ok.values()) and cer["name"][0] == 0
    ok, cer = compare_holding(gt, None)
    assert not any(any(v) for v in ok.values()) and cer["name"][0] == cer["name"][1]
    pred = _gt()
    pred.owners[1].share = "1/3"
    ok, _ = compare_holding(gt, pred)
    assert ok["share"] == [True, False] and all(ok["owner_name"])


def test_match_rows_and_contradiction_metrics():
    assert match_rows([(0, 100), (100, 200)], [(105, 198), (2, 98), (300, 400)]) == [1, 0]
    from lre.schema import to_dict
    g = to_dict(_gt())
    g.update({"row_bbox": [0, 0, 10, 100], "cells": []})
    p = to_dict(_gt())
    p.update({"row_span": [0, 100], "needs_review": False, "contradiction": True})
    m, _ = evaluate_extraction([{"page_id": "P", "holdings": [g]}], {"P": {"holdings": [p]}}, {"G1": False})
    assert m["mean_field_accuracy"] == 1.0 and m["review_share"] == 0.0
    assert m["contradiction"]["fp"] == 1 and m["contradiction"]["precision"] == 0.0


def test_unet_shapes_and_size():
    m = UNetLite()
    assert 400_000 < n_params(m) < 600_000            # report 5.3.1: about 0.5 M parameters
    x = torch.rand(2, 1, 128, 128)
    y = m(x)
    assert y.shape == (2, 2, 128, 128)
    loss = dice_bce(y, torch.zeros_like(y))
    loss.backward()
    assert torch.isfinite(loss)


def test_masks_from_gt_and_tiled_prediction():
    gt = {"h_lines": [{"y": 100, "x0": 10, "x1": 900, "w": 3}], "v_lines": [{"x": 50, "y0": 0, "y1": 600, "w": 3}]}
    m = masks_from_gt(gt, (300, 500))
    assert m.shape == (2, 300, 500) and m[0, 50, 200] == 1 and m[1, 150, 25] == 1 and m[0, 150, 200] == 0
    h_mask, v_mask = predict_masks(UNetLite().eval(), np.full((700, 1100), 240, np.uint8))
    assert h_mask.shape == (700, 1100) and v_mask.dtype == np.uint8
