"""Phase 2 checks: preprocessing and grid detection on a drawn form, and on a rendered SynJama page."""
import cv2
import numpy as np
import pytest
from PIL import features

from lre.config import load_config
from lre.evaluate.layout_eval import evaluate_layout, iou
from lre.layout.cells import find_cells, normalise_line, split_lines
from lre.layout.grid import detect_grid
from lre.preprocess import steps
from lre.preprocess.pipeline import preprocess


def _form(w=2400, h=1700, rows=(300, 600, 900, 1200)):
    img = np.full((h, w), 235, np.uint8)
    xs = np.linspace(100, w - 100, 13).astype(int)
    ys = [100, 200, *rows, h - 100]
    for x in xs:
        cv2.line(img, (int(x), ys[0]), (int(x), ys[-1]), 40, 3)
    for y in ys:
        cv2.line(img, (int(xs[0]), y), (int(xs[-1]), y), 40, 3)
    # some "handwriting" blobs in the first body rows
    for y in rows[:-1]:
        for x in xs[:-1]:
            cv2.putText(img, "41", (int(x) + 20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 30, 3)
    return img, xs, ys


def test_skew_estimate_and_rotation():
    img, *_ = _form()
    skewed = steps.rotate(img, 2.5, border=235)
    est = steps.estimate_skew(skewed)
    assert abs(est - 2.5) < 0.2
    assert abs(steps.estimate_skew(steps.rotate(skewed, -est, border=235))) < 0.2


def test_grid_on_drawn_form():
    cfg = load_config()
    img, xs, ys = _form()
    binary = steps.binarise(img)
    grid = detect_grid(binary, cfg.layout, img)
    assert grid.ok, grid.reason
    assert grid.n_columns == 12
    assert np.allclose(grid.x_lines, xs, atol=3) and np.allclose(grid.y_lines, ys, atol=3)


def test_wrong_column_count_fails_check():
    cfg = load_config()
    img, xs, _ = _form()
    cv2.line(img, (int(xs[5]), 100), (int(xs[5]), 1600), 235, 9)   # erase one column rule
    grid = detect_grid(steps.binarise(img), cfg.layout, img)
    assert not grid.ok and "columns" in grid.reason


def test_cells_skip_empty_rows():
    cfg = load_config()
    img, *_ = _form()
    pre = preprocess(img, cfg.preprocess, denoise=False)
    grid, cells = find_cells(pre, cfg.layout, "P1")
    assert grid.ok
    assert {c.row for c in cells} == {0, 1, 2}   # the 4th body row and the bottom strip are empty
    assert len(cells) == 3 * 12 and cells[3].field_type == "owners"


def test_split_lines_and_normalise():
    crop = np.full((200, 400), 255, np.uint8)
    cv2.putText(crop, "line one", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.putText(crop, "line two", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    spans = split_lines(crop)
    assert len(spans) == 2
    y0, y1 = spans[0]
    line = normalise_line(crop[y0:y1], 64)
    assert line.shape[0] == 64 and line.shape[1] in (128, 256, 384, 512)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


@pytest.mark.skipif(not features.check("raqm"), reason="Pillow without libraqm cannot shape Devanagari")
def test_layout_on_rendered_synjama_page(tmp_path):
    from lre.synth.generator import generate_split
    cfg = load_config()
    generate_split(tmp_path, "sample", 1, cfg, seed=5)
    m = evaluate_layout(tmp_path, cfg)
    assert m["acceptance_passed"], m
    assert m["row_count_accuracy"] == 1.0
