"""Phase 5 recogniser checks: charset, CRNN, decoding, line data, extractor plumbing."""
import numpy as np
import pytest
import torch

from lre.recognise.charset import INDEX, NUM_CLASSES, decode_ids, encode
from lre.recognise.crnn import CRNN, greedy_decode, n_params
from lre.recognise.data import LineStore, augment, build_lines, collate, line_to_input
from lre.recognise.decode import build_lexicon, decode_line, grammar_ok, lexicon_match, prefix_beam_search


def _fake_lp(text: str, confusions=()) -> np.ndarray:
    ids = encode(text)
    T = 2 * len(ids) + 1
    lp = np.full((T, NUM_CLASSES), np.log(1e-4), np.float32)
    lp[:, 0] = np.log(0.9)
    for k, c in enumerate(ids):
        lp[2 * k + 1, :] = np.log(1e-4)
        lp[2 * k + 1, c] = np.log(0.9)
    for t, a, b in confusions:
        lp[t, :] = np.log(1e-4)
        lp[t, INDEX[a]], lp[t, INDEX[b]] = np.log(0.55), np.log(0.4)
    return lp


def test_charset_round_trip():
    s = "राम सिंह पुत्र हरी सिंह हिस्सा १/२ 112//3 (मुजारा)"
    assert decode_ids(encode(s)) == s
    assert 140 <= NUM_CLASSES <= 170           # report: about 150 symbols plus the blank


def test_crnn_shapes_and_size():
    m = CRNN().eval()
    assert 7_000_000 < n_params(m) < 9_000_000   # report: about 8 M parameters
    lp = m(torch.rand(2, 1, 64, 200))
    assert lp.shape == (50, 2, NUM_CLASSES)
    assert len(greedy_decode(lp)) == 2


def test_beam_search_matches_greedy_on_clean_input():
    lp = _fake_lp("112//3")
    (ids, _), *_ = prefix_beam_search(lp, 5)
    assert decode_ids(ids) == "112//3"


def test_field_aware_decoding():
    d = decode_line(_fake_lp("112//3", [(1, "l", "1")]), "khasra_nos")
    assert d.raw_text == "l12//3" and d.text == "112//3" and d.grammar_ok
    assert not decode_line(_fake_lp("6-23 चाही"), "area").grammar_ok
    assert grammar_ok("area", "६-१२ चाही") and grammar_ok("khatauni_no", "59-60")
    assert not grammar_ok("khasra_nos", "112///3")


def test_lexicon_rescoring_prefers_known_names():
    lex = {"राम", "सिंह", "पुत्र", "हरी", "हिस्सा"}
    assert lexicon_match("राम सिंह पुत्र हरी सिंह हिस्सा 1/2", lex) == 1.0
    assert lexicon_match("रास सिंह", lex) == 0.5


def test_build_lexicon_uses_register_not_row(tmp_path):
    from lre.schema import Owner, RegisterRow
    rows = [RegisterRow("V-1", "V", "1", owners=[Owner("राम सिंह", "s/o", "हरी सिंह", "1/1")], cultivator="khudkasht")]
    lex = build_lexicon(rows)
    assert {"राम", "सिंह", "हरी", "पुत्र", "हिस्सा", "चाही"} <= lex


def test_line_input_and_augment_and_collate():
    crop = np.full((50, 300), 240, np.uint8)
    crop[20:30, 20:280] = 20
    img = line_to_input(crop)
    assert img.shape == (64, 384)
    aug = augment(img, np.random.default_rng(0))
    assert aug.shape == img.shape and aug.dtype == np.uint8
    b = collate([img, img[:, :100]], ["राम", "12"], ["owners", "khewat_no"])
    assert b.images.shape == (2, 1, 64, 384) and b.target_lengths.tolist() == [len(encode("राम")), 2]
    assert b.input_lengths.tolist() == [96, 25]


@pytest.fixture(scope="module")
def sample_lines(tmp_path_factory):
    from lre.config import load_config
    out = tmp_path_factory.mktemp("lines")
    build_lines("data/sample", out, load_config(), workers=2, limit=1)
    return out


def test_build_lines_and_store(sample_lines):
    s = LineStore(sample_lines)
    assert len(s) > 50
    im = s.image(0)
    assert im.shape[0] == 64 and im.shape[1] == s.lines[0]["width"]
    assert all(rec["text"] for rec in s.lines)


def test_crnn_extractor_plumbing_on_sample_page():
    """An untrained model reads nothing useful, but the page must flow through to holdings."""
    import cv2

    from lre.config import load_config
    from lre.io import read_json
    from lre.recognise.extractor import extract_page
    cfg = load_config()
    gt = read_json("data/sample/gt/V000-P0001.json")
    gray = cv2.imread("data/sample/" + gt["image"], cv2.IMREAD_GRAYSCALE)
    hs = extract_page(gray, gt["page_id"], "V000", cfg, CRNN().eval(), torch.device("cpu"), set())
    assert len(hs) == len(gt["holdings"])
    h = hs[0]
    assert {"row_span", "raw_texts", "cell_boxes", "column_conf", "needs_review"} <= set(h)
    assert set(h["cell_boxes"]) == {"0", "1", "3", "4", "6", "7", "11"}
