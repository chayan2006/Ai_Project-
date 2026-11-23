"""Phase 1 checks on the SynJama generator."""
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
from PIL import features

from lre.config import load_config
from lre.io import load_register, read_json
from lre.synth.generator import generate_split, plan_split
from lre.synth.perturb import build_register
from lre.synth.sample import sample_village


@pytest.fixture()
def cfg():
    c = load_config("config.yaml")
    c.synth.khewats_per_village = [20, 30]
    c.synth.contradiction_rate = 0.5      # high rates so every code path runs in a small sample
    c.synth.benign_rate = 0.5
    c.synth.page_error_rate = 0.2
    c.synth.renumber_rate = 0.2
    return c


def test_village_invariants(cfg):
    rng = np.random.default_rng(1)
    v = sample_village(rng, cfg.synth, "V000")
    khasras = [p.khasra for h in v.holdings for p in h.plots]
    assert len(khasras) == len(set(khasras)), "khasra numbers must be unique within a village"
    for h in v.holdings:
        assert sum(Fraction(o.share) for o in h.owners) == 1
        assert 1 <= len(h.owners) <= 4 and 1 <= len(h.plots) <= 5
        assert all(p.area_marla > 0 for p in h.plots)
        if h.total_area_marla is not None:
            assert h.total_area_marla == sum(p.area_marla for p in h.plots)
        if h.remarks.mutation_no:
            assert h.remarks.mutation_no in v.mutation_history[h.khewat_no]


def test_register_perturbations_are_consistent(cfg):
    rng = np.random.default_rng(2)
    v = sample_village(rng, cfg.synth, "V000")
    page = {h.khewat_no: h for h in v.holdings}
    rows, labels = build_register(v, page, cfg.synth, rng)
    assert len(rows) == len(v.holdings)
    kinds = Counter(lab.kind for lab in labels)
    assert kinds["contradiction"] > 0 and kinds["benign"] > 0
    for r in rows:  # the digital register stays internally consistent
        assert sum(Fraction(o.share) for o in r.owners) == 1
    by_khewat = {r.register_row_id: r for r in rows}
    for lab in labels:
        if lab.type == "khasra_moved":
            khasra, target = lab.register_value.split("@")
            assert any(p.khasra == khasra for p in by_khewat[f"V000-{target}"].plots)
        if lab.kind == "contradiction":
            assert lab.rules.startswith("R")
        if lab.kind == "benign":
            assert lab.rules == ""


def test_plan_is_deterministic(cfg):
    a = plan_split("train", 3, cfg, seed=7, village_start=0)
    b = plan_split("train", 3, cfg, seed=7, village_start=0)
    assert [j.page_id for j in a[0]] == [j.page_id for j in b[0]]
    assert [h.holding_id for j in a[0] for h in j.holdings] == [h.holding_id for j in b[0] for h in j.holdings]
    ids = [h.holding_id for j in a[0] for h in j.holdings]
    assert len(ids) == len(set(ids)) and all(ids)


def test_test_split_uses_held_out_fonts(cfg):
    jobs, *_ = plan_split("test", 2, cfg, seed=0, village_start=0)
    assert all(j.style.font_path.endswith(tuple(cfg.synth.fonts.test)) for j in jobs)


@pytest.mark.skipif(not features.check("raqm"), reason="Pillow without libraqm cannot shape Devanagari")
def test_render_one_small_page(cfg, tmp_path):
    cfg.synth.dpi = 100   # small page keeps the test fast
    m = generate_split(tmp_path, "sample", 1, cfg, seed=3)
    assert m["pages"] == 1 and m["holdings_on_pages"] >= 1
    gt = read_json(next((tmp_path / "gt").glob("*.json")))
    assert (tmp_path / gt["image"]).exists()
    assert len(gt["columns"]) == 12 and len(gt["v_lines"]) == 13
    h = gt["holdings"][0]
    assert len(h["cells"]) == 12
    assert any(c["lines"] for c in h["cells"] if c["field"] == "owners")
    assert load_register(tmp_path / "register.csv")
