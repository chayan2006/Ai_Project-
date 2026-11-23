import numpy as np

from lre.route.calibrate import Calibrators, column_correct, ece
from lre.schema import Holding, Owner, Plot
from lre.validate.scorer import FEATURES, ContradictionScorer, flag_features


def _flag(rule, **ev):
    return {"rule": rule, "severity": "high", "evidence": ev}


def test_flag_features_shape_and_values():
    h = {"flags": [], "column_conf": {"3": 0.7, "7": 0.95}, "link": {"method": "khewat", "score": 1.0}}
    f6 = _flag("R6", scan_name="राम सिंह", register_name="राम सिंह", similarity=1.0, scan_share="1/2",
               register_share="1/3")
    h["flags"] = [f6, _flag("R7", scan_area=132, register_area=150)]
    x = flag_features(f6, h)
    assert x.shape == (len(FEATURES),)
    d = dict(zip(FEATURES, x))
    assert d["rule_R6"] == 1 and d["other_R7"] == 1 and d["field_conf"] == 0.7 and d["jw"] == 1.0
    assert abs(d["num_abs_diff"] - (1 / 2 - 1 / 3)) < 1e-9


def test_scorer_learns_and_tunes_theta():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random(n) < 0.3
    X = rng.normal(0, 1, (n, len(FEATURES)))
    X[:, FEATURES.index("field_conf")] = np.where(y, rng.uniform(0.9, 1, n), rng.uniform(0, 0.6, n))
    s = ContradictionScorer().fit(X, y)
    p = s.predict(X)
    assert p[y].mean() > p[~y].mean() + 0.5
    theta = s.tune_theta(list(p), list(y))
    assert 0 < theta <= 1
    flagged = np.round(p, 4) >= theta
    assert flagged[y].mean() > 0.9 and flagged[~y].mean() < 0.1


def test_calibration_threshold_meets_alpha():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 2000)
    correct = rng.random(2000) < conf ** 0.5                 # over-confident raw scores
    cal = Calibrators()
    rep = cal.fit({"3": (list(conf), list(correct))}, alpha=0.95)["owners"]
    assert rep["ece_calibrated"] < rep["ece_raw"]
    sel = np.array([cal.passes("3", c) for c in conf])
    assert correct[sel].mean() >= 0.95 and 0 < sel.mean() < 1
    assert not cal.passes("3", None) and not cal.passes("6", 0.99)   # unknown field never auto-accepts


def test_ece_and_column_correct():
    assert ece(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    g = Holding("G", "V", "41", owners=[Owner("राम", "s/o", "हरी", "1/1")], plots=[Plot("112//3", 132, "chahi")])
    p = Holding("P", "V", "४१", owners=[Owner("राम", "s/o", "हरी", "2/2")], plots=[Plot("११२//३", 132, "chahi")])
    assert all(column_correct(c, g, p) for c in ("0", "3", "6", "7", "11"))
