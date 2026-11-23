"""ML contradiction scorer (report 5.3.7).

A rule fires whenever values disagree, but a disagreement may be a recognition error or a harmless
variant. For each fired rule we build features (string distances on original and transliterated
text, relative numeric difference, the recognition confidence of the field, link method, which
other rules fired) and a gradient-boosted tree classifier estimates P(true contradiction).
It is trained on the extractor's own outputs on SynJama pages, so the noise is realistic; the
flag threshold theta maximises F0.5 on validation because precision is the assessed metric.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .similarity import jaro_winkler, norm_edit_distance, translit

RULES = [f"R{i}" for i in range(1, 11)]
RULE_COLUMN = {"R1": "3", "R5": "3", "R6": "3", "R2": "7", "R3": "7", "R4": "7", "R7": "7", "R8": "6",
               "R9": "4", "R10": "11"}
FEATURES = (["rule_" + r for r in RULES] + ["other_" + r for r in RULES]
            + ["n_flags", "field_conf", "min_conf", "link_khewat", "link_khasra", "link_score",
               "similarity", "jw", "jw_translit", "edit", "num_rel_diff", "num_abs_diff", "unreadable"])


def _frac(s) -> Optional[float]:
    try:
        return float(Fraction(str(s)))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _strings(rule: str, ev: dict) -> tuple[str, str]:
    if rule == "R6":
        return ev.get("scan_name", ""), ev.get("register_name", "")
    if rule == "R9":
        return ev.get("scan", ""), ev.get("register", "")
    if rule == "R10":
        hist = ev.get("register_history") or []
        m = ev.get("mutation_no", "")
        best = min(hist, key=lambda h: norm_edit_distance(m, h), default="")
        return m, best
    return "", ""


def flag_features(flag: dict, holding: dict) -> np.ndarray:
    ev = flag.get("evidence") or {}
    rule = flag["rule"]
    f = dict.fromkeys(FEATURES, np.nan)
    for r in RULES:
        f["rule_" + r] = float(r == rule)
        f["other_" + r] = float(any(x["rule"] == r for x in holding.get("flags", []) if x is not flag))
    f["n_flags"] = len(holding.get("flags", []))
    confs = {k: v for k, v in (holding.get("column_conf") or {}).items() if v is not None}
    f["field_conf"] = confs.get(RULE_COLUMN.get(rule, ""), np.nan)
    f["min_conf"] = min(confs.values()) if confs else np.nan
    link = holding.get("link") or {}
    f["link_khewat"] = float(link.get("method") == "khewat")
    f["link_khasra"] = float(link.get("method") == "khasra_jaccard")
    f["link_score"] = link.get("score", np.nan)
    f["similarity"] = ev.get("similarity", ev.get("best_similarity", np.nan))
    a, b = _strings(rule, ev)
    if a or b:
        f["jw"] = jaro_winkler(a, b)
        f["jw_translit"] = jaro_winkler(translit(a), translit(b))
        f["edit"] = norm_edit_distance(a, b)
    pairs = {"R6": ("scan_share", "register_share"), "R7": ("scan_area", "register_area"),
             "R2": ("written", "computed")}
    if rule in pairs:
        x, y = (_frac(ev.get(k)) for k in pairs[rule])
        if x is not None and y is not None:
            f["num_abs_diff"] = abs(x - y)
            f["num_rel_diff"] = abs(x - y) / max(abs(y), 1e-6)
    if rule == "R1":
        f["unreadable"] = float(not ev.get("readable", True))
        t = _frac(ev.get("total"))
        if t is not None:
            f["num_abs_diff"] = abs(t - 1)
    return np.array([f[k] for k in FEATURES], dtype=np.float64)


def _fbeta(p: float, r: float, beta: float = 0.5) -> float:
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


class ContradictionScorer:
    def __init__(self, model=None, theta: float = 0.5):
        self.model, self.theta = model, theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ContradictionScorer":
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=300,
                                                    class_weight="balanced", random_state=0)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)

    def tune_theta(self, holding_scores: list[float], holding_truth: list[bool]) -> float:
        """Holding-level threshold on max flag probability that maximises F0.5."""
        best_t, best_f = 0.5, -1.0
        s, t = np.round(np.array(holding_scores, float), 4), np.array(holding_truth, bool)
        for th in np.unique(s):
            pred = s >= th
            tp = (pred & t).sum()
            p = tp / max(1, pred.sum())
            r = tp / max(1, t.sum())
            if _fbeta(p, r) > best_f:
                best_f, best_t = _fbeta(p, r), float(th)
        self.theta = best_t
        return best_t

    def score_holding(self, holding: dict, severities=("high", "medium")) -> float:
        """Sets ``score`` on every flag of the holding dict; returns the max over contradiction-severity flags."""
        flags = holding.get("flags", [])
        if not flags:
            return 0.0
        p = self.predict(np.stack([flag_features(fl, holding) for fl in flags]))
        for fl, v in zip(flags, p):
            fl["score"] = float(v)
        return max((fl["score"] for fl in flags if fl["severity"] in severities), default=0.0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "theta": self.theta, "features": FEATURES}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ContradictionScorer":
        d = joblib.load(path)
        return cls(d["model"], d["theta"])
