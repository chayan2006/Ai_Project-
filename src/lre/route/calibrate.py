"""Per-field confidence calibration and thresholds (report 5.3.8).

Raw neural confidences are usually over-confident, so for each field type an isotonic regression
maps the raw confidence to c_hat, the estimated probability that the field is correct, fitted on
validation data. The threshold for field type f is the lowest value that keeps the accuracy of
auto-accepted validation fields at or above alpha (default 0.98):

    tau_f = min { t : accuracy(fields of type f with c_hat >= t) >= alpha }
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..extract import normalise as N
from ..schema import Holding

COLUMNS = {"0": "khewat_no", "1": "khatauni_no", "3": "owners", "4": "cultivator", "6": "khasra_nos",
           "7": "area", "11": "remarks"}


def _share(s) -> Optional[Fraction]:
    try:
        return Fraction(s) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def column_correct(col: str, gt: Holding, pred: Holding) -> bool:
    """Is the whole column of the prediction right after normalisation?"""
    if col == "0":
        return N.normalise_khewat(pred.khewat_no) == N.normalise_khewat(gt.khewat_no)
    if col == "1":
        return N.parse_khatauni(pred.khatauni_no) == N.parse_khatauni(gt.khatauni_no)
    if col == "3":
        key = lambda o: (N.normalise_name(o.name), N.normalise_name(o.parent or ""), _share(o.share))  # noqa: E731
        return [key(o) for o in pred.owners] == [key(o) for o in gt.owners]
    if col == "4":
        return N.normalise_cultivator(pred.cultivator) == N.normalise_cultivator(gt.cultivator)
    if col == "6":
        return [N.parse_khasra(p.khasra or "") for p in pred.plots] == [p.khasra for p in gt.plots]
    if col == "7":
        return ([(p.area_marla, p.land_type) for p in pred.plots] == [(p.area_marla, p.land_type) for p in gt.plots]
                and pred.total_area_marla == gt.total_area_marla)
    if col == "11":
        return (pred.remarks.mutation_no or None) == (gt.remarks.mutation_no or None)
    raise KeyError(col)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


class Calibrators:
    def __init__(self, models: dict | None = None, thresholds: dict | None = None, alpha: float = 0.98):
        self.models = models or {}
        self.thresholds = thresholds or {}
        self.alpha = alpha

    def fit(self, samples: dict[str, tuple[list[float], list[bool]]], alpha: float = 0.98) -> dict:
        from sklearn.isotonic import IsotonicRegression
        self.alpha = alpha
        report = {}
        for col, (conf, ok) in samples.items():
            c, y = np.asarray(conf, float), np.asarray(ok, float)
            if len(c) < 20:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(c, y)
            cal = iso.predict(c)
            tau = float("inf")
            for t in np.unique(cal):
                sel = cal >= t
                if sel.any() and y[sel].mean() >= alpha:
                    tau = float(t)
                    break
            self.models[col], self.thresholds[col] = iso, tau
            report[COLUMNS.get(col, col)] = {
                "n": int(len(c)), "accuracy": round(float(y.mean()), 4), "ece_raw": round(ece(c, y), 4),
                "ece_calibrated": round(ece(cal, y), 4), "tau": None if tau == float("inf") else round(tau, 4),
                "auto_share": round(float((cal >= tau).mean()), 4)}
        return report

    def calibrate(self, col: str, conf: Optional[float]) -> Optional[float]:
        if conf is None or col not in self.models:
            return None
        return float(self.models[col].predict([conf])[0])

    def passes(self, col: str, conf: Optional[float]) -> bool:
        c = self.calibrate(col, conf)
        return c is not None and c >= self.thresholds.get(col, float("inf"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "thresholds": self.thresholds, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrators":
        d = joblib.load(path)
        return cls(d["models"], d["thresholds"], d["alpha"])
