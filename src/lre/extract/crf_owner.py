"""Owner-cell parsing with a linear-chain CRF (report 5.3.4).

An owner cell such as "राम सिंह पुत्र हरी सिंह हिस्सा 1/2 मोहन लाल ..." is tagged token by token:
NAME_B (first token of an owner), NAME, RELATION, PARENT, SHARE (share word or fraction), OTHER.
Training strings come from SynJama ground truth with OCR-style noise injected, so the tagger
tolerates recognition errors that break the baseline's regular expressions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..schema import Owner
from .normalise import SHARE_WORDS, clean, format_share, normalise_relation, parse_share

_FRACTION = re.compile(r"^[\d०-९Il|]+\s*/\s*[\d०-९Il|]+$")
CONFUSIONS = [("ब", "व"), ("व", "ब"), ("घ", "ध"), ("ध", "घ"), ("भ", "म"), ("म", "भ"), ("ी", "ि"), ("ि", "ी"),
              ("ु", "ू"), ("ं", ""), ("्", ""), ("1", "l"), ("1", "I"), ("१", "9"), ("७", "6"), ("/", "1"),
              ("स", "रा"), ("र", "२"), ("प", "ष"), ("ष", "प")]


# ----------------------------------------------------------------------------- features


def _token_features(tokens: list[str], i: int) -> dict:
    t = tokens[i]
    f = {
        "bias": 1.0, "tok": t, "pre1": t[:1], "pre2": t[:2], "suf1": t[-1:], "suf2": t[-2:], "suf3": t[-3:],
        "len": min(len(t), 8), "is_fraction": bool(_FRACTION.match(t)), "has_slash": "/" in t,
        "has_digit": any(ch.isdigit() for ch in t), "is_relation": normalise_relation(t) is not None,
        "is_share_word": t in SHARE_WORDS, "is_latin": t.isascii() and t.isalpha(),
        "pos": min(i, 10), "rel_pos": round(i / max(1, len(tokens) - 1), 1),
    }
    for k, name in ((-1, "prev"), (1, "next"), (-2, "prev2"), (2, "next2")):
        j = i + k
        if 0 <= j < len(tokens):
            u = tokens[j]
            f.update({f"{name}_tok": u, f"{name}_rel": normalise_relation(u) is not None,
                      f"{name}_frac": bool(_FRACTION.match(u)), f"{name}_share": u in SHARE_WORDS})
        else:
            f[f"{name}_edge"] = True
    return f


def featurise(tokens: list[str]) -> list[dict]:
    return [_token_features(tokens, i) for i in range(len(tokens))]


# ----------------------------------------------------------------------------- training data


def align_owner_cell(text: str, owners: list[dict]) -> Optional[list[str]]:
    """Token tags for a ground-truth owner cell, or None if the text does not align with the owners."""
    tokens = clean(text).split()
    tags: list[str] = []
    for o in owners:
        name_n = len(o["name"].split())
        parent_n = len((o.get("parent") or "").split())
        tags += ["NAME_B"] + ["NAME"] * (name_n - 1)
        if o.get("relation"):
            tags += ["RELATION"] + ["PARENT"] * parent_n
        tags += ["SHARE", "SHARE"]                   # share word + fraction
    if len(tags) != len(tokens):
        return None
    for tok, tag in zip(tokens, tags):
        if tag == "RELATION" and normalise_relation(tok) is None:
            return None
    return tags


def add_noise(tokens: list[str], rng: np.random.Generator, p_char: float = 0.08, p_merge: float = 0.03,
              p_drop: float = 0.02) -> tuple[list[str], list[int]]:
    """OCR-like corruption. Returns new tokens and, for each, the index of its source token."""
    out, src = [], []
    for i, t in enumerate(tokens):
        if rng.random() < p_drop and len(tokens) > 3:
            continue
        chars = list(t)
        for k in range(len(chars)):
            if rng.random() < p_char:
                a, b = CONFUSIONS[int(rng.integers(len(CONFUSIONS)))]
                if chars[k] == a or rng.random() < 0.3:
                    chars[k] = b if chars[k] == a else chars[k] + (b if rng.random() < 0.3 else "")
        t2 = "".join(chars) or t
        if out and rng.random() < p_merge:
            out[-1] += t2                              # lost space: merge with the previous token
            continue
        out.append(t2)
        src.append(i)
    return out, src


def training_examples(gt_dirs: list[str | Path], rng: np.random.Generator, noisy_copies: int = 2,
                      limit_pages: Optional[int] = None) -> tuple[list[list[dict]], list[list[str]]]:
    X, Y = [], []
    for d in gt_dirs:
        for p in sorted(Path(d, "gt").glob("*.json"))[:limit_pages]:
            gt = json.loads(p.read_text(encoding="utf-8"))
            for h in gt["holdings"]:
                cell = next(c for c in h["cells"] if c["field"] == "owners")
                text = " ".join(ln["text"] for ln in cell["lines"])
                tags = align_owner_cell(text, h["owners"])
                if tags is None:
                    continue
                tokens = clean(text).split()
                X.append(featurise(tokens))
                Y.append(tags)
                for _ in range(noisy_copies):
                    nt, src = add_noise(tokens, rng)
                    X.append(featurise(nt))
                    Y.append([tags[s] for s in src])
    return X, Y


# ----------------------------------------------------------------------------- model


class CRFOwnerParser:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, Y, c1: float = 0.1, c2: float = 0.01, max_iter: int = 150) -> "CRFOwnerParser":
        import sklearn_crfsuite
        self.model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter,
                                          all_possible_transitions=True)
        self.model.fit(X, Y)
        return self

    def tag(self, tokens: list[str]) -> list[str]:
        return self.model.predict_single(featurise(tokens)) if tokens else []

    def __call__(self, text: str) -> list[Owner]:
        tokens = clean(text).split()
        owners: list[dict] = []
        for tok, tag in zip(tokens, self.tag(tokens)):
            if tag == "NAME_B" or (tag == "NAME" and (not owners or owners[-1]["parts"]["SHARE"])):
                owners.append({"parts": {"NAME": [], "RELATION": [], "PARENT": [], "SHARE": []}})
                tag = "NAME"
            if not owners:
                continue
            if tag in owners[-1]["parts"]:
                owners[-1]["parts"][tag].append(tok)
        out = []
        for o in owners:
            p = o["parts"]
            share_txt = " ".join(t for t in p["SHARE"] if t not in SHARE_WORDS)
            share = parse_share(share_txt) if share_txt else None
            rel = p["RELATION"][0] if p["RELATION"] else None
            out.append(Owner(name=" ".join(p["NAME"]), relation=normalise_relation(rel) if rel else None,
                             parent=" ".join(p["PARENT"]) or None,
                             share=format_share(share) if share is not None else ""))
        return [o for o in out if o.name]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "CRFOwnerParser":
        return cls(joblib.load(path))



def train_crf(gt_dirs: list[str | Path], out: str | Path, limit_pages: Optional[int] = 400, seed: int = 0) -> int:
    X, Y = training_examples(gt_dirs, np.random.default_rng(seed), noisy_copies=2, limit_pages=limit_pages)
    CRFOwnerParser().fit(X, Y).save(out)
    return len(X)


def load_default(models_dir: str | Path) -> Optional["CRFOwnerParser"]:
    p = Path(models_dir) / "crf_owner_v1.joblib"
    return CRFOwnerParser.load(p) if p.exists() else None
