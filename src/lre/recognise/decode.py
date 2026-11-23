"""Field-aware CTC decoding (report 5.3.3) and per-line confidence (5.3.8).

* numeric columns (khewat, khatauni, khasra) are decoded with a digit-only alphabet and must match
  a grammar; the area column must start with kanal-marla (marla <= 19);
* name columns (owners, cultivator) are re-scored with a lexicon:
  score = log P_CTC(text) + lambda * match(text, lexicon). The lexicon is built from the whole
  village register, never from the row being checked, so the decoder is not pulled towards
  agreement and real contradictions stay visible;
* the unconstrained best path is always kept as ``raw_text``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..extract.normalise import to_ascii_digits
from .charset import DIGIT_CHARS, INDEX, SYMBOLS, decode_ids

NEG = -1e30


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def prefix_beam_search(lp: np.ndarray, beam: int = 10, allowed: Optional[set[int]] = None,
                       prune: float = -10.0) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search on T x C log-probs. Returns [(ids, log P)], best first."""
    T, C = lp.shape
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG]}        # prefix -> [log p_blank, log p_nonblank]
    for t in range(T):
        row = lp[t]
        cands = [c for c in np.nonzero(row > prune)[0].tolist() if c != 0 and (allowed is None or c in allowed)]
        nxt: dict[tuple[int, ...], list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            total = _lse(pb, pnb)
            e = nxt.setdefault(prefix, [NEG, NEG])
            e[0] = _lse(e[0], total + row[0])                           # blank keeps the prefix
            if prefix:
                e[1] = _lse(e[1], pnb + row[prefix[-1]])                # repeated last char collapses
            for c in cands:
                new = prefix + (c,)
                en = nxt.setdefault(new, [NEG, NEG])
                if prefix and c == prefix[-1]:
                    en[1] = _lse(en[1], pb + row[c])                    # needs a blank in between
                else:
                    en[1] = _lse(en[1], total + row[c])
        beams = dict(sorted(nxt.items(), key=lambda kv: _lse(*kv[1]), reverse=True)[:beam])
    return [(p, _lse(*v)) for p, v in sorted(beams.items(), key=lambda kv: _lse(*kv[1]), reverse=True)]


def best_path(lp: np.ndarray) -> tuple[str, float]:
    """Greedy text and raw confidence (geometric mean of emitted-character posteriors)."""
    idx = lp.argmax(-1)
    ids, logs, prev = [], [], 0
    for t, i in enumerate(idx.tolist()):
        if i != 0 and i != prev:
            ids.append(i)
            logs.append(float(lp[t, i]))
        prev = i
    conf = math.exp(sum(logs) / len(logs)) if logs else math.exp(float(lp[:, 0].mean()))
    return decode_ids(ids), conf


# ----------------------------------------------------------------------------- field rules

_DIGIT_IDS = {INDEX[c] for c in DIGIT_CHARS}
ALPHABETS = {
    "khewat_no": _DIGIT_IDS | {INDEX["-"]},
    "khatauni_no": _DIGIT_IDS | {INDEX["-"], INDEX[","], INDEX[" "]},
    "khasra_nos": _DIGIT_IDS | {INDEX["/"]},
}
GRAMMARS = {
    "khewat_no": re.compile(r"^\d+$"),
    "khatauni_no": re.compile(r"^\d+(-\d+)?$"),
    "khasra_nos": re.compile(r"^\d+(//\d+|/\d+)?$"),
}
_AREA = re.compile(r"^(?:कुल\s+)?(\d+)-(\d{1,2})(\s|$)")
NAME_FIELDS = {"owners", "cultivator"}


def grammar_ok(field: str, text: str) -> bool:
    t = to_ascii_digits(text.strip())
    if field in GRAMMARS:
        return bool(GRAMMARS[field].match(t))
    if field == "area":
        m = _AREA.match(t)
        return bool(m) and int(m.group(2)) < 20
    return True


def lexicon_match(text: str, lexicon: set[str]) -> float:
    """Share of words in the hypothesis that are known words (names, relation and form words)."""
    words = [w for w in text.split() if not any(ch in DIGIT_CHARS or ch == "/" for ch in w)]
    if not words:
        return 0.0
    return sum(w in lexicon for w in words) / len(words)


@dataclass
class LineDecode:
    text: str           # field-aware decoding
    raw_text: str       # unconstrained best path
    conf_raw: float     # geometric mean of character posteriors along the best path
    margin: float       # log P(best) - log P(second) of the final beam (inf when only one hypothesis)
    grammar_ok: bool


def decode_line(lp: np.ndarray, field: str, lexicon: Optional[set[str]] = None, beam: int = 10,
                lexicon_weight: float = 0.6) -> LineDecode:
    raw, conf = best_path(lp)
    if field in ALPHABETS:
        hyps = prefix_beam_search(lp, beam, ALPHABETS[field])
    else:
        hyps = prefix_beam_search(lp, beam)
    if not hyps:
        return LineDecode(raw, raw, conf, 0.0, grammar_ok(field, raw))
    texts = [(decode_ids(ids), lp_) for ids, lp_ in hyps]
    margin = texts[0][1] - texts[1][1] if len(texts) > 1 else float("inf")
    scored = texts
    if field in NAME_FIELDS and lexicon:
        scored = sorted(((t, s + lexicon_weight * lexicon_match(t, lexicon)) for t, s in texts),
                        key=lambda x: x[1], reverse=True)
    if field in GRAMMARS or field == "area":
        ok = [(t, s) for t, s in scored if grammar_ok(field, t)]
        if ok:
            scored = ok
    best = scored[0][0]
    return LineDecode(best, raw, conf, margin, grammar_ok(field, best))


def build_lexicon(register_rows) -> set[str]:
    """Words of the whole village register plus the fixed vocabulary of the form (relations, shares, land types)."""
    from ..extract.normalise import KHUDKASHT_WORDS, LAND_TYPES, RELATION_WORDS, SHARE_WORDS
    words: set[str] = set()
    for r in register_rows:
        for o in r.owners:
            words.update((o.name or "").split())
            words.update((o.parent or "").split())
        words.update(w for w in (r.cultivator or "").split() if w not in ("s/o", "d/o", "w/o"))
    words.update(w for k in RELATION_WORDS for w in k.split())
    words.update(SHARE_WORDS)
    words.update(w for k in LAND_TYPES for w in k.split())
    words.update(w for k in KHUDKASHT_WORDS for w in k.split())
    words.update({"(मुजारा)", "मुजारा", "कुल", "इंतकाल", "नं.", "दि."})
    return {w for w in words if w}


__all__ = ["LineDecode", "decode_line", "prefix_beam_search", "best_path", "build_lexicon", "grammar_ok",
           "SYMBOLS"]
