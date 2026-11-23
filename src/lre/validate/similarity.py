"""String similarity for names (report 5.3.5, 5.3.7).

Names are compared on a phonetic key built from the ISO 15919 transliteration,
so that harmless spelling variants (सिंह / सिंघ, long / short i, anusvara / half-n)
score high while different people score low.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from indic_transliteration import sanscript
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ..extract.normalise import clean, to_ascii_digits


@lru_cache(maxsize=65536)
def translit(s: str) -> str:
    """Devanagari -> ISO 15919 Latin; Latin text passes through unchanged."""
    return sanscript.transliterate(clean(s), sanscript.DEVANAGARI, sanscript.ISO)


@lru_cache(maxsize=65536)
def phonetic_key(s: str) -> str:
    t = translit(s).lower()
    t = t.replace("ṁ", "n").replace("ṃ", "n").replace("m̐", "n")      # anusvara / chandrabindu ~ n
    t = "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")
    t = t.replace("w", "v").replace("sh", "s").replace("ś", "s").replace("ṣ", "s")
    t = t.replace("ee", "i").replace("oo", "u")
    t = re.sub(r"(?<=[bcdfghjklmnpqrstvxyz])h", "", t)                  # aspiration: gh ~ g, bh ~ b
    t = re.sub(r"a(?=\s|$)", "", t)                                      # final schwa: rama ~ ram
    t = t.replace("v", "b")
    t = re.sub(r"(.)\1+", r"\1", t)                                      # doubled letters
    return re.sub(r"\s+", " ", t).strip()


def name_similarity(a: str | None, b: str | None) -> float:
    """1.0 for identical names, about 0.85-1.0 for spelling variants, low for different people."""
    a, b = clean(a), clean(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ka, kb = phonetic_key(a), phonetic_key(b)
    if ka == kb:
        return 0.98
    return max(fuzz.ratio(ka, kb), fuzz.token_sort_ratio(ka, kb)) / 100.0


def jaro_winkler(a: str, b: str) -> float:
    return JaroWinkler.similarity(clean(a), clean(b))


def norm_edit_distance(a: str, b: str) -> float:
    a, b = clean(a), clean(b)
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def numeric_key(s: str) -> str:
    return to_ascii_digits(clean(s)).replace(" ", "")
