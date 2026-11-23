"""Recogniser charset (report 5.3.2): Devanagari letters, matras, halant and nukta, Latin letters,
ASCII and Devanagari digits and the punctuation used on registers. Index 0 is the CTC blank.

Labels are Unicode code points in logical order after NFC normalisation (U+0958-095F, the
precomposed nukta letters, are composition exclusions, so NFC already stores them as letter + nukta).
"""
from __future__ import annotations

import unicodedata

_DEV = (
    [chr(c) for c in range(0x0901, 0x0904)]          # ँ ं ः
    + [chr(c) for c in range(0x0905, 0x0915)]        # vowels अ..औ (incl. ऋ ऌ ऍ ऎ ए ऐ ऑ ऒ ओ औ)
    + [chr(c) for c in range(0x0915, 0x093A)]        # consonants क..ह
    + ["़", "ऽ"]                           # nukta, avagraha
    + [chr(c) for c in range(0x093E, 0x094E)]        # matras ा..ौ and halant ्
    + ["ॐ", "।", "॥", "॰"]       # ॐ । ॥ ॰
    + [chr(c) for c in range(0x0966, 0x0970)]        # ० .. ९
)
_LATIN = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_DIGITS = [str(d) for d in range(10)]
_PUNCT = list(" /-.(),:")

SYMBOLS: list[str] = list(dict.fromkeys(_DEV + _LATIN + _DIGITS + _PUNCT))
BLANK = 0
INDEX = {ch: i + 1 for i, ch in enumerate(SYMBOLS)}      # 0 is the CTC blank
NUM_CLASSES = len(SYMBOLS) + 1

DIGIT_CHARS = set("0123456789०१२३४५६७८९")


def normalise_label(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def encode(text: str) -> list[int]:
    """Text -> class ids; characters outside the charset are dropped."""
    return [INDEX[ch] for ch in normalise_label(text) if ch in INDEX]


def decode_ids(ids) -> str:
    return "".join(SYMBOLS[i - 1] for i in ids if i > 0)


def coverage(texts) -> tuple[float, dict[str, int]]:
    """Share of characters covered by the charset, and the missing characters with counts."""
    total, missing = 0, {}
    for t in texts:
        for ch in normalise_label(t):
            total += 1
            if ch not in INDEX:
                missing[ch] = missing.get(ch, 0) + 1
    return (1 - sum(missing.values()) / max(1, total)), missing
