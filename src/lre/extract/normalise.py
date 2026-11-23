"""Field normalisers (report Table 4 and 4.4 'Label cleaning').

All functions are pure and tolerant: they return None (or an empty list) for
text they cannot parse instead of raising, because their input will often be
noisy recogniser output.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from ..schema import Owner

# ----------------------------------------------------------------------------- digits

DEV_DIGITS = "०१२३४५६७८९"
_TO_ASCII = str.maketrans(DEV_DIGITS, "0123456789")
_TO_DEV = str.maketrans("0123456789", DEV_DIGITS)


def to_ascii_digits(s: str) -> str:
    return s.translate(_TO_ASCII)


def to_dev_digits(s: str) -> str:
    return s.translate(_TO_DEV)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean(s: Optional[str]) -> str:
    """NFC + collapsed whitespace; the first step for every field."""
    return collapse_ws(nfc(s or ""))


# ----------------------------------------------------------------------------- relations

RELATION_WORDS: dict[str, str] = {
    "पुत्र": "s/o", "पु.": "s/o", "पु०": "s/o", "वल्द": "s/o", "s/o": "s/o", "S/O": "s/o", "s/o.": "s/o",
    "पुत्री": "d/o", "दुख्तर": "d/o", "d/o": "d/o", "D/O": "d/o",
    "पत्नी": "w/o", "धर्मपत्नी": "w/o", "जौजा": "w/o", "बेवा": "w/o", "w/o": "w/o", "W/O": "w/o",
}


def normalise_relation(word: str) -> Optional[str]:
    return RELATION_WORDS.get(clean(word))


# ----------------------------------------------------------------------------- shares

_SHARE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
SHARE_WORDS = ("हिस्सा", "हि.", "भाग", "share")


def parse_share(s: str) -> Optional[Fraction]:
    """'हिस्सा 1/2', '१/२', '1/1' -> Fraction. Returns None for a zero denominator."""
    t = to_ascii_digits(clean(s))
    m = _SHARE_RE.search(t)
    if not m:
        bare = re.fullmatch(r"(?:\D*\s)?(\d+)", t)
        return Fraction(int(bare.group(1))) if bare and bare.group(1) == "1" else None
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return None
    return Fraction(num, den)


def format_share(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


# ----------------------------------------------------------------------------- area


@dataclass(frozen=True)
class Area:
    kanal: int
    marla: int

    @property
    def total_marla(self) -> int:
        return self.kanal * 20 + self.marla

    @property
    def valid(self) -> bool:  # rule R3
        return 0 <= self.marla < 20 and self.total_marla > 0

    def __str__(self) -> str:
        return f"{self.kanal}-{self.marla:02d}"


_AREA_RE = re.compile(r"(\d+)\s*[-–—]\s*(\d{1,2})(?!\d)")


def parse_area(s: str) -> Optional[Area]:
    """'6-12' (kanal-marla) -> Area(6, 12). Marla >= 20 is returned as-is so R3 can flag it."""
    m = _AREA_RE.search(to_ascii_digits(clean(s)))
    if not m:
        return None
    return Area(int(m.group(1)), int(m.group(2)))


def marla_to_kanal_marla(total: int) -> str:
    return str(Area(total // 20, total % 20))


# ----------------------------------------------------------------------------- land type

LAND_TYPES: dict[str, str] = {
    "चाही": "chahi", "chahi": "chahi",
    "नहरी": "nehri", "nehri": "nehri",
    "बारानी": "barani", "barani": "barani",
    "आबी": "abi", "abi": "abi",
    "बंजर कदीम": "banjar_kadim", "banjar kadim": "banjar_kadim",
    "बंजर जदीद": "banjar_jadid", "banjar jadid": "banjar_jadid",
    "बंजर": "banjar", "banjar": "banjar",
    "गैर मुमकिन": "gair_mumkin", "गैरमुमकिन": "gair_mumkin", "gair mumkin": "gair_mumkin",
}
LAND_TYPE_CODES = sorted(set(LAND_TYPES.values()))
# Longest keys first so that 'बंजर कदीम' wins over 'बंजर'.
_LAND_KEYS = sorted(LAND_TYPES, key=len, reverse=True)


def normalise_land_type(s: str) -> Optional[str]:
    t = clean(s).lower()
    for key in _LAND_KEYS:
        if key.lower() in t:
            return LAND_TYPES[key]
    return None


# ----------------------------------------------------------------------------- khasra / khatauni / khewat

_KHASRA_RE = re.compile(r"(\d+)\s*(//|/)\s*(\d+)|(\d+)")


def parse_khasra(s: str) -> Optional[str]:
    """Return the first khasra number in canonical form: '112//3', '119/1' or '118'."""
    m = _KHASRA_RE.search(to_ascii_digits(clean(s)))
    if not m:
        return None
    if m.group(4) is not None:
        return m.group(4)
    return f"{int(m.group(1))}{m.group(2)}{int(m.group(3))}"


def parse_khasra_list(text: str) -> list[str]:
    """Every khasra number in a cell, one per line or comma-separated."""
    out = []
    for m in _KHASRA_RE.finditer(to_ascii_digits(clean(text))):
        if m.group(4) is not None:
            out.append(m.group(4))
        else:
            out.append(f"{int(m.group(1))}{m.group(2)}{int(m.group(3))}")
    return out


def is_valid_khasra(s: str) -> bool:
    return re.fullmatch(r"\d+(//\d+|/\d+)?", s or "") is not None


def parse_khatauni(s: str, max_range: int = 20) -> list[str]:
    """'59-60' -> ['59', '60']; '59, 61' -> ['59', '61']. Long ranges keep their endpoints only."""
    t = to_ascii_digits(clean(s))
    out: list[str] = []
    for part in re.split(r"[,\s]+", t):
        m = re.fullmatch(r"(\d+)[-–](\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a <= max_range:
                out.extend(str(i) for i in range(a, b + 1))
            else:
                out.extend([str(a), str(b)])
        elif re.fullmatch(r"\d+", part):
            out.append(str(int(part)))
    return out


def normalise_khewat(s: str) -> str:
    """Digits to ASCII, keep a suffix such as 'min' (मिन)."""
    t = to_ascii_digits(clean(s)).replace(" ", "")
    t = t.replace("मिन", "min")
    m = re.match(r"(\d+)(.*)", t)
    if not m:
        return t
    return f"{int(m.group(1))}{m.group(2)}"


# ----------------------------------------------------------------------------- mutation

_MUT_NO_RE = re.compile(r"(?:इंतकाल|इन्तकाल|म्यू\.?|mut\.?|mutation)\s*(?:नं\.?|न\.|no\.?|संख्या)?\s*(\d+)", re.I)
_DATE_RE = re.compile(r"(\d{1,2})[-./](\d{1,2})[-./](\d{4})")


def parse_mutation(s: str) -> tuple[Optional[str], Optional[str]]:
    """'इंतकाल नं. 2231 दि. 12-03-2019' -> ('2231', '2019-03-12')."""
    t = to_ascii_digits(clean(s))
    m = _MUT_NO_RE.search(t)
    mut = m.group(1) if m else None
    d = _DATE_RE.search(t)
    date = None
    if d:
        dd, mm, yyyy = int(d.group(1)), int(d.group(2)), int(d.group(3))
        if 1 <= dd <= 31 and 1 <= mm <= 12:
            date = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    return mut, date


# ----------------------------------------------------------------------------- cultivator

KHUDKASHT_WORDS = ("खुदकाश्त", "खुद काश्त", "khudkasht")


def normalise_cultivator(s: str) -> str:
    """'खुदकाश्त' -> 'khudkasht' (owner cultivates); otherwise the cleaned name text."""
    t = clean(s)
    if any(w in t.lower() for w in KHUDKASHT_WORDS):
        return "khudkasht"
    t = re.sub(r"\((?:मुजारा|tenant)\)", "", t).strip()
    return " ".join(normalise_relation(tok) or tok for tok in t.split())


# ----------------------------------------------------------------------------- names and owners


def normalise_name(s: str) -> str:
    return clean(s).replace("‌", "").replace("‍", "")


_ENTRY_END_RE = re.compile(r"((?:हिस्सा|हि\.|भाग|share)?\s*[\d०-९]+\s*/\s*[\d०-९]+)")
_SHARE_WORD_END_RE = re.compile(r"(?:हिस्सा|हि\.|भाग|share)\s*\S+")


def split_owner_entries(text: str) -> list[str]:
    """Split an owner cell into entries; each entry ends with its share.

    The share word ('हिस्सा') is the anchor, so an entry still ends correctly when the
    fraction after it is misread; without share words, bare fractions are used.
    """
    t = clean(text)
    pattern = _SHARE_WORD_END_RE if _SHARE_WORD_END_RE.search(t) else _ENTRY_END_RE
    parts, last = [], 0
    for m in pattern.finditer(t):
        parts.append(t[last:m.end()].strip())
        last = m.end()
    tail = t[last:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def parse_owner_line(s: str) -> Optional[Owner]:
    """Regex parse of one owner entry: 'राम सिंह पुत्र हरी सिंह हिस्सा 1/2'.

    This is the template-baseline parser; the proposed system uses a CRF (report 5.3.4).
    """
    t = clean(s)
    share = None
    for w in SHARE_WORDS:
        idx = t.find(w)
        if idx >= 0:
            share = parse_share(t[idx:])
            t = t[:idx].strip()
            break
    if share is None:
        m = _SHARE_RE.search(to_ascii_digits(t))
        if m:
            share = Fraction(int(m.group(1)), int(m.group(2))) if int(m.group(2)) else None
            t = _SHARE_RE.sub("", to_ascii_digits(t)).strip()
    tokens = t.split()
    for i, tok in enumerate(tokens):
        rel = normalise_relation(tok)
        if rel:
            name = " ".join(tokens[:i])
            parent = " ".join(tokens[i + 1:])
            return Owner(name=name, relation=rel, parent=parent or None,
                         share=format_share(share) if share is not None else "")
    if not tokens:
        return None
    return Owner(name=" ".join(tokens), share=format_share(share) if share is not None else "")
