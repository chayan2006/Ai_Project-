"""Knowledge base of typed facts with forward-chaining derivations (report 5.3.6, Figure 5).

Base facts come from the extracted holding (side 'scan') and its linked register
row (side 'register'). Derivation rules add facts such as the share sum, the
owner matching or the plots present on both sides; they run repeatedly until no
new fact appears (a fixpoint), so a derived fact can feed further derivations.
The check rules in rules.yaml then fire on these facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..extract import normalise as N
from ..schema import Holding, RegisterRow
from .similarity import name_similarity

RELATIONS = frozenset({"s/o", "d/o", "w/o"})
LAND_TYPES = frozenset(N.LAND_TYPE_CODES)


# ----------------------------------------------------------------------------- base facts

@dataclass(frozen=True)
class HoldingFact:
    holding_id: str
    village_code: str
    khewat: str
    linked_row: Optional[str]


@dataclass(frozen=True)
class ScanOwner:
    idx: int
    name: str
    relation: Optional[str]
    parent: Optional[str]
    share: Optional[Fraction]


@dataclass(frozen=True)
class RegOwner:
    idx: int
    name: str
    relation: Optional[str]
    parent: Optional[str]
    share: Optional[Fraction]


@dataclass(frozen=True)
class ScanPlot:
    idx: int
    khasra: str
    kanal: Optional[int]
    marla: Optional[int]
    area_marla: Optional[int]
    land_type: Optional[str]


@dataclass(frozen=True)
class RegPlot:
    idx: int
    khasra: str
    area_marla: Optional[int]


@dataclass(frozen=True)
class WrittenTotal:
    written: int


@dataclass(frozen=True)
class Cultivator:
    side: str
    value: str


@dataclass(frozen=True)
class Mutation:
    mutation_no: str


@dataclass(frozen=True)
class RegisterHistory:
    mutations: frozenset


@dataclass(frozen=True)
class KhasraOwner:
    """Register index entry: which register row holds a khasra the scan mentions."""
    khasra: str
    register_row_id: str
    khewat: str


# ----------------------------------------------------------------------------- derived facts

@dataclass(frozen=True)
class ShareSum:
    total: Optional[Fraction]
    readable: bool
    n_owners: int


@dataclass(frozen=True)
class AreaTotal:
    written: int
    computed: int


@dataclass(frozen=True)
class OwnerMatch:
    scan_idx: int
    register_idx: int
    scan_name: str
    register_name: str
    similarity: float
    scan_share: Optional[Fraction]
    register_share: Optional[Fraction]


@dataclass(frozen=True)
class OwnerUnmatched:
    side: str
    idx: int
    name: str
    best_similarity: float


@dataclass(frozen=True)
class PlotPair:
    khasra: str
    scan_area: int
    register_area: int


@dataclass(frozen=True)
class KhasraElsewhere:
    khasra: str
    register_row_id: str
    khewat: str


@dataclass(frozen=True)
class CultivatorPair:
    scan: str
    register: str
    same: bool
    similarity: float


@dataclass(frozen=True)
class ScanMutation:
    mutation_no: str
    register_history: frozenset


@dataclass(frozen=True)
class VocabTerm:
    field: str
    value: str
    allowed: frozenset


# ----------------------------------------------------------------------------- KB


class KB:
    def __init__(self):
        self.facts: dict[type, set] = {}

    def add(self, fact) -> bool:
        bucket = self.facts.setdefault(type(fact), set())
        if fact in bucket:
            return False
        bucket.add(fact)
        return True

    def all(self, kind: type) -> list:
        return sorted(self.facts.get(kind, set()), key=repr)

    def has(self, kind: type) -> bool:
        return bool(self.facts.get(kind))

    def one(self, kind: type):
        items = self.all(kind)
        return items[0] if items else None

    def __len__(self) -> int:
        return sum(len(v) for v in self.facts.values())


def _share(s: Optional[str]) -> Optional[Fraction]:
    if not s:
        return None
    return N.parse_share(s)


def load_facts(kb: KB, h: Holding, row: Optional[RegisterRow], khasra_index: dict | None) -> None:
    kb.add(HoldingFact(h.holding_id, h.village_code, N.normalise_khewat(h.khewat_no),
                       row.register_row_id if row else None))
    for i, o in enumerate(h.owners):
        kb.add(ScanOwner(i, N.normalise_name(o.name), N.normalise_relation(o.relation or "") or o.relation,
                         N.normalise_name(o.parent or "") or None, _share(o.share)))
    for i, p in enumerate(h.plots):
        area = N.parse_area(p.area_raw) if p.area_raw else None
        kanal, marla = (area.kanal, area.marla) if area else (
            (p.area_marla // 20, p.area_marla % 20) if p.area_marla is not None else (None, None))
        kb.add(ScanPlot(i, N.parse_khasra(p.khasra or "") or "", kanal, marla, p.area_marla, p.land_type))
    if h.total_area_marla is not None:
        kb.add(WrittenTotal(h.total_area_marla))
    kb.add(Cultivator("scan", N.normalise_cultivator(h.cultivator)))
    if h.remarks.mutation_no:
        kb.add(Mutation(N.to_ascii_digits(h.remarks.mutation_no)))
    if row is None:
        return
    for i, o in enumerate(row.owners):
        kb.add(RegOwner(i, N.normalise_name(o.name), N.normalise_relation(o.relation or "") or o.relation,
                        N.normalise_name(o.parent or "") or None, _share(o.share)))
    for i, p in enumerate(row.plots):
        kb.add(RegPlot(i, N.parse_khasra(p.khasra or "") or "", p.area_marla))
    kb.add(Cultivator("register", N.normalise_cultivator(row.cultivator)))
    kb.add(RegisterHistory(frozenset(N.to_ascii_digits(m) for m in row.mutation_history)))
    if khasra_index:
        for p in h.plots:
            k = N.parse_khasra(p.khasra or "")
            if k and k in khasra_index:
                rid, khewat = khasra_index[k]
                kb.add(KhasraOwner(k, rid, khewat))


# ----------------------------------------------------------------------------- derivations


def d_share_sum(kb: KB, params: dict) -> Iterable:
    owners = kb.all(ScanOwner)
    shares = [o.share for o in owners]
    readable = bool(owners) and all(s is not None for s in shares)
    yield ShareSum(sum(shares, Fraction(0)) if readable else None, readable, len(owners))


def d_area_total(kb: KB, params: dict) -> Iterable:
    written = kb.one(WrittenTotal)
    areas = [p.area_marla for p in kb.all(ScanPlot)]
    if written and areas and all(a is not None for a in areas):
        yield AreaTotal(written.written, sum(areas))


def d_owner_matching(kb: KB, params: dict) -> Iterable:
    """Hungarian matching of scan and register owners on name similarity (R5, R6)."""
    scan, reg = kb.all(ScanOwner), kb.all(RegOwner)
    thr = params["name_match_threshold"]
    matched_s, matched_r = set(), set()
    best_s = {o.idx: 0.0 for o in scan}
    best_r = {o.idx: 0.0 for o in reg}
    if scan and reg:
        sim = np.array([[name_similarity(s.name, r.name) for r in reg] for s in scan])
        for i, s in enumerate(scan):
            best_s[s.idx] = float(sim[i].max())
        for j, r in enumerate(reg):
            best_r[r.idx] = float(sim[:, j].max())
        rows, cols = linear_sum_assignment(-sim)
        for i, j in zip(rows, cols):
            if sim[i, j] >= thr:
                s, r = scan[i], reg[j]
                matched_s.add(s.idx)
                matched_r.add(r.idx)
                yield OwnerMatch(s.idx, r.idx, s.name, r.name, round(float(sim[i, j]), 3), s.share, r.share)
    for s in scan:
        if s.idx not in matched_s:
            yield OwnerUnmatched("scan", s.idx, s.name, round(best_s[s.idx], 3))
    for r in reg:
        if r.idx not in matched_r:
            yield OwnerUnmatched("register", r.idx, r.name, round(best_r[r.idx], 3))


def d_plot_pairs(kb: KB, params: dict) -> Iterable:
    reg = {p.khasra: p for p in kb.all(RegPlot)}
    for p in kb.all(ScanPlot):
        r = reg.get(p.khasra)
        if r and p.area_marla is not None and r.area_marla is not None:
            yield PlotPair(p.khasra, p.area_marla, r.area_marla)


def d_khasra_elsewhere(kb: KB, params: dict) -> Iterable:
    h = kb.one(HoldingFact)
    own = {p.khasra for p in kb.all(RegPlot)}
    for ko in kb.all(KhasraOwner):
        if ko.khasra not in own and ko.register_row_id != h.linked_row:
            yield KhasraElsewhere(ko.khasra, ko.register_row_id, ko.khewat)


def d_cultivator(kb: KB, params: dict) -> Iterable:
    sides = {c.side: c.value for c in kb.all(Cultivator)}
    if "scan" not in sides or "register" not in sides:
        return
    a, b = sides["scan"], sides["register"]
    if a == b:
        same, sim = True, 1.0
    elif "khudkasht" in (a, b):
        same, sim = False, 0.0
    else:
        sim = name_similarity(a, b)
        same = sim >= params["name_match_threshold"]
    yield CultivatorPair(a, b, same, round(sim, 3))


def d_mutation(kb: KB, params: dict) -> Iterable:
    hist = kb.one(RegisterHistory)
    if hist is None:
        return
    for m in kb.all(Mutation):
        yield ScanMutation(m.mutation_no, hist.mutations)


def d_vocabulary(kb: KB, params: dict) -> Iterable:
    for p in kb.all(ScanPlot):
        yield VocabTerm("land_type", p.land_type or "", LAND_TYPES)
    for o in kb.all(ScanOwner):
        yield VocabTerm("relation", o.relation or "", RELATIONS)


# (name, fact types that must be present, function). Cross-record derivations need a linked row.
DERIVATIONS: list[tuple[str, tuple[type, ...], Callable]] = [
    ("share_sum", (HoldingFact,), d_share_sum),
    ("area_total", (WrittenTotal, ScanPlot), d_area_total),
    ("vocabulary", (HoldingFact,), d_vocabulary),
    ("owner_matching", (RegisterHistory,), d_owner_matching),
    ("plot_pairs", (ScanPlot, RegPlot), d_plot_pairs),
    ("khasra_elsewhere", (KhasraOwner, RegisterHistory), d_khasra_elsewhere),
    ("cultivator", (RegisterHistory,), d_cultivator),
    ("mutation", (Mutation, RegisterHistory), d_mutation),
]


def forward_chain(kb: KB, params: dict, max_rounds: int = 10) -> int:
    """Run derivations until no new fact is added. Returns the number of rounds."""
    for rnd in range(1, max_rounds + 1):
        added = False
        for _, needs, fn in DERIVATIONS:
            if all(kb.has(t) for t in needs):
                for fact in fn(kb, params):
                    added |= kb.add(fact)
        if not added:
            return rnd
    return max_rounds


FACT_TYPES = {cls.__name__: cls for cls in [
    HoldingFact, ScanOwner, RegOwner, ScanPlot, RegPlot, WrittenTotal, Cultivator, Mutation, RegisterHistory,
    KhasraOwner, ShareSum, AreaTotal, OwnerMatch, OwnerUnmatched, PlotPair, KhasraElsewhere, CultivatorPair,
    ScanMutation, VocabTerm]}
