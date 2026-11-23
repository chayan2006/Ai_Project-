"""Build the digital register from the true holdings and inject labelled changes (report D.6 step 4).

Three kinds of change are injected, and every one is logged as a Label:

* contradiction - the digital register genuinely disagrees with the legacy page (R5-R10);
* intra         - the legacy page itself is inconsistent (R1, R2);
* benign        - a harmless variant that must NOT be flagged (spelling, digit script, relation word);
* renumber      - the khewat number changed between registers (tests the linker fallback).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction

import numpy as np

from ..config import SynthCfg
from ..extract.normalise import format_share, to_dev_digits
from ..schema import Holding, RegisterRow
from ..validate.similarity import name_similarity
from . import vocab as V
from .sample import Village, _person, _pick

# A replacement owner must be clearly a different person, and a spelling variant
# clearly the same one, under the similarity the validation engine uses.
DIFFERENT_PERSON_MAX_SIM = 0.6
SAME_PERSON_MIN_SIM = 0.85

CONTRADICTION_WEIGHTS = {
    "owner_replaced": 0.20, "owner_removed": 0.15, "share_changed": 0.20, "area_changed": 0.20,
    "khasra_moved": 0.10, "mutation_missing": 0.10, "cultivator_changed": 0.05,
}
CONTRADICTION_RULES = {
    "owner_replaced": "R5", "owner_removed": "R5", "share_changed": "R6", "area_changed": "R7",
    "khasra_moved": "R8", "mutation_missing": "R10", "cultivator_changed": "R9",
}


@dataclass
class Label:
    village_code: str
    khewat_no: str
    kind: str          # contradiction | intra | benign | renumber
    type: str
    rules: str         # expected rule ids, ';'-separated; empty for benign / renumber
    field: str
    page_value: str
    register_value: str
    holding_id: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def _row_from(h: Holding, history: list[str]) -> RegisterRow:
    owners = [replace(o, conf=None) for o in h.owners]
    plots = [replace(p, conf=None) for p in h.plots]
    return RegisterRow(
        register_row_id=f"{h.village_code}-{h.khewat_no}", village_code=h.village_code,
        khewat_no=h.khewat_no, khatauni_no=h.khatauni_no, owners=owners, cultivator=h.cultivator,
        plots=plots, mutation_history=list(history), last_mutation_ref=history[-1] if history else None,
    )


# ----------------------------------------------------------------------------- page (intra) errors


def apply_page_error(h: Holding, rng: np.random.Generator) -> Label | None:
    """Make the legacy page itself inconsistent. Mutates and returns a label, or None if not applicable."""
    options = []
    if h.total_area_marla is not None:
        options.append("total_mismatch")
    if len(h.owners) >= 2:
        options.append("share_sum")
    if not options:
        return None
    kind = _pick(rng, options)
    if kind == "total_mismatch":
        old = h.total_area_marla
        delta = int(rng.integers(3, 41)) * (1 if rng.random() < 0.5 else -1)
        h.total_area_marla = max(1, old + delta)
        return Label(h.village_code, h.khewat_no, "intra", "total_mismatch", "R2", "total_area",
                     str(h.total_area_marla), str(sum(p.area_marla for p in h.plots)))
    o = h.owners[int(rng.integers(len(h.owners)))]
    old = Fraction(o.share)
    new = old + Fraction(1, max(old.denominator, 2) * 2)
    o.share = format_share(new)
    return Label(h.village_code, h.khewat_no, "intra", "share_sum", "R1;R6", "share", o.share, format_share(old))


# ----------------------------------------------------------------------------- contradictions


def _different_person(rng, like: str, others: list[str], female: bool) -> str:
    for _ in range(200):
        cand = _person(rng, female)
        if all(name_similarity(cand, x) < DIFFERENT_PERSON_MAX_SIM for x in [like, *others]):
            return cand
    raise RuntimeError("could not sample a clearly different name")


def _is_female(name: str) -> bool:
    return name.split()[-1] in V.FEMALE_SECOND


def _applicable(h: Holding, row: RegisterRow, n_rows: int) -> list[str]:
    out = ["owner_replaced", "area_changed", "cultivator_changed"]
    if len(row.owners) >= 2:
        out += ["owner_removed", "share_changed"]
    if len(row.plots) >= 2 and n_rows > 1:
        out.append("khasra_moved")
    if h.remarks.mutation_no and h.remarks.mutation_no in row.mutation_history:
        out.append("mutation_missing")
    return out


def _inject_contradiction(kind: str, h: Holding, row: RegisterRow, rows: list[RegisterRow],
                          rng: np.random.Generator) -> Label:
    lab = lambda field, pv, rv: Label(h.village_code, h.khewat_no, "contradiction", kind,  # noqa: E731
                                      CONTRADICTION_RULES[kind], field, str(pv), str(rv))
    if kind == "owner_replaced":
        i = int(rng.integers(len(row.owners)))
        old = row.owners[i].name
        new = _different_person(rng, old, [o.name for o in row.owners], _is_female(old))
        row.owners[i].name = new
        return lab("owner.name", old, new)
    if kind == "owner_removed":
        i, j = rng.choice(len(row.owners), size=2, replace=False).tolist()
        removed = row.owners[i]
        row.owners[j].share = format_share(Fraction(row.owners[j].share) + Fraction(removed.share))
        del row.owners[i]
        return lab("owner", removed.name, "")
    if kind == "share_changed":
        i, j = rng.choice(len(row.owners), size=2, replace=False).tolist()
        si, sj = Fraction(row.owners[i].share), Fraction(row.owners[j].share)
        total = si + sj
        candidates = [total * f for f in (Fraction(1, 3), Fraction(2, 3), Fraction(1, 4), Fraction(3, 4),
                                          Fraction(1, 5), Fraction(4, 5))]
        valid = [x for x in candidates if x != sj and total - x != si]
        x = valid[int(rng.integers(len(valid)))]
        row.owners[i].share, row.owners[j].share = format_share(total - x), format_share(x)
        return lab("share", format_share(si), row.owners[i].share)
    if kind == "area_changed":
        p = row.plots[int(rng.integers(len(row.plots)))]
        old = p.area_marla
        tol = max(1, 0.02 * old)
        new = max(1, round(old * (1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.5))))
        if abs(new - old) <= tol + 1:
            new = old + int(np.ceil(tol)) + int(rng.integers(2, 20))
        p.area_marla = int(new)
        return lab(f"area[{p.khasra}]", old, new)
    if kind == "khasra_moved":
        p = row.plots.pop(int(rng.integers(len(row.plots))))
        target = row
        while target is row:
            target = rows[int(rng.integers(len(rows)))]
        target.plots.append(p)
        return lab("khasra", f"{p.khasra}@{row.khewat_no}", f"{p.khasra}@{target.khewat_no}")
    if kind == "mutation_missing":
        row.mutation_history = [m for m in row.mutation_history if m != h.remarks.mutation_no]
        row.last_mutation_ref = row.mutation_history[-1] if row.mutation_history else None
        return lab("mutation_no", h.remarks.mutation_no, "")
    if kind == "cultivator_changed":
        old = row.cultivator
        if old == "khudkasht":
            row.cultivator = f"{_person(rng, False)} s/o {_person(rng, False)}"
        elif rng.random() < 0.5:
            row.cultivator = "khudkasht"
        else:
            tenant = old.split(" s/o ")[0]
            row.cultivator = f"{_different_person(rng, tenant, [], False)} s/o {_person(rng, False)}"
        return lab("cultivator", old, row.cultivator)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- benign variants


def _spelling_variant(name: str, rng: np.random.Generator) -> str | None:
    swaps = [s for s in V.SPELLING_SWAPS if s[0] in name]
    for idx in rng.permutation(len(swaps)).tolist():
        a, b = swaps[idx]
        positions = [i for i in range(len(name)) if name.startswith(a, i)]
        pos = positions[int(rng.integers(len(positions)))]
        cand = name[:pos] + b + name[pos + len(a):]
        if cand != name and name_similarity(cand, name) >= SAME_PERSON_MIN_SIM:
            return cand
    return None


def _inject_benign(h: Holding, row: RegisterRow, rng: np.random.Generator) -> Label | None:
    kinds = ["spelling", "digits", "relation_word"]
    for kind in rng.permutation(kinds).tolist():
        lab = lambda field, pv, rv: Label(h.village_code, h.khewat_no, "benign", kind, "",  # noqa: E731
                                          field, str(pv), str(rv))
        if kind == "spelling":
            o = row.owners[int(rng.integers(len(row.owners)))]
            attr = "name" if rng.random() < 0.6 or not o.parent else "parent"
            old = getattr(o, attr)
            new = _spelling_variant(old, rng)
            if new:
                setattr(o, attr, new)
                return lab(f"owner.{attr}", old, new)
        elif kind == "digits":
            if rng.random() < 0.5:
                old = row.khewat_no
                row.khewat_no = to_dev_digits(old)
                return lab("khewat_no", old, row.khewat_no)
            p = row.plots[int(rng.integers(len(row.plots)))]
            old = p.khasra
            p.khasra = to_dev_digits(old)
            return lab("khasra", old, p.khasra)
        elif kind == "relation_word":
            o = row.owners[int(rng.integers(len(row.owners)))]
            if o.relation:
                old = o.relation
                o.relation = V.PAGE_RELATIONS[old][0]
                return lab("owner.relation", old, o.relation)
    return None


# ----------------------------------------------------------------------------- village


def build_register(village: Village, page_holdings: dict[str, Holding], cfg: SynthCfg,
                   rng: np.random.Generator) -> tuple[list[RegisterRow], list[Label]]:
    """Register rows for every khewat in the village, with labelled perturbations.

    ``page_holdings`` maps khewat -> the holding as written on the legacy page (after any page errors).
    """
    rows = [_row_from(h, village.mutation_history.get(h.khewat_no, [])) for h in village.holdings]
    labels: list[Label] = []
    moved: set[str] = set()
    w = CONTRADICTION_WEIGHTS
    for h, row in zip(village.holdings, rows):
        page_h = page_holdings.get(h.khewat_no, h)
        if rng.random() < cfg.contradiction_rate:
            kinds = _applicable(h, row, len(rows))
            p = np.array([w[k] for k in kinds])
            kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
            labels.append(_inject_contradiction(kind, page_h, row, rows, rng))
            if kind == "khasra_moved":
                moved.add(h.khewat_no)
        if rng.random() < cfg.benign_rate:
            lab = _inject_benign(page_h, row, rng)
            if lab:
                labels.append(lab)
    # Renumber some khewats to unused numbers so the linker must fall back to khasra overlap.
    next_free = max(int(h.khewat_no) for h in village.holdings) + 1
    for h, row in zip(village.holdings, rows):
        if h.khewat_no not in moved and rng.random() < cfg.renumber_rate:
            new = str(next_free)
            next_free += 1
            labels.append(Label(h.village_code, h.khewat_no, "renumber", "khewat_renumbered", "",
                                "khewat_no", h.khewat_no, new))
            row.khewat_no = new
            row.register_row_id = f"{village.code}-{new}"
    return rows, labels
