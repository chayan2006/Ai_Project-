"""Sample villages of fictitious holdings (report D.6 step 1) and the text written for them."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..config import SynthCfg
from ..extract.normalise import format_share, to_dev_digits
from ..schema import Holding, Owner, Plot, Remarks
from . import vocab as V


@dataclass
class Village:
    code: str
    name: str
    hadbast: str
    register_year: str
    digit_style: str                 # "ascii" or "dev": how this village's Patwari writes digits
    khasra_style: str                # "rect" (rect//killa) or "plain" (n or n/m)
    holdings: list[Holding] = field(default_factory=list)          # true values
    mutation_history: dict[str, list[str]] = field(default_factory=dict)  # khewat -> older + current
    raw_extra: dict[str, dict[int, list[str]]] = field(default_factory=dict)  # khewat -> col -> text


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _person(rng: np.random.Generator, female: bool) -> str:
    if female:
        return f"{_pick(rng, V.FEMALE_GIVEN)} {_pick(rng, V.FEMALE_SECOND)}"
    return f"{_pick(rng, V.MALE_GIVEN)} {_pick(rng, V.MALE_SECOND)}"


def _shares(rng: np.random.Generator, k: int) -> list[Fraction]:
    if k == 1:
        return [Fraction(1)]
    if rng.random() < 0.6:
        return [Fraction(1, k)] * k
    dens = [d for d in (2, 3, 4, 6, 8, 12) if d >= k]
    d = int(_pick(rng, dens))
    # random composition of d into k positive parts
    cuts = sorted(rng.choice(np.arange(1, d), size=k - 1, replace=False).tolist())
    parts = [b - a for a, b in zip([0] + cuts, cuts + [d])]
    return [Fraction(p, d) for p in parts]


def _owners(rng: np.random.Generator, k: int) -> list[Owner]:
    siblings = k > 1 and rng.random() < 0.6
    common_parent = _person(rng, female=False)
    shares = _shares(rng, k)
    owners, used = [], set()
    for i in range(k):
        female = rng.random() < 0.25
        while True:
            name = _person(rng, female)
            if name not in used:
                used.add(name)
                break
        if female:
            relation = "d/o" if siblings or rng.random() < 0.5 else "w/o"
        else:
            relation = "s/o"
        # siblings share a father; a wife's 'parent' field holds her husband's name
        parent = common_parent if siblings and relation in ("s/o", "d/o") else _person(rng, female=False)
        owners.append(Owner(name=name, relation=relation, parent=parent, share=format_share(shares[i])))
    return owners


def _land_type(rng: np.random.Generator) -> str:
    keys = list(V.LAND_TYPE_WEIGHTS)
    p = np.array([V.LAND_TYPE_WEIGHTS[k] for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _area_marla(rng: np.random.Generator) -> int:
    kanal = int(rng.integers(0, 16))
    marla = int(rng.integers(0, 20))
    return max(1, kanal * 20 + marla)


def sample_village(rng: np.random.Generator, cfg: SynthCfg, code: str) -> Village:
    lo, hi = cfg.khewats_per_village
    n = int(rng.integers(lo, hi + 1))
    start_year = int(rng.integers(1960, 2005))
    village = Village(
        code=code,
        name=f"{_pick(rng, V.VILLAGE_PARTS_A)}{_pick(rng, V.VILLAGE_PARTS_B)}",
        hadbast=str(int(rng.integers(1, 400))),
        register_year=f"{start_year}-{(start_year + 1) % 100:02d}",
        digit_style="ascii" if rng.random() < cfg.ascii_digit_prob else "dev",
        khasra_style="rect" if rng.random() < 0.7 else "plain",
    )
    used_khasra: set[str] = set()
    next_rect, next_plain = 1, 1
    next_khatauni = 1
    mutation_pool = rng.choice(np.arange(1000, 9999), size=3 * n, replace=False).tolist()
    olo, ohi = cfg.owners_per_holding
    plo, phi = cfg.khasras_per_holding

    for kh in range(1, n + 1):
        khewat = str(kh)
        if rng.random() < 0.15:
            khatauni = f"{next_khatauni}-{next_khatauni + 1}"
            next_khatauni += 2
        else:
            khatauni = str(next_khatauni)
            next_khatauni += 1

        owners = _owners(rng, int(rng.integers(olo, ohi + 1)))

        plots = []
        for _ in range(int(rng.integers(plo, phi + 1))):
            while True:
                if village.khasra_style == "rect":
                    if rng.random() < 0.35:
                        next_rect += 1
                    khasra = f"{next_rect}//{int(rng.integers(1, 26))}"
                else:
                    if rng.random() < 0.3:
                        khasra = f"{next_plain}/{int(rng.integers(1, 4))}"
                    else:
                        khasra = str(next_plain)
                    next_plain += 1
                if khasra not in used_khasra:
                    used_khasra.add(khasra)
                    break
            plots.append(Plot(khasra=khasra, area_marla=_area_marla(rng), land_type=_land_type(rng)))
        total = sum(p.area_marla for p in plots) if len(plots) >= 2 and rng.random() < 0.5 else None

        if rng.random() < 0.7:
            cultivator = "khudkasht"
        else:
            cultivator = f"{_person(rng, False)} s/o {_person(rng, False)}"

        history = sorted(mutation_pool.pop() for _ in range(int(rng.integers(0, 3))))
        remarks = Remarks()
        if rng.random() < 0.35:
            mut = str(mutation_pool.pop())
            remarks.mutation_no = mut
            history.append(int(mut))
            if rng.random() < 0.85:
                y = start_year + int(rng.integers(0, 6))
                d = dt.date(y, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
                remarks.date = d.isoformat()
        village.mutation_history[khewat] = [str(m) for m in history]

        h = Holding(
            holding_id="", village_code=code, khewat_no=khewat, khatauni_no=khatauni,
            patti=_pick(rng, V.PATTI_NAMES), owners=owners, cultivator=cultivator,
            irrigation=V.IRRIGATION_OF.get(plots[0].land_type, "बारानी"),
            plots=plots, total_area_marla=total, remarks=remarks,
        )
        village.holdings.append(h)
        village.raw_extra[khewat] = {
            8: [f"{rng.integers(1, 60)}.{_pick(rng, ['00', '25', '50', '75'])}"] if rng.random() < 0.5 else [],
            9: [f"{rng.integers(0, 6)}.{_pick(rng, ['00', '25', '50'])}"] if rng.random() < 0.2 else [],
            10: [f"{rng.integers(5, 90)}.00"] if cultivator != "khudkasht" and rng.random() < 0.6 else [],
        }
    return village


# ----------------------------------------------------------------------------- page text


def _digits(s: str, style: str) -> str:
    return to_dev_digits(s) if style == "dev" else s


def _area_text(marla: int) -> str:
    return f"{marla // 20}-{marla % 20:02d}"


def page_texts(h: Holding, village: Village, rng: np.random.Generator) -> dict[int, list[str]]:
    """Logical text segments written in each column for one holding (before line wrapping)."""
    ds = village.digit_style
    d = lambda s: _digits(s, ds)  # noqa: E731
    texts: dict[int, list[str]] = {c: [] for c in range(12)}
    texts[0] = [d(h.khewat_no)]
    texts[1] = [d(h.khatauni_no)]
    texts[2] = [h.patti]
    for o in h.owners:
        rel = _pick(rng, V.PAGE_RELATIONS[o.relation]) if o.relation else ""
        who = f"{o.name} {rel} {o.parent}" if o.relation else o.name
        texts[3].append(f"{who} {V.SHARE_WORD} {d(o.share)}")
    if h.cultivator == "khudkasht":
        texts[4] = [V.KHUDKASHT]
    else:
        name, parent = h.cultivator.split(" s/o ")
        texts[4] = [f"{name} {_pick(rng, V.PAGE_RELATIONS['s/o'])} {parent} {V.TENANT_TAG}"]
    texts[5] = [h.irrigation]
    texts[6] = [d(p.khasra) for p in h.plots]
    texts[7] = [f"{d(_area_text(p.area_marla))} {V.LAND_TYPE_WORDS[p.land_type]}" for p in h.plots]
    if h.total_area_marla is not None:
        texts[7].append(f"{V.TOTAL_WORD} {d(_area_text(h.total_area_marla))}")
    for c, vals in village.raw_extra.get(h.khewat_no, {}).items():
        texts[c] = [d(v) for v in vals]
    if h.remarks.mutation_no:
        texts[11] = [f"{V.MUTATION_PREFIX} {d(h.remarks.mutation_no)}"]
        if h.remarks.date:
            y, m, dd = h.remarks.date.split("-")
            texts[11].append(f"{V.DATE_PREFIX} {d(f'{dd}-{m}-{y}')}")
        else:
            texts[11].append(V.PENDING_WORD)
    return texts


def remarks_raw(texts: dict[int, list[str]]) -> str:
    return " ".join(texts.get(11, []))
