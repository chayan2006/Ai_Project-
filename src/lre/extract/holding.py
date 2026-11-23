"""Build structured holdings from per-cell text (report 5.1 step 5, Table 4).

The input is one text per column (lines joined with newlines) plus an optional
confidence per column. The owner parser is pluggable: the template baseline uses
the regular-expression parser, the proposed system will pass the CRF tagger.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..schema import COLUMN_FIELDS, FieldType, Holding, Owner, Plot, Remarks
from . import normalise as N

OwnerParser = Callable[[str], list[Owner]]


def regex_owner_parser(text: str) -> list[Owner]:
    owners = []
    for entry in N.split_owner_entries(text):
        o = N.parse_owner_line(entry)
        if o:
            owners.append(o)
    return owners


def parse_plots(khasra_text: str, area_text: str) -> tuple[list[Plot], Optional[int]]:
    """Pair khasra numbers with area lines in order; a 'कुल' line is the written total."""
    khasras = N.parse_khasra_list(khasra_text)
    areas, total = [], None
    for line in (area_text or "").splitlines():
        line = N.clean(line)
        if not line:
            continue
        a = N.parse_area(line)
        if N.clean(line).startswith(("कुल", "total")):
            total = a.total_marla if a else None
            continue
        areas.append((a, N.normalise_land_type(line), line))
    plots = []
    for i in range(max(len(khasras), len(areas))):
        k = khasras[i] if i < len(khasras) else ""
        a, lt, raw = areas[i] if i < len(areas) else (None, None, "")
        plots.append(Plot(khasra=k, area_marla=a.total_marla if a else None, land_type=lt,
                          area_raw=str(a) if a else (raw or None)))
    return plots, total


def build_holding(texts: dict[int, str], holding_id: str, village_code: str, page_id: str | None = None,
                  row_index: int | None = None, confs: dict[int, Optional[float]] | None = None,
                  owner_parser: OwnerParser = regex_owner_parser) -> Holding:
    t = {c: texts.get(c, "") or "" for c in range(len(COLUMN_FIELDS))}
    confs = confs or {}
    owners = owner_parser(t[3])
    for o in owners:
        o.conf = confs.get(3)
    plots, total = parse_plots(t[6], t[7])
    plot_confs = [c for c in (confs.get(6), confs.get(7)) if c is not None]
    for p in plots:
        p.conf = min(plot_confs) if plot_confs else None
    mut, date = N.parse_mutation(t[11])
    ks = N.parse_khatauni(t[1])
    khatauni = f"{ks[0]}-{ks[-1]}" if len(ks) > 1 else (ks[0] if ks else "")
    return Holding(
        holding_id=holding_id, village_code=village_code, page_id=page_id, row_index=row_index,
        khewat_no=N.normalise_khewat(t[0]), khatauni_no=khatauni,
        patti=N.clean(t[2]), owners=owners, cultivator=N.normalise_cultivator(t[4]),
        irrigation=N.clean(t[5]), plots=plots, total_area_marla=total,
        remarks=Remarks(raw=N.clean(t[11]), mutation_no=mut, date=date, conf=confs.get(11)),
    )


FIELD_OF_COLUMN = {c: f.value for c, f in enumerate(COLUMN_FIELDS)}
EXTRACTED_COLUMNS = [0, 1, 3, 4, 6, 7, 11]   # columns whose values are extracted and evaluated
REQUIRED_COLUMNS = [0, 3, 6, 7]              # a holding without these cannot be auto-accepted
assert all(FIELD_OF_COLUMN[c] in {f.value for f in FieldType} for c in EXTRACTED_COLUMNS)
