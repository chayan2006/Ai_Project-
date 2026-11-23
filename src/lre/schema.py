"""Data contracts passed between pipeline stages (report D.5, E.5).

Every stage is a pure function from one of these types to the next, and every
type round-trips through JSON so that per-stage checkpoints can be written to
data/interim/<page_id>/ and inspected or re-run on their own.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Optional


class Route(str, Enum):
    AUTO = "AUTO"
    REVIEW = "REVIEW"
    CONTRADICTION = "CONTRADICTION"


class FieldType(str, Enum):
    """Field carried by each of the 12 Jamabandi columns (report Figure 1, Table 4)."""
    KHEWAT = "khewat_no"
    KHATAUNI = "khatauni_no"
    PATTI = "patti"
    OWNERS = "owners"
    CULTIVATOR = "cultivator"
    IRRIGATION = "irrigation"
    KHASRA = "khasra_nos"
    AREA = "area"
    REVENUE = "revenue"
    CESS = "cess"
    RENT = "rent"
    REMARKS = "remarks"


# Column index (0-based) -> field type. The column index of a cell gives its field type.
COLUMN_FIELDS: list[FieldType] = [
    FieldType.KHEWAT, FieldType.KHATAUNI, FieldType.PATTI, FieldType.OWNERS,
    FieldType.CULTIVATOR, FieldType.IRRIGATION, FieldType.KHASRA, FieldType.AREA,
    FieldType.REVENUE, FieldType.CESS, FieldType.RENT, FieldType.REMARKS,
]


BBox = tuple[int, int, int, int]  # x, y, w, h in page pixels


@dataclass
class FieldValue:
    field: str
    raw_text: str                       # unconstrained decoder output
    text: str                           # constrained / normalised text
    conf_raw: Optional[float] = None
    conf_cal: Optional[float] = None
    margin: Optional[float] = None
    bbox: Optional[BBox] = None
    row: Optional[int] = None
    col: Optional[int] = None
    model_version: Optional[str] = None


@dataclass
class Owner:
    name: str
    relation: Optional[str] = None      # one of s/o, d/o, w/o
    parent: Optional[str] = None
    share: str = "1/1"                  # exact fraction as text, e.g. "1/2"
    conf: Optional[float] = None

    @property
    def share_fraction(self) -> Optional[Fraction]:
        try:
            return Fraction(self.share)
        except (ValueError, ZeroDivisionError, TypeError):
            return None


@dataclass
class Plot:
    khasra: str                         # normalised, e.g. "112//3"
    area_marla: Optional[int] = None    # kanal-marla converted to marla (1 kanal = 20 marla)
    land_type: Optional[str] = None     # code from the land-type vocabulary
    conf: Optional[float] = None
    area_raw: Optional[str] = None      # kanal-marla as read, e.g. "6-23", so R3 can see invalid units


@dataclass
class Remarks:
    raw: str = ""
    mutation_no: Optional[str] = None
    date: Optional[str] = None          # ISO yyyy-mm-dd
    conf: Optional[float] = None


@dataclass
class LinkResult:
    register_row_id: Optional[str]
    method: str                         # khewat | khasra_jaccard | unlinked
    score: float = 0.0


@dataclass
class Flag:
    rule: str                           # R1..R10
    name: str
    kind: str                           # intra | cross
    severity: str                       # high | medium | low
    holding_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None       # P(true contradiction) from the scorer


@dataclass
class Holding:
    holding_id: str
    village_code: str
    khewat_no: str
    khatauni_no: str = ""
    page_id: Optional[str] = None
    row_index: Optional[int] = None
    patti: str = ""
    owners: list[Owner] = field(default_factory=list)
    cultivator: str = ""
    irrigation: str = ""
    plots: list[Plot] = field(default_factory=list)
    total_area_marla: Optional[int] = None  # only when a total is written on the page (R2)
    remarks: Remarks = field(default_factory=Remarks)
    link: Optional[LinkResult] = None
    flags: list[Flag] = field(default_factory=list)
    route: Optional[Route] = None
    models: dict[str, str] = field(default_factory=dict)


@dataclass
class RegisterRow:
    """One row of the existing digital Record of Rights (report Figure 9)."""
    register_row_id: str
    village_code: str
    khewat_no: str
    khatauni_no: str = ""
    owners: list[Owner] = field(default_factory=list)
    cultivator: str = ""
    plots: list[Plot] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    last_mutation_ref: Optional[str] = None


@dataclass
class Cell:
    page_id: str
    row: int
    col: int
    field_type: str
    bbox: BBox
    crop: Any = None                    # numpy array; never serialised


# --------------------------------------------------------------------------- JSON


def to_dict(obj: Any) -> Any:
    """Dataclass -> plain JSON-ready structure (enums become their values, crops are dropped)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name == "crop":
                continue
            out[f.name] = to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _owner(d: dict) -> Owner:
    return Owner(**d)


def _plot(d: dict) -> Plot:
    return Plot(**d)


def holding_from_dict(d: dict) -> Holding:
    d = dict(d)
    d["owners"] = [_owner(o) for o in d.get("owners", [])]
    d["plots"] = [_plot(p) for p in d.get("plots", [])]
    d["remarks"] = Remarks(**d.get("remarks", {}))
    if d.get("link"):
        d["link"] = LinkResult(**d["link"])
    d["flags"] = [Flag(**f) for f in d.get("flags", [])]
    if d.get("route"):
        d["route"] = Route(d["route"])
    return Holding(**d)


def register_row_from_dict(d: dict) -> RegisterRow:
    d = dict(d)
    d["owners"] = [_owner(o) for o in d.get("owners", [])]
    d["plots"] = [_plot(p) for p in d.get("plots", [])]
    return RegisterRow(**d)
