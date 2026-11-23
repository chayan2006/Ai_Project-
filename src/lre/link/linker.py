"""Link an extracted holding to its row in the digital register (report 5.3.5).

1. Block on (village, khewat number).
2. Khewat numbers change between Jamabandis, so if no row matches, take the
   register row with the highest Jaccard overlap of khasra numbers (>= 0.5).
   Khasra numbers identify parcels, so a strong overlap (>= 0.8) links on its own;
   a weaker one must be confirmed by owner-name similarity, in the spirit of
   Fellegi-Sunter matching. (A changed owner on the same parcels is a contradiction
   to flag, not a reason to refuse the link.)
3. A khewat match that shares no khasra with the holding while another row has
   a strong khasra match is treated as a misread khewat number.
4. Otherwise the holding is UNLINKED and goes to review.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import LinkCfg
from ..extract import normalise as N
from ..schema import Holding, LinkResult, RegisterRow
from ..validate.similarity import name_similarity


@dataclass
class RegisterIndex:
    rows: dict[str, RegisterRow]                     # register_row_id -> row
    by_khewat: dict[str, str]                        # normalised khewat -> register_row_id
    by_khasra: dict[str, list[str]]                  # normalised khasra -> register_row_ids
    khasras_of: dict[str, set[str]]

    @property
    def khasra_owner(self) -> dict[str, tuple[str, str]]:
        """khasra -> (register_row_id, khewat), for rule R8."""
        out = {}
        for k, rids in self.by_khasra.items():
            out[k] = (rids[0], N.normalise_khewat(self.rows[rids[0]].khewat_no))
        return out


def build_index(rows: list[RegisterRow]) -> RegisterIndex:
    by_id = {r.register_row_id: r for r in rows}
    by_khewat = {N.normalise_khewat(r.khewat_no): r.register_row_id for r in rows}
    by_khasra: dict[str, list[str]] = defaultdict(list)
    khasras_of = {}
    for r in rows:
        ks = {N.parse_khasra(p.khasra or "") for p in r.plots} - {None, ""}
        khasras_of[r.register_row_id] = ks
        for k in ks:
            by_khasra[k].append(r.register_row_id)
    return RegisterIndex(by_id, by_khewat, dict(by_khasra), khasras_of)


def _owner_similarity(h: Holding, row: RegisterRow) -> float:
    if not h.owners or not row.owners:
        return 0.0
    return float(np.mean([max(name_similarity(o.name, r.name) for r in row.owners) for o in h.owners]))


def _best_khasra_row(h: Holding, idx: RegisterIndex) -> tuple[Optional[str], float]:
    scan = {N.parse_khasra(p.khasra or "") for p in h.plots} - {None, ""}
    if not scan:
        return None, 0.0
    candidates = {rid for k in scan for rid in idx.by_khasra.get(k, [])}
    best, best_j = None, 0.0
    for rid in sorted(candidates):
        reg = idx.khasras_of[rid]
        j = len(scan & reg) / len(scan | reg)
        if j > best_j:
            best, best_j = rid, j
    return best, best_j


def link(h: Holding, idx: RegisterIndex, cfg: LinkCfg) -> LinkResult:
    rid = idx.by_khewat.get(N.normalise_khewat(h.khewat_no)) if h.khewat_no else None
    alt, j = _best_khasra_row(h, idx)
    if rid is not None:
        scan = {N.parse_khasra(p.khasra or "") for p in h.plots} - {None, ""}
        shares_khasra = bool(scan & idx.khasras_of.get(rid, set()))
        if shares_khasra or not scan or alt is None or alt == rid or j < cfg.min_khasra_jaccard:
            return LinkResult(rid, "khewat", 1.0)
    if alt is not None and j >= cfg.min_khasra_jaccard and (
            j >= cfg.strong_khasra_jaccard or _owner_similarity(h, idx.rows[alt]) >= cfg.min_owner_similarity):
        return LinkResult(alt, "khasra_jaccard", round(j, 3))
    return LinkResult(None, "unlinked", round(j, 3))
