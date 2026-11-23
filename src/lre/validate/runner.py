"""Link and validate a batch of holdings, and write the contradiction report (Objective 2)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ..config import LinkCfg
from ..link.linker import build_index, link
from ..schema import Holding, RegisterRow
from .engine import RuleSet, is_contradiction, validate


def validate_holdings(holdings: list[Holding], rows: list[RegisterRow], rules: RuleSet,
                      link_cfg: LinkCfg) -> list[Holding]:
    """Sets ``link`` and ``flags`` on every holding (in place) and returns them."""
    by_village: dict[str, list[RegisterRow]] = defaultdict(list)
    for r in rows:
        by_village[r.village_code].append(r)
    indexes = {v: build_index(rs) for v, rs in by_village.items()}
    owners = {v: idx.khasra_owner for v, idx in indexes.items()}
    for h in holdings:
        idx = indexes.get(h.village_code)
        if idx is None:
            h.link, h.flags = None, validate(h, None, rules)
            continue
        h.link = link(h, idx, link_cfg)
        row = idx.rows.get(h.link.register_row_id) if h.link.register_row_id else None
        h.flags = validate(h, row, rules, owners[h.village_code])
    return holdings


REPORT_COLUMNS = ["holding_id", "page_id", "village_code", "khewat_no", "register_row_id", "link_method",
                  "rule", "name", "kind", "severity", "score", "evidence"]


def write_contradiction_report(holdings: list[Holding], rules: RuleSet, path: str | Path) -> int:
    """One row per flag on every holding that is a contradiction. Returns the number of flagged holdings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        for h in holdings:
            if not is_contradiction(h.flags, rules):
                continue
            n += 1
            for f in h.flags:
                w.writerow({"holding_id": h.holding_id, "page_id": h.page_id, "village_code": h.village_code,
                            "khewat_no": h.khewat_no,
                            "register_row_id": h.link.register_row_id if h.link else "",
                            "link_method": h.link.method if h.link else "", "rule": f.rule, "name": f.name,
                            "kind": f.kind, "severity": f.severity,
                            "score": "" if f.score is None else f"{f.score:.3f}",
                            "evidence": json.dumps(f.evidence, ensure_ascii=False)})
    return n
