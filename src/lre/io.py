"""File helpers: hashing, JSON checkpoints and the digital-register CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema import RegisterRow, register_row_from_dict, to_dict

REGISTER_COLUMNS = [
    "register_row_id", "village_code", "khewat_no", "khatauni_no",
    "owners_json", "cultivator", "plots_json", "mutation_history_json", "last_mutation_ref",
]


class RegisterSchemaError(ValueError):
    """Register CSV does not match the expected schema (report E.11: stop before processing)."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def register_rows_to_frame(rows: Iterable[RegisterRow]) -> pd.DataFrame:
    records = []
    for r in rows:
        records.append({
            "register_row_id": r.register_row_id,
            "village_code": r.village_code,
            "khewat_no": r.khewat_no,
            "khatauni_no": r.khatauni_no,
            "owners_json": json.dumps(to_dict(r.owners), ensure_ascii=False),
            "cultivator": r.cultivator,
            "plots_json": json.dumps(to_dict(r.plots), ensure_ascii=False),
            "mutation_history_json": json.dumps(r.mutation_history, ensure_ascii=False),
            "last_mutation_ref": r.last_mutation_ref or "",
        })
    return pd.DataFrame.from_records(records, columns=REGISTER_COLUMNS)


def save_register(rows: Iterable[RegisterRow], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    register_rows_to_frame(rows).to_csv(p, index=False, encoding="utf-8")


def load_register(path: str | Path, village: str | None = None) -> list[RegisterRow]:
    """Load and validate the digital register; optionally keep one village only."""
    p = Path(path)
    if not p.exists():
        raise RegisterSchemaError(f"register file not found: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [c for c in REGISTER_COLUMNS if c not in df.columns]
    if missing:
        raise RegisterSchemaError(f"{p}: missing columns {missing}; expected {REGISTER_COLUMNS}")
    if village is not None:
        df = df[df["village_code"] == village]
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        try:
            rows.append(register_row_from_dict({
                "register_row_id": rec["register_row_id"],
                "village_code": rec["village_code"],
                "khewat_no": rec["khewat_no"],
                "khatauni_no": rec["khatauni_no"],
                "owners": json.loads(rec["owners_json"] or "[]"),
                "cultivator": rec["cultivator"],
                "plots": json.loads(rec["plots_json"] or "[]"),
                "mutation_history": json.loads(rec["mutation_history_json"] or "[]"),
                "last_mutation_ref": rec["last_mutation_ref"] or None,
            }))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RegisterSchemaError(f"{p}: bad JSON in data row {i + 1}: {exc}") from exc
    return rows
