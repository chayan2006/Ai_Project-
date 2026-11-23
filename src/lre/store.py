"""SQLite result store (report 6.2, Figure 9) with an append-only, hash-chained audit log (E.10).

Tables: page, register_row, holding, field_extraction, flag, review_decision, audit_log.
Every reviewer decision is mirrored into the audit log; each audit entry stores the hash
of the previous one, so editing or deleting any entry breaks the chain.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS page (
    page_id TEXT PRIMARY KEY, village_code TEXT, register_year TEXT, image_path TEXT,
    display_path TEXT, display_scale REAL, sha256 TEXT UNIQUE, dpi INTEGER, skew_deg REAL,
    status TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS register_row (
    register_row_id TEXT PRIMARY KEY, village_code TEXT, khewat_no TEXT, khatauni_no TEXT,
    owners_json TEXT, plots_json TEXT, cultivator TEXT, mutation_history_json TEXT, last_mutation_ref TEXT);
CREATE TABLE IF NOT EXISTS holding (
    holding_id TEXT PRIMARY KEY, page_id TEXT REFERENCES page(page_id), row_index INTEGER,
    village_code TEXT, khewat_no TEXT, khatauni_no TEXT, route TEXT, register_row_id TEXT,
    link_method TEXT, link_score REAL, row_y0 INTEGER, row_y1 INTEGER, holding_json TEXT,
    models_json TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS field_extraction (
    field_id INTEGER PRIMARY KEY AUTOINCREMENT, holding_id TEXT REFERENCES holding(holding_id),
    field_name TEXT, raw_text TEXT, normalised_value TEXT, register_value TEXT,
    confidence_raw REAL, confidence_cal REAL, bbox TEXT, crop_path TEXT, model_version TEXT,
    status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS flag (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT, holding_id TEXT REFERENCES holding(holding_id),
    rule_id TEXT, name TEXT, kind TEXT, severity TEXT, score REAL, evidence_json TEXT,
    status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS review_decision (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT, holding_id TEXT, field_id INTEGER, flag_id INTEGER,
    action TEXT, corrected_value TEXT, reviewer_id TEXT, timestamp TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, payload TEXT, timestamp TEXT,
    prev_hash TEXT, hash TEXT);
CREATE INDEX IF NOT EXISTS ix_holding_village ON holding(village_code, route, status);
CREATE INDEX IF NOT EXISTS ix_field_holding ON field_extraction(holding_id);
CREATE INDEX IF NOT EXISTS ix_flag_holding ON flag(holding_id);
"""

FIELD_ACTIONS = {"accept", "correct", "ocr_error"}
FLAG_ACTIONS = {"confirm", "dismiss"}       # confirm = escalate to the revenue officer
GENESIS = "0" * 64


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# ----------------------------------------------------------------------------- audit log


def _entry_hash(prev: str, event: str, payload: str, ts: str) -> str:
    return hashlib.sha256(f"{prev}|{event}|{payload}|{ts}".encode()).hexdigest()


def audit(conn: sqlite3.Connection, event: str, payload: dict) -> str:
    row = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    ts = now()
    h = _entry_hash(prev, event, body, ts)
    conn.execute("INSERT INTO audit_log(event, payload, timestamp, prev_hash, hash) VALUES (?,?,?,?,?)",
                 (event, body, ts, prev, h))
    return h


def verify_audit_chain(conn: sqlite3.Connection) -> tuple[bool, Optional[int]]:
    """(True, None) if intact, else (False, seq of the first bad entry)."""
    prev = GENESIS
    for r in conn.execute("SELECT * FROM audit_log ORDER BY seq"):
        if r["prev_hash"] != prev or _entry_hash(prev, r["event"], r["payload"], r["timestamp"]) != r["hash"]:
            return False, r["seq"]
        prev = r["hash"]
    return True, None


# ----------------------------------------------------------------------------- writes from the pipeline


def page_exists(conn: sqlite3.Connection, sha256: str) -> Optional[str]:
    r = conn.execute("SELECT page_id FROM page WHERE sha256 = ?", (sha256,)).fetchone()
    return r["page_id"] if r else None


def add_page(conn, page_id, village_code, image_path, sha256, dpi, skew_deg, display_path, display_scale,
             register_year="", status="processed") -> None:
    conn.execute("INSERT OR REPLACE INTO page VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (page_id, village_code, register_year, str(image_path), str(display_path), display_scale,
                  sha256, dpi, skew_deg, status, now()))
    audit(conn, "page_ingested", {"page_id": page_id, "sha256": sha256})


def set_page_status(conn, page_id: str, status: str) -> None:
    conn.execute("UPDATE page SET status = ? WHERE page_id = ?", (status, page_id))


def add_register_rows(conn, rows) -> None:
    from .io import register_rows_to_frame
    for rec in register_rows_to_frame(rows).to_dict("records"):
        conn.execute("INSERT OR REPLACE INTO register_row VALUES (?,?,?,?,?,?,?,?,?)",
                     (rec["register_row_id"], rec["village_code"], rec["khewat_no"], rec["khatauni_no"],
                      rec["owners_json"], rec["plots_json"], rec["cultivator"], rec["mutation_history_json"],
                      rec["last_mutation_ref"]))


def add_holding(conn, holding: dict, fields: list[dict], flags: list[dict]) -> None:
    """``holding``: dict with holding_id, page_id, row_index, village_code, khewat_no, khatauni_no, route,
    link, row_span, holding_json, models. ``fields``: field_extraction rows. ``flags``: Flag dicts."""
    link = holding.get("link") or {}
    conn.execute("DELETE FROM field_extraction WHERE holding_id = ?", (holding["holding_id"],))
    conn.execute("DELETE FROM flag WHERE holding_id = ?", (holding["holding_id"],))
    conn.execute("INSERT OR REPLACE INTO holding VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (holding["holding_id"], holding["page_id"], holding["row_index"], holding["village_code"],
                  holding["khewat_no"], holding["khatauni_no"], holding["route"], link.get("register_row_id"),
                  link.get("method"), link.get("score"), holding["row_span"][0], holding["row_span"][1],
                  json.dumps(holding["holding_json"], ensure_ascii=False), json.dumps(holding.get("models", {})),
                  "open"))
    for f in fields:
        conn.execute("INSERT INTO field_extraction(holding_id, field_name, raw_text, normalised_value, register_value,"
                     " confidence_raw, confidence_cal, bbox, crop_path, model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (holding["holding_id"], f["field_name"], f.get("raw_text", ""), f.get("normalised_value", ""),
                      f.get("register_value", ""), f.get("confidence_raw"), f.get("confidence_cal"),
                      json.dumps(f.get("bbox")), f.get("crop_path", ""), f.get("model_version", "")))
    for fl in flags:
        conn.execute("INSERT INTO flag(holding_id, rule_id, name, kind, severity, score, evidence_json)"
                     " VALUES (?,?,?,?,?,?,?)",
                     (holding["holding_id"], fl["rule"], fl["name"], fl["kind"], fl["severity"], fl.get("score"),
                      json.dumps(fl.get("evidence", {}), ensure_ascii=False)))


# ----------------------------------------------------------------------------- reads for the reviewer


def villages(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT village_code FROM holding ORDER BY 1")]


def route_counts(conn, village: str) -> dict[str, int]:
    return {r["route"]: r["n"] for r in conn.execute(
        "SELECT route, COUNT(*) AS n FROM holding WHERE village_code = ? GROUP BY route", (village,))}


def queue(conn, village: str, routes=("CONTRADICTION", "REVIEW"), include_done: bool = False) -> list[dict]:
    """Holdings awaiting review: CONTRADICTION first, then REVIEW, in page order."""
    q = (f"SELECT * FROM holding WHERE village_code = ? AND route IN ({','.join('?' * len(routes))})"
         + ("" if include_done else " AND status = 'open'")
         + " ORDER BY CASE route WHEN 'CONTRADICTION' THEN 0 ELSE 1 END, page_id, row_index")
    return [dict(r) for r in conn.execute(q, (village, *routes))]


def get_holding(conn, holding_id: str) -> dict:
    h = dict(conn.execute("SELECT * FROM holding WHERE holding_id = ?", (holding_id,)).fetchone())
    h["page"] = dict(conn.execute("SELECT * FROM page WHERE page_id = ?", (h["page_id"],)).fetchone())
    h["fields"] = [dict(r) for r in conn.execute(
        "SELECT * FROM field_extraction WHERE holding_id = ? ORDER BY field_id", (holding_id,))]
    h["flags"] = [dict(r) for r in conn.execute(
        "SELECT * FROM flag WHERE holding_id = ? ORDER BY flag_id", (holding_id,))]
    return h


# ----------------------------------------------------------------------------- reviewer decisions


def decide_field(conn, field_id: int, action: str, reviewer: str, corrected_value: str = "") -> None:
    if action not in FIELD_ACTIONS:
        raise ValueError(f"field action must be one of {sorted(FIELD_ACTIONS)}")
    if action == "correct" and not corrected_value.strip():
        raise ValueError("a correction needs a value")
    f = conn.execute("SELECT holding_id, field_name, normalised_value FROM field_extraction WHERE field_id = ?",
                     (field_id,)).fetchone()
    ts = now()
    conn.execute("INSERT INTO review_decision(holding_id, field_id, action, corrected_value, reviewer_id, timestamp)"
                 " VALUES (?,?,?,?,?,?)", (f["holding_id"], field_id, action, corrected_value, reviewer, ts))
    conn.execute("UPDATE field_extraction SET status = ? WHERE field_id = ?", (action, field_id))
    audit(conn, "field_" + action, {"holding_id": f["holding_id"], "field_id": field_id, "field": f["field_name"],
                                    "before": f["normalised_value"], "after": corrected_value, "reviewer": reviewer})
    conn.commit()


def decide_flag(conn, flag_id: int, action: str, reviewer: str) -> None:
    if action not in FLAG_ACTIONS:
        raise ValueError(f"flag action must be one of {sorted(FLAG_ACTIONS)}")
    fl = conn.execute("SELECT holding_id, rule_id FROM flag WHERE flag_id = ?", (flag_id,)).fetchone()
    conn.execute("INSERT INTO review_decision(holding_id, flag_id, action, reviewer_id, timestamp) VALUES (?,?,?,?,?)",
                 (fl["holding_id"], flag_id, action, reviewer, now()))
    conn.execute("UPDATE flag SET status = ? WHERE flag_id = ?",
                 ("escalated" if action == "confirm" else "dismissed", flag_id))
    audit(conn, "flag_" + action, {"holding_id": fl["holding_id"], "flag_id": flag_id, "rule": fl["rule_id"],
                                   "reviewer": reviewer})
    conn.commit()


def close_holding(conn, holding_id: str, reviewer: str) -> str:
    """Mark a holding reviewed; 'escalated' if any of its flags was confirmed."""
    esc = conn.execute("SELECT COUNT(*) FROM flag WHERE holding_id = ? AND status = 'escalated'",
                       (holding_id,)).fetchone()[0]
    status = "escalated" if esc else "reviewed"
    conn.execute("UPDATE holding SET status = ? WHERE holding_id = ?", (status, holding_id))
    audit(conn, "holding_" + status, {"holding_id": holding_id, "reviewer": reviewer})
    conn.commit()
    return status


# ----------------------------------------------------------------------------- exports


def _csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def export_contradictions(conn, village: str) -> str:
    rows = [dict(r) for r in conn.execute(
        "SELECT h.holding_id, h.page_id, h.khewat_no, h.register_row_id, h.link_method, h.status AS holding_status,"
        " f.rule_id, f.name, f.severity, f.score, f.status AS flag_status, f.evidence_json"
        " FROM flag f JOIN holding h USING(holding_id) WHERE h.village_code = ? AND h.route = 'CONTRADICTION'"
        " ORDER BY h.page_id, h.row_index, f.flag_id", (village,))]
    return _csv(rows, ["holding_id", "page_id", "khewat_no", "register_row_id", "link_method", "holding_status",
                       "rule_id", "name", "severity", "score", "flag_status", "evidence_json"])


def export_corrections(conn, village: Optional[str] = None) -> str:
    """Reviewer corrections with the cell crop they refer to: new training data for fine-tuning (E.8)."""
    q = ("SELECT d.decision_id, d.timestamp, d.reviewer_id, d.action, d.corrected_value, fe.holding_id,"
         " fe.field_name, fe.raw_text, fe.normalised_value, fe.bbox, fe.crop_path, h.page_id"
         " FROM review_decision d JOIN field_extraction fe ON fe.field_id = d.field_id"
         " JOIN holding h ON h.holding_id = fe.holding_id WHERE d.action IN ('correct', 'ocr_error')")
    args: tuple = ()
    if village:
        q += " AND h.village_code = ?"
        args = (village,)
    rows = [dict(r) for r in conn.execute(q + " ORDER BY d.decision_id", args)]
    return _csv(rows, ["decision_id", "timestamp", "reviewer_id", "action", "page_id", "holding_id", "field_name",
                       "raw_text", "normalised_value", "corrected_value", "bbox", "crop_path"])
