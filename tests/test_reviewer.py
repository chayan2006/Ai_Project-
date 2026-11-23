"""Phase 6 check (report D.3): a full review session works and corrections are saved in SQLite."""
import json

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from lre import store

APP = "../app/reviewer.py"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    disp = tmp_path / "display.jpg"
    Image.fromarray(np.full((600, 1600), 230, np.uint8)).save(disp)
    crop = tmp_path / "cell.png"
    Image.fromarray(np.full((60, 200), 200, np.uint8)).save(crop)
    path = tmp_path / "lre.db"
    conn = store.connect(path)
    store.add_page(conn, "V000-P0001", "V000", "scan.png", "abc", 300, 0.4, disp, 0.5, status="processing")
    fields = [{"field_name": name, "raw_text": raw, "normalised_value": raw, "register_value": reg,
               "confidence_raw": 0.4, "bbox": [100, 300, 200, 120], "crop_path": str(crop)}
              for name, raw, reg in (("khewat_no", "4l", "41"), ("owners", "राम सिंह s/o हरी सिंह 1/2", "राम सिंह s/o हरी सिंह 1/2"),
                                     ("khasra_nos", "112//3", "112//3"))]
    flags = [{"rule": "R6", "name": "share_differs", "kind": "cross", "severity": "high",
              "evidence": {"scan_share": "1/2", "register_share": "1/3"}}]
    store.add_holding(conn, {"holding_id": "V000-P0001-B01", "page_id": "V000-P0001", "row_index": 0,
                             "village_code": "V000", "khewat_no": "41", "khatauni_no": "57",
                             "route": "CONTRADICTION", "link": {"register_row_id": "V000-41", "method": "khewat",
                                                                "score": 1.0},
                             "row_span": [280, 460], "holding_json": {}, "models": {}}, fields, flags)
    store.add_holding(conn, {"holding_id": "V000-P0001-B02", "page_id": "V000-P0001", "row_index": 1,
                             "village_code": "V000", "khewat_no": "42", "khatauni_no": "58", "route": "AUTO",
                             "link": None, "row_span": [460, 600], "holding_json": {}, "models": {}}, [], [])
    conn.commit()
    conn.close()
    monkeypatch.setenv("LRE_DB", str(path))
    return path


def test_full_review_session(db):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "V000-P0001-B01" in at.subheader[0].value          # AUTO holding is not in the queue
    conn = store.connect(db)
    fid = {r["field_name"]: r["field_id"] for r in conn.execute("SELECT * FROM field_extraction")}
    flag_id = conn.execute("SELECT flag_id FROM flag").fetchone()[0]

    at.button(key=f"accept_{fid['khewat_no']}").click().run()       # khewat_no is the selected field
    assert not at.exception, at.exception
    at.radio(key="field_V000-P0001-B01").set_value("owners").run()
    at.text_input(key=f"corr_{fid['owners']}").input("राम सिंह s/o हरी सिंह 1/3")
    form_submit = next(b for b in at.button if b.label == "Save correction")
    form_submit.click().run()
    assert not at.exception, at.exception
    at.radio(key="field_V000-P0001-B01").set_value("khasra_nos").run()
    at.button(key=f"ocr_{fid['khasra_nos']}").click().run()
    at.button(key=f"confirm_{flag_id}").click().run()
    at.button(key="finish_V000-P0001-B01").click().run()
    assert not at.exception, at.exception

    conn = store.connect(db)
    decisions = [dict(r) for r in conn.execute("SELECT action, corrected_value FROM review_decision ORDER BY 1")]
    assert [d["action"] for d in decisions] == ["accept", "confirm", "correct", "ocr_error"]
    assert any(d["corrected_value"] == "राम सिंह s/o हरी सिंह 1/3" for d in decisions)
    assert conn.execute("SELECT status FROM holding WHERE holding_id = 'V000-P0001-B01'").fetchone()[0] == "escalated"
    assert conn.execute("SELECT status FROM flag").fetchone()[0] == "escalated"
    corrections = store.export_corrections(conn, "V000")
    assert "1/3" in corrections and "ocr_error" in corrections
    assert store.verify_audit_chain(conn) == (True, None)
    assert "Queue empty" in at.success[0].value                # the only queued holding is finished


def test_audit_chain_detects_tampering(db):
    conn = store.connect(db)
    store.audit(conn, "test", {"x": 1})
    store.audit(conn, "test", {"x": 2})
    conn.commit()
    assert store.verify_audit_chain(conn)[0]
    conn.execute("UPDATE audit_log SET payload = ? WHERE seq = 2", (json.dumps({"x": 99}),))
    ok, seq = store.verify_audit_chain(conn)
    assert not ok and seq == 2


def test_field_actions_validated(db):
    conn = store.connect(db)
    fid = conn.execute("SELECT field_id FROM field_extraction").fetchone()[0]
    with pytest.raises(ValueError):
        store.decide_field(conn, fid, "correct", "r1", "  ")
    with pytest.raises(ValueError):
        store.decide_field(conn, fid, "delete", "r1")
