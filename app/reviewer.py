"""Streamlit reviewer (report 6.3): the scanned page beside the extracted fields.

Run from the project root after `lre run`:  streamlit run app/reviewer.py
The store defaults to results/lre.db (override with the LRE_DB environment variable).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from lre import store

st.set_page_config(page_title="LRE Reviewer", layout="wide")

DB = os.environ.get("LRE_DB", "results/lre.db")


@st.cache_resource
def get_conn(path: str):
    return store.connect(path)


def page_image(h: dict, field: dict | None) -> Image.Image:
    p = h["page"]
    img = Image.open(p["display_path"]).convert("RGB")
    s = p["display_scale"] or 1.0
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, h["row_y0"] * s, img.width, h["row_y1"] * s], fill=(255, 196, 0, 60), outline=(200, 120, 0, 255),
                width=3)
    if field and field.get("bbox"):
        x, y, w, hh = json.loads(field["bbox"])
        d.rectangle([x * s, y * s, (x + w) * s, (y + hh) * s], outline=(220, 30, 30, 255), width=4)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def crop_view(img: Image.Image, h: dict, margin: int = 140) -> Image.Image:
    """The holding's band of the page with some context above and below."""
    s = h["page"]["display_scale"] or 1.0
    y0 = max(0, int(h["row_y0"] * s) - margin)
    y1 = min(img.height, int(h["row_y1"] * s) + margin)
    return img.crop((0, y0, img.width, y1))


def conf_text(v) -> str:
    return "-" if v is None else f"{v:.2f}"


if not Path(DB).exists():
    st.title("LRE Reviewer")
    st.warning(f"No result store at `{DB}`. Run `lre run --village <code> --pages <folder>` first.")
    st.stop()

conn = get_conn(DB)

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("LRE Reviewer")
villages = store.villages(conn)
if not villages:
    st.sidebar.warning("The store has no holdings yet.")
    st.stop()
village = st.sidebar.selectbox("Village", villages, key="village")
reviewer = st.sidebar.text_input("Reviewer id", value="reviewer1", key="reviewer")
which = st.sidebar.radio("Queue", ["Contradictions and review", "Contradictions only", "Review only"], key="which")
routes = {"Contradictions and review": ("CONTRADICTION", "REVIEW"), "Contradictions only": ("CONTRADICTION",),
          "Review only": ("REVIEW",)}[which]
show_done = st.sidebar.checkbox("Include finished holdings", value=False, key="show_done")

counts = store.route_counts(conn, village)
c1, c2, c3 = st.sidebar.columns(3)
c1.metric("Auto", counts.get("AUTO", 0))
c2.metric("Review", counts.get("REVIEW", 0))
c3.metric("Contra.", counts.get("CONTRADICTION", 0))

ok, bad_seq = store.verify_audit_chain(conn)
st.sidebar.caption("Audit log: intact" if ok else f"Audit log BROKEN at entry {bad_seq}")
st.sidebar.download_button("Export contradiction report", store.export_contradictions(conn, village),
                           file_name=f"contradiction_report_{village}.csv", mime="text/csv", key="export_contra")
st.sidebar.download_button("Export corrections", store.export_corrections(conn, village),
                           file_name=f"corrections_{village}.csv", mime="text/csv", key="export_corr")

queue = store.queue(conn, village, routes, include_done=show_done)
if not queue:
    st.success("Queue empty: nothing left to review for this village.")
    st.stop()

labels = {q["holding_id"]: f"{q['route'][:6]} | khewat {q['khewat_no'] or '?'} | {q['holding_id']}"
          + ("" if q["status"] == "open" else f" ({q['status']})") for q in queue}
ids = list(labels)
if st.session_state.get("holding") not in ids:
    st.session_state["holding"] = ids[0]
hid = st.sidebar.selectbox("Holding", ids, format_func=labels.get, key="holding")
st.sidebar.caption(f"{ids.index(hid) + 1} of {len(ids)} in queue")

h = store.get_holding(conn, hid)
fields = h["fields"]

# ----------------------------------------------------------------------------- main view
st.subheader(f"{h['holding_id']}  ·  khewat {h['khewat_no'] or '?'}  ·  route {h['route']}")
link_txt = (f"linked to register row **{h['register_row_id']}** by {h['link_method']} (score {h['link_score']})"
            if h["register_row_id"] else "**not linked** to any register row")
st.caption(f"Page {h['page_id']} · {link_txt} · status {h['status']}")

left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown("**Extracted fields**")
    table = pd.DataFrame([{"field": f["field_name"], "extracted": f["normalised_value"],
                           "register": f["register_value"], "conf": conf_text(f["confidence_raw"]),
                           "status": f["status"]} for f in fields])
    st.dataframe(table, hide_index=True, use_container_width=True)
    names = [f["field_name"] for f in fields]
    chosen = st.radio("Field to check", names, horizontal=True, key=f"field_{hid}")
    field = fields[names.index(chosen)]

    if field.get("crop_path") and Path(field["crop_path"]).exists():
        st.image(field["crop_path"], caption=f"{chosen}: cell as scanned", use_column_width=True)
    st.markdown(f"Read as: `{field['raw_text'] or '(nothing)'}`")
    st.markdown(f"Normalised: **{field['normalised_value'] or '-'}**  ·  register: **{field['register_value'] or '-'}**")

    b1, b2 = st.columns(2)
    if b1.button("Accept", key=f"accept_{field['field_id']}", use_container_width=True):
        store.decide_field(conn, field["field_id"], "accept", reviewer)
        st.rerun()
    if b2.button("Mark OCR error", key=f"ocr_{field['field_id']}", use_container_width=True):
        store.decide_field(conn, field["field_id"], "ocr_error", reviewer)
        st.rerun()
    with st.form(key=f"form_{field['field_id']}", clear_on_submit=True):
        corrected = st.text_input("Correct value", value="", key=f"corr_{field['field_id']}")
        if st.form_submit_button("Save correction"):
            if corrected.strip():
                store.decide_field(conn, field["field_id"], "correct", reviewer, corrected)
                st.rerun()
            else:
                st.error("Type the correct value first.")

with left:
    img = page_image(h, field)
    st.image(crop_view(img, h), caption="Holding on the deskewed scan (yellow), selected cell (red)",
             use_column_width=True)
    with st.expander("Whole page"):
        st.image(img, use_column_width=True)

st.markdown("---")
st.markdown(f"**Flags ({len(h['flags'])})**")
if not h["flags"]:
    st.caption("No rule fired for this holding.")
for fl in h["flags"]:
    with st.container(border=True):
        a, b, c = st.columns([4, 1, 1])
        score = "" if fl["score"] is None else f" · score {fl['score']:.2f}"
        a.markdown(f"**{fl['rule_id']} {fl['name']}** · {fl['severity']}{score} · _{fl['status']}_")
        a.json(json.loads(fl["evidence_json"] or "{}"), expanded=False)
        is_open = fl["status"] == "open"
        if b.button("Confirm & escalate", key=f"confirm_{fl['flag_id']}", disabled=not is_open):
            store.decide_flag(conn, fl["flag_id"], "confirm", reviewer)
            st.rerun()
        if c.button("Dismiss", key=f"dismiss_{fl['flag_id']}", disabled=not is_open):
            store.decide_flag(conn, fl["flag_id"], "dismiss", reviewer)
            st.rerun()

open_flags = sum(fl["status"] == "open" for fl in h["flags"])
open_fields = sum(f["status"] == "open" for f in fields)
st.caption(f"{open_fields} field(s) and {open_flags} flag(s) still open. "
           "Finishing marks the holding reviewed, or escalated if a flag was confirmed.")
if st.button("Finish holding", type="primary", key=f"finish_{hid}", disabled=h["status"] != "open"):
    status = store.close_holding(conn, hid, reviewer)
    st.session_state.pop("holding", None)
    st.toast(f"{hid}: {status}")
    st.rerun()
