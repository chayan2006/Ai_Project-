"""Pipeline orchestrator (report 6.2).

Runs the stages page by page and writes a JSON checkpoint after each stage to
data/interim/<page_id>/<stage>.json so any stage can be re-run or inspected.
Stages are filled in as the build phases land (report Appendix D.3).
"""
from __future__ import annotations

from pathlib import Path

from .io import read_json, write_json

STAGES = ["ingest", "preprocess", "layout", "recognise", "extract", "link", "validate", "route"]


def checkpoint_path(interim_dir: str | Path, page_id: str, stage: str) -> Path:
    return Path(interim_dir) / page_id / f"{stage}.json"


def save_checkpoint(interim_dir: str | Path, page_id: str, stage: str, obj) -> Path:
    p = checkpoint_path(interim_dir, page_id, stage)
    write_json(p, obj)
    return p


def load_checkpoint(interim_dir: str | Path, page_id: str, stage: str):
    p = checkpoint_path(interim_dir, page_id, stage)
    return read_json(p) if p.exists() else None


def last_completed_stage(interim_dir: str | Path, page_id: str) -> str | None:
    """Resume point after a crash mid-batch (report E.11)."""
    done = None
    for s in STAGES:
        if checkpoint_path(interim_dir, page_id, s).exists():
            done = s
        else:
            break
    return done
