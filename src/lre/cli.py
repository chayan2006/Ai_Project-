"""Command-line interface (report Table 20)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import Config, load_config
from .logging_utils import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Legacy Land Record Digitisation and Validation Engine.")
log = get_logger()
_state: dict = {"config_path": "config.yaml"}

FONT_URLS = {
    "Kalam-Light.ttf": "kalam/Kalam-Light.ttf",
    "Kalam-Regular.ttf": "kalam/Kalam-Regular.ttf",
    "Kalam-Bold.ttf": "kalam/Kalam-Bold.ttf",
    "Amita-Regular.ttf": "amita/Amita-Regular.ttf",
    "Tillana-Regular.ttf": "tillana/Tillana-Regular.ttf",
    "Tillana-Medium.ttf": "tillana/Tillana-Medium.ttf",
    "Hind-Regular.ttf": "hind/Hind-Regular.ttf",
}
FONT_BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"


def cfg() -> Config:
    return load_config(_state["config_path"])


def _todo(phase: str) -> None:
    typer.secho(f"Not implemented yet: planned for {phase} of the build plan (report Appendix D.3).",
                fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


@app.callback()
def main(config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml.")):
    from ._platform import ensure_fribidi
    ensure_fribidi()
    _state["config_path"] = config


@app.command()
def version():
    """Print the package version and config hash."""
    typer.echo(f"lre {__version__} (config {cfg().hash()})")


@app.command()
def fonts(force: bool = typer.Option(False, help="Re-download fonts that already exist.")):
    """Download the OFL handwriting-style Devanagari fonts used by SynJama (Google Fonts)."""
    out = Path(cfg().paths.fonts_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in FONT_URLS.items():
        dest = out / name
        if dest.exists() and not force:
            typer.echo(f"have  {name}")
            continue
        urllib.request.urlretrieve(FONT_BASE + rel, dest)
        typer.echo(f"saved {name}")


@app.command()
def tessdata(force: bool = typer.Option(False, help="Re-download models that already exist.")):
    """Download Tesseract's Hindi and English LSTM models into models/tessdata (baseline only)."""
    out = Path(cfg().paths.models_dir) / "tessdata"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("hin", "eng"):
        dest = out / f"{lang}.traineddata"
        if dest.exists() and not force:
            typer.echo(f"have  {dest.name}")
            continue
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/{lang}.traineddata", dest)
        typer.echo(f"saved {dest.name}")


@app.command()
def synth(
    pages: int = typer.Option(1000, help="Total number of pages to generate."),
    out: Path = typer.Option(Path("data/synjama"), help="Output folder."),
    split: str = typer.Option("all", help="all (train/val/test by village) or one split name, e.g. sample."),
    seed: int = typer.Option(0, help="Base random seed."),
    workers: int = typer.Option(1, help="Parallel render processes."),
    village_start: int = typer.Option(0, help="First village number (single split only), e.g. to regenerate "
                                             "one split of an existing set without reusing village codes."),
):
    """Generate the synthetic SynJama register with ground truth and injected contradictions."""
    from .synth.generator import generate_all, generate_split
    c = cfg()
    if split == "all":
        res = generate_all(out, pages, c, seed=seed, workers=workers)
    else:
        res = {split: generate_split(out, split, pages, c, seed=seed, village_start=village_start, workers=workers)}
    for name, m in res.items():
        typer.echo(f"{name}: {m['pages']} pages, {m['holdings_on_pages']} holdings, "
                   f"{m['text_lines']} text lines, {len(m['villages'])} villages")
        typer.echo("  labels on pages: " + json.dumps(m["labels_on_pages"], sort_keys=True))


@app.command()
def ingest(village: str = typer.Option(...), pages: Path = typer.Option(...)):
    """Register scanned pages (PDF/TIFF/JPG/PNG) in the store."""
    _todo("Phase 2")


@app.command()
def extract(village: str = typer.Option(...)):
    """Preprocess, segment, recognise and build holdings."""
    _todo("Phases 2 and 5")


@app.command()
def validate(
    holdings: Path = typer.Option(..., help="Holdings JSONL (one holding per line), e.g. extractor output."),
    register: Optional[Path] = typer.Option(None, help="Digital register CSV (default paths.register_csv)."),
    village: Optional[str] = typer.Option(None, help="Only this village code."),
    profile: str = typer.Option("default", help="Rule parameter profile from rules.yaml (default or exact)."),
    out: Path = typer.Option(Path("results/contradiction_report.csv"), help="Contradiction report CSV."),
):
    """Link holdings to the register, run rules R1-R10 and write the contradiction report."""
    from .io import load_register
    from .schema import holding_from_dict, to_dict
    from .validate.engine import load_rules
    from .validate.runner import validate_holdings, write_contradiction_report
    c = cfg()
    rules = load_rules(c.paths.rules, profile)
    rows = load_register(register or c.paths.register_csv, village)
    with open(holdings, encoding="utf-8") as fh:
        hs = [holding_from_dict(json.loads(line)) for line in fh if line.strip()]
    if village:
        hs = [h for h in hs if h.village_code == village]
    validate_holdings(hs, rows, rules, c.linking)
    n = write_contradiction_report(hs, rules, out)
    flagged_out = out.with_suffix(".holdings.jsonl")
    with open(flagged_out, "w", encoding="utf-8") as fh:
        for h in hs:
            fh.write(json.dumps(to_dict(h), ensure_ascii=False) + "\n")
    unlinked = sum(1 for h in hs if h.link and h.link.method == "unlinked")
    typer.echo(f"{len(hs)} holdings, {n} contradictions, {unlinked} unlinked "
               f"(profile {rules.profile}) -> {out}, {flagged_out}")


@app.command()
def route(village: str = typer.Option(...)):
    """Calibrated routing to AUTO / REVIEW / CONTRADICTION."""
    _todo("Phase 5")


@app.command()
def run(
    village: str = typer.Option(..., help="Village code of the pages (must exist in the register)."),
    pages: Path = typer.Option(..., help="A page file (PNG/JPG/TIFF/PDF) or a folder of them."),
    register: Optional[Path] = typer.Option(None, help="Digital register CSV (default paths.register_csv)."),
    db: Optional[Path] = typer.Option(None, help="Result store (default paths.db_path)."),
    extractor: str = typer.Option("baseline", help="baseline (template + Tesseract) or crnn (proposed)."),
    crnn: Path = typer.Option(Path("models/crnn_B.pt"), help="CRNN checkpoint for --extractor crnn."),
):
    """Ingest, extract, link, validate and route pages into the result store for the reviewer."""
    from .run import run_pages
    c = cfg()
    s = run_pages(pages, village, register or Path(c.paths.register_csv), c, db or Path(c.paths.db_path),
                  extractor=extractor, log=typer.echo, crnn_path=crnn)
    typer.echo(f"{s['pages']} pages ({s['duplicates']} duplicates skipped), {s['holdings']} holdings: "
               f"AUTO {s['AUTO']}, REVIEW {s['REVIEW']}, CONTRADICTION {s['CONTRADICTION']} "
               f"-> {db or c.paths.db_path}; open the reviewer with: streamlit run app/reviewer.py")


@app.command()
def evaluate(
    split: str = typer.Option("val", help="SynJama split under data/synjama (ignored if --data is given)."),
    stage: str = typer.Option("layout", help="layout | baseline | validation | proposed | proposed_b."),
    data: Optional[Path] = typer.Option(None, help="Folder with pages/ and gt/ (default data/synjama/<split>)."),
    limit: Optional[int] = typer.Option(None, help="Evaluate only the first N pages."),
    workers: int = typer.Option(1, help="Parallel processes (baseline stage)."),
    unet: Optional[Path] = typer.Option(None, help="Layout: U-Net-lite checkpoint used as fallback."),
    stress: bool = typer.Option(False, help="Layout: fade and break the ruling lines before detection."),
    crnn: Path = typer.Option(Path("models/crnn_B.pt"), help="Proposed: CRNN checkpoint."),
):
    """Compute metrics against ground truth and write them to results/."""
    c = cfg()
    folder = data or Path(c.paths.data_dir) / "synjama" / split
    if stage in ("baseline", "proposed"):
        from .evaluate.report import evaluate_baseline, evaluate_proposed
        if stage == "baseline":
            m = evaluate_baseline(folder, c, workers=workers, limit=limit)
        else:
            m = evaluate_proposed(folder, c, crnn, unet, limit=limit)
        for k in ("pages", "holdings", "missed_holdings", "mean_field_accuracy", "numeric_field_accuracy",
                  "name_cer", "remarks_cer", "review_share", "auto_accept_accuracy"):
            v = m[k]
            typer.echo(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")
        for f, v in m["per_field_accuracy"].items():
            typer.echo(f"  {f:22s} {v:.4f}" if v is not None else f"  {f:22s} -")
        ct = m.get("contradiction", {})
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        typer.echo(f"contradiction precision {fmt(ct.get('precision'))}, recall {fmt(ct.get('recall'))} "
                   f"(tp {ct.get('tp')}, fp {ct.get('fp')}, fn {ct.get('fn')})")
        typer.echo(f"-> {c.paths.results_dir}/{m['system']}_metrics.json [{folder.name}]")
        return
    if stage == "proposed_b":
        from .evaluate.proposed_b import evaluate_proposed_b
        root = Path(c.paths.data_dir) / "synjama"
        m = evaluate_proposed_b(c, root / "train", root / "val", folder)
        ct = m["contradiction"]
        typer.echo(f"theta {m['theta']:.3f} (scorer trained on {m['scorer_train_flags']} flags, "
                   f"{m['scorer_train_positive']} true)")
        for k in ("holdings", "mean_field_accuracy", "numeric_field_accuracy", "name_cer", "review_share",
                  "auto_accept_accuracy"):
            v = m[k]
            typer.echo(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")
        typer.echo(f"contradiction precision {ct['precision']}, recall {ct['recall']} "
                   f"(tp {ct['tp']}, fp {ct['fp']}, fn {ct['fn']})")
        typer.echo(json.dumps(m["calibration"], indent=1, ensure_ascii=False))
        return
    if stage == "validation":
        from .evaluate.report import update_metrics_file
        from .evaluate.validation_eval import evaluate_validation_on_gt
        from .validate.engine import load_rules
        m = evaluate_validation_on_gt(folder, c, load_rules(c.paths.rules))
        update_metrics_file(Path(c.paths.results_dir) / "validation_metrics.json", folder.name, m)
        hl = m["holding_level"]
        typer.echo(f"holdings {m['holdings']}, positives {m['positives']}, flagged {m['flagged_holdings']}")
        typer.echo(f"precision {hl['precision']}, recall {hl['recall']} (tp {hl['tp']}, fp {hl['fp']}, fn {hl['fn']})")
        typer.echo(f"link accuracy {m['link_accuracy']:.4f} {m['link_methods']}")
        for t, v in m["expected_rule_fired"].items():
            typer.echo(f"  expected rule fired for {t:20s} {v:.3f}")
        verdict = "PASS" if m["acceptance_passed"] else "FAIL"
        typer.echo(f"Phase 4 acceptance (ground-truth fields: precision 1.0, recall 1.0): {verdict} -> {m['report']}")
        return
    if stage != "layout":
        _todo("Phase 3")
    from .evaluate.layout_eval import evaluate_layout
    model = None
    if unet is not None:
        from .layout.unet import load_model
        model = load_model(unet)
        if model is None:
            raise typer.BadParameter(f"no checkpoint at {unet}")
    m = evaluate_layout(folder, c, limit=limit, unet=model, stress=stress)
    m.update({"data": str(folder), "config_hash": c.hash(), "unet": str(unet) if unet else None})
    tag = folder.name + ("_stress" if stress else "") + ("_unet" if unet else "")
    out = Path(c.paths.results_dir) / f"layout_metrics_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    for k in ("pages", "column_count_accuracy", "grid_ok_rate", "mean_cell_iou", "row_count_accuracy",
              "line_split_accuracy", "mean_skew_err_deg", "unet_fallback_used"):
        typer.echo(f"{k:24s} {m[k]:.4f}" if isinstance(m[k], float) else f"{k:24s} {m[k]}")
    verdict = "PASS" if m["acceptance_passed"] else "FAIL"
    typer.echo(f"Phase 2 acceptance (columns >= 0.95, cell IoU >= 0.90): {verdict}  -> {out}")


@app.command("build-lines")
def build_lines_cmd(
    split: str = typer.Option("train", help="SynJama split under data/synjama."),
    data: Optional[Path] = typer.Option(None, help="Folder with pages/ and gt/ (overrides --split)."),
    out: Optional[Path] = typer.Option(None, help="Output folder (default data/lines/<split>)."),
    workers: int = typer.Option(6),
    limit: Optional[int] = typer.Option(None, help="Use only the first N pages of the split."),
):
    """Cut every ground-truth text line into a packed 64 px line dataset for CRNN training."""
    from .recognise.data import build_lines
    c = cfg()
    src = data or Path(c.paths.data_dir) / "synjama" / split
    dst = out or Path(c.paths.data_dir) / "lines" / src.name
    m = build_lines(src, dst, c, workers=workers, limit=limit)
    typer.echo(f"{m['lines']} lines -> {dst}")


@app.command("train-crnn")
def train_crnn(
    stage: str = typer.Option("B", help="A (public handwriting), B (SynJama) or C (HandFilled)."),
    data: Path = typer.Option(Path("data/lines/train"), help="Packed training lines (lre build-lines)."),
    val: Path = typer.Option(Path("data/lines/val"), help="Packed validation lines."),
    epochs: int = typer.Option(15),
    init: Optional[Path] = typer.Option(None, help="Checkpoint of the previous stage."),
    out: Path = typer.Option(Path("models/crnn_B.pt")),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    max_minutes: Optional[float] = typer.Option(None, help="Stop after this many minutes (keeps the best epoch)."),
):
    """Train the CRNN recogniser (report 5.3.2, D.8)."""
    from .recognise.train import train
    if not (data / "index.json").exists():
        raise typer.BadParameter(f"{data} is not a packed line set: run `lre build-lines` first "
                                 "(stage A needs IIIT-HW-Dev / IAM converted the same way)")
    train(stage, data, val, out, epochs=epochs, init=init, batch_size=batch_size, lr=lr, seed=seed,
          max_minutes=max_minutes, log=lambda s: typer.echo(s))


@app.command("train-unet")
def train_unet(
    data: Path = typer.Option(Path("data/synjama/train"), help="Training split."),
    val: Path = typer.Option(Path("data/synjama/val"), help="Validation split."),
    steps: int = typer.Option(1500, help="Optimiser steps (batches of 8 tiles)."),
    pages: int = typer.Option(120, help="Training pages held in memory."),
    out: Path = typer.Option(Path("models/unet_v1.pt"), help="Output checkpoint (a .card.json is written beside it)."),
    seed: int = typer.Option(0),
):
    """Train the U-Net-lite ruling-line segmenter on SynJama line masks."""
    from .layout.unet import train
    card = train(data, val, cfg(), out, steps=steps, n_train_pages=pages, seed=seed, log=typer.echo)
    typer.echo(f"saved {out} ({card['params']} parameters, {card['minutes']} min); val IoU {card['val_iou']}")


@app.command("train-crf")
def train_crf_cmd(
    data: Path = typer.Option(Path("data/synjama/train"), help="SynJama split with gt/."),
    pages: int = typer.Option(400, help="Pages to take owner cells from."),
    out: Path = typer.Option(Path("models/crf_owner_v1.joblib")),
):
    """Train the CRF owner-cell tagger on SynJama owner cells with OCR-style noise (report 5.3.4)."""
    from .extract.crf_owner import train_crf
    n = train_crf([data], out, limit_pages=pages)
    typer.echo(f"trained on {n} sequences -> {out}")


@app.command("train-scorer")
def train_scorer():
    """Train the gradient-boosted contradiction scorer (runs inside `evaluate --stage proposed_b`)."""
    typer.echo("The scorer is fitted on the CRNN system's train-page flags by "
               "`lre evaluate --stage proposed_b --split test` (writes models/gbm_v1.joblib).")


@app.command()
def calibrate():
    """Fit per-field isotonic calibrators and thresholds (runs inside `evaluate --stage proposed_b`)."""
    typer.echo("Calibrators and thresholds are fitted on val by "
               "`lre evaluate --stage proposed_b --split test` (writes models/calibrators_v1.joblib).")


if __name__ == "__main__":
    app()
