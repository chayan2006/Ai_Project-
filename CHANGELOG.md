# Changelog

One entry per build phase (report Appendix D.3). A phase is closed only when its check passes.

## Phase 5 - AI extractor (2026-11-06, code complete; training moves to Colab)

* `src/lre/recognise/`: charset (158 symbols + CTC blank, 100 % coverage of SynJama labels);
  CRNN (6 conv + BN, 2 x BiLSTM 256, CTC; 7.7 M parameters); packed 64 px line datasets
  (`lre build-lines`, memory-mapped); training with Table 6 settings, mixed precision on CUDA,
  early stopping and model cards (`lre train-crnn`); field-aware CTC beam search with digit-only
  alphabets, khasra / kanal-marla grammars and village-register lexicon rescoring (`decode.py`);
  the proposed extractor (grid + U-Net fallback -> lines -> CRNN -> decoding -> holdings).
* `src/lre/extract/crf_owner.py` (`lre train-crf`): CRF owner tagger trained on SynJama owner cells
  with OCR-style noise. Val owner cells: clean text 100 % (regex 100 %); noisy text 63.0 % of owners
  fully right (regex 53.8 %).
* `src/lre/validate/scorer.py`: gradient-boosted contradiction scorer (Table 7 features, F0.5 theta).
* `src/lre/route/calibrate.py`: per-field isotonic calibration, thresholds for alpha = 0.98, ECE.
* `lre evaluate --stage proposed` (Proposed A) and `--stage proposed_b` (Proposed B);
  `lre run --extractor crnn` uses the trained CRNN in the full pipeline and the reviewer.
* `notebooks/04_colab_training.ipynb` + `requirements-colab.txt`: all training and the Table 11
  evaluation on a Colab T4, with data and checkpoints kept in Google Drive.
* Not done: CRNN stage A (IIIT-HW-Dev / IAM need a converter to the packed line format, and the
  datasets need registration), stage C (needs HandFilled pages), the trained CRNN itself and the
  Proposed A / B numbers (Colab). The U-Net and the CRF are already trained locally.
* `requires-python` relaxed to >= 3.11 so the package installs on Colab's newer Python.
* `lre build-lines` denoises each line crop (with a 12 px margin) instead of the whole page:
  same lines and shapes, 0-4 grey levels apart, 3.1x faster (6.8 s -> 2.2 s per page locally;
  ~15 s -> ~5 s per page on Colab's 2 CPUs). New `--limit N` uses the first N pages; the Colab
  notebook builds lines from 600 of the 800 train pages and zips pages and lines to Drive separately.

## Phase 6 - Reviewer app (2026-11-06)

* `src/lre/store.py`: SQLite store with the tables of report Figure 9 (page, register_row, holding,
  field_extraction, flag, review_decision, audit_log). Every decision is mirrored into an
  append-only audit log in which each entry hashes the previous one; `verify_audit_chain` finds
  the first tampered entry.
* `lre run --village V --pages DIR --register CSV`: ingest (duplicate pages rejected by SHA-256),
  preprocess, extract, link, validate, route, store; stage checkpoints and per-cell crops in
  `data/interim/<page_id>/`. Until the CRNN exists the extractor is the template baseline, validated
  with the `exact` rule profile, so on the sample every holding is routed to CONTRADICTION.
* `src/lre/route/router.py`: AUTO / REVIEW / CONTRADICTION decision shared by `lre run` and the
  evaluation (calibrated per-field thresholds arrive with Phase 5).
* `app/reviewer.py` (report 6.3): village and queue selection (contradictions first), the deskewed
  scan with the holding and the selected cell highlighted, fields beside the linked register values
  with confidence, cell zoom, accept / correct / mark OCR error, confirm (escalate) or dismiss each
  flag with its evidence, finish holding, export of the contradiction report and of corrections
  (with cell crops, for fine-tuning), audit-chain status.
* Check (report D.3: a full review session works, corrections saved in SQLite): `tests/test_reviewer.py`
  drives the app headlessly (Streamlit AppTest) through accept, correct, OCR error, escalate and
  finish, then checks the decisions, exports and audit chain. The app also runs on the sample store.
* `pyarrow` pinned to 17.0.0: newer versions need NumPy 2 and broke `st.dataframe`.

## Phase 4 - Validation (2026-11-06)

* `rules.yaml`: rules R1-R10 (report Table 7) as declarative checks over facts, with parameter
  profiles (`default`; `exact` for the template baseline: exact string and number equality).
  Check expressions are parsed into a restricted syntax tree, so the YAML cannot run code.
* `src/lre/validate/kb.py`: typed facts for holdings, owners, plots, cultivator, mutations and the
  register's khasra index; derivation rules run to a fixpoint (forward chaining), e.g. share sum,
  Hungarian owner matching on name similarity, plot pairs, khasra held by another khewat.
* `src/lre/validate/engine.py`: fires the checks and stores each fact as evidence; high / medium
  flags make a CONTRADICTION, low ones (R4 vocabulary) only REVIEW.
* `src/lre/link/linker.py`: khewat blocking; khasra-Jaccard fallback for renumbered khewats; a
  khewat match sharing no khasra while another row matches strongly is treated as a misread number.
* `lre validate` writes `results/contradiction_report.csv` (Objective 2): rule, severity and evidence
  per flagged holding. `lre evaluate --stage validation` checks the rules against SynJama labels.
* Check (report D.3: on ground-truth fields every injected contradiction found, precision 1.0):
  val 105/105 found, 0 false alarms; test 108/108, 0 false alarms; link accuracy 100 % on both,
  including 80 renumbered khewats. Benign variants (spelling, digit script, relation words) never fire.
* Found on the way: a holding that was both renumbered and given a new owner was refused by the
  linker (owner confirmation failed). Khasras identify parcels, so an overlap >= 0.8 now links
  without owner confirmation (`linking.strong_khasra_jaccard`).
* One unit test per rule with a passing and a failing case (`tests/test_validate.py`).
* Rule thresholds moved from config.yaml to rules.yaml.

## Phase 3 - Baseline and evaluation harness (2026-11-06)

* `src/lre/baseline/template_tesseract.py`: the template baseline of report 5.2 - column
  boundaries measured once on a blank form (`models/baseline_template.json`), rows by horizontal
  projection, Tesseract 5 LSTM `hin+eng --psm 6` per cell, regular-expression parsing, rules with
  the `exact` profile, review whenever a word confidence is below 60.
* `src/lre/extract/holding.py`: cell texts -> structured holding (shared with the proposed system;
  the owner parser is pluggable for the CRF). An unreadable share is left empty rather than `1/1`.
* `src/lre/evaluate/metrics.py`: per-field accuracy (Table 14 fields), name / remarks CER, review
  share, auto-accept accuracy, holding-level contradiction precision / recall (Table 10).
* `lre tessdata` downloads hin + eng models into `models/tessdata/`;
  `lre evaluate --stage baseline` writes `results/baseline_metrics.json` and
  `results/tables/baseline_per_field_<split>.csv`.
* Results (all holdings routed to review: every one has a word below confidence 60):

  | split | holdings | mean field acc. | numeric acc. | name CER | contr. precision | contr. recall |
  |---|---|---|---|---|---|---|
  | val (train fonts) | 720 | 0.285 | 0.190 | 1.09 | 0.127 | 0.971 |
  | test (held-out fonts) | 751 | 0.225 | 0.221 | 1.28 | 0.128 | 0.954 |

  Tesseract reads the handwritten `1` as a danda and `खुदकाश्त` as Latin words; exact-equality rules
  then flag nearly every holding, so contradiction precision stays at the base rate (hypotheses H1, H3).
  Name CER exceeds 1 where a misread relation word makes the regex parser swallow the whole entry.
* The full SynJama set was generated: train 800 / val 100 / test 100 pages (villages V000-V065).

## Phase 2 - Preprocess + layout (2026-11-06)

* `src/lre/preprocess/`: page loading (images; PDF via pdf2image, needs poppler), illumination
  correction + CLAHE, non-local-means denoising, Hough deskew, Sauvola binarisation.
* `src/lre/layout/grid.py`: morphological grid detector with sanity checks (column count, row
  height) that will trigger the U-Net fallback. Candidate rules must be dark in the grayscale
  image, so bleed-through of the reverse side's printed form is not taken for a rule.
* `src/lre/layout/cells.py`: cell crops (empty rows skipped), projection-profile line splitting,
  64 px height normalisation with width buckets.
* `lre evaluate --stage layout`: acceptance check against SynJama ground truth.
* Check (report D.3: columns correct on >= 95 % of pages, mean cell IoU >= 0.90), on freshly
  generated pages: val 40 pages - columns 100 %, mean cell IoU 0.997, rows 100 %, mean deskew
  error 0.05 deg; test 20 pages (held-out fonts) - columns 100 %, cell IoU 0.996, rows 100 %.
  Text-line split matches the ground-truth line count on 95 % of cells.
* About 2 s per page on the laptop CPU, most of it NL-means denoising (target: >= 2 pages/min).
* U-Net-lite (`src/lre/layout/unet.py`, `lre train-unet`): 482 k parameters, 4 levels (16-128),
  512 px tiles at half resolution, BCE + Dice, masks drawn from SynJama ground truth, augmentation that
  fades and breaks the rules. 1,500 steps on 120 train pages took 24 min on Apple MPS; pixel IoU on
  augmented val tiles 0.80 (horizontal) / 0.88 (vertical). Model card: `models/unet_v1.card.json`.
* It runs only when the morphological grid fails its check. Stress test (`--stress`: rules of real val
  pages faded and broken, 20 pages): morphology alone recovers 0 / 20 grids; with the U-Net fallback
  16 / 20, at about 0.99 cell IoU on the recovered pages. The other 4 still miss 1-2 faded column rules
  and, per report E.11, go to manual entry. Possible next step: complete the grid from the known form
  template when only a few rules are missing.

## Phase 1 - Data (2026-11-06)

* `lre synth`: SynJama generator (`src/lre/synth/`): village sampler, register perturbation with
  labels, page renderer, scan degradation, train / val / test split by village with held-out test
  fonts and seeds.
* Fonts: Kalam (Light/Regular/Bold) and Amita for train / val, Tillana (Regular/Medium) held out
  for test, Hind for the printed form. `lre fonts` downloads them.
* Deviations from the report:
  * split is **by village** rather than by page, so one village's register rows never span two splits;
  * a `cultivator_changed` contradiction (R9) was added to the D.6 list so every cross rule has positives;
  * 2 % of pages carry an intra-record error (shares not summing to one, wrong written total)
    so that R1 / R2 have positives;
  * 5 % of khewats are renumbered in the digital register to exercise the linker fallback;
  * DHCD digit insertion (D.6 step 2) is not done yet; it needs the DHCD download.
* Public datasets (IAM, IIIT-HW-Dev) need registration and are downloaded by hand
  (data/README_data.md).

## Phase 0 - Scaffold (2026-11-06)

* Package layout as in Appendix B; `lre` CLI (typer) with every command from Table 20
  (commands of later phases exit with a "planned for Phase N" message).
* `config.yaml` + typed loader that rejects unknown keys; `schema.py` data contracts with JSON
  round-trip; register CSV loader with schema check; name-masking log filter; stage checkpoints.
* `requirements.txt`: `sklearn-crfsuite` pinned to 0.5.0 because 0.5.2 (report D.2) does not
  exist on PyPI; Pillow and SciPy pinned explicitly.
