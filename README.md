# Legacy Land Record Digitisation and Validation Engine (CA-1, Group [G No])

Members: Chayan Khatua (12518616), Aditya Kumar Singh (12509260)

## 1. What it does

Reads a scanned Jamabandi-style land register page, extracts structured fields for every
holding, and flags entries that contradict the existing digital register for human review.
Inspired by Smart India Hackathon 2026 problem **SIH26018** (Intelligent Land Record
Digitization and Validation System, Ministry of Rural Development): https://sih.gov.in/sih2026PS

## 2. Install

Python **3.11** is required (PyTorch 2.4 has no wheels for newer Pythons).

```bash
python3.11 -m venv .venv            # or: conda create -n lre311 python=3.11 && <env>/bin/python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                    # installs the `lre` package and command
lre fonts                           # downloads the OFL Devanagari fonts SynJama renders with
pytest -q                           # all tests green before any commit
```

Rendering Devanagari needs Pillow with **libraqm** (`python -c "from PIL import features; print(features.check('raqm'))"`
must print `True`). macOS: `brew install libraqm fribidi`; Ubuntu: `sudo apt install libraqm0`.

Tesseract 5 with Hindi + English is needed for the baseline only (Phase 3):
Ubuntu `sudo apt install tesseract-ocr tesseract-ocr-hin`; macOS `brew install tesseract tesseract-lang`.

Dataset download steps (IAM, DHCD, IIIT-HW-Dev) are in [data/README_data.md](data/README_data.md).

## 3. Quick demo

```bash
lre synth --split sample --pages 5 --out data/sample     # 5 synthetic pages + register.csv
lre validate --holdings data/sample/holdings_gt.jsonl --register data/sample/register.csv
                                                          # -> results/contradiction_report.csv
lre run --village V000 --pages data/sample/pages --register data/sample/register.csv
                                                          # extract + validate + route into results/lre.db
                                                          # (baseline extractor until the CRNN is trained)
streamlit run app/reviewer.py                             # review queue: scan beside fields
```

## 4. Reproduce the report

```bash
lre synth --pages 1000 --out data/synjama --workers 6    # SynJama train / val / test (about 9 min, 6 GB)
lre evaluate --stage layout --split val                   # Phase 2 check: column count and cell IoU
lre train-unet                                            # U-Net-lite line segmenter (about 10 min on Apple MPS)
lre evaluate --stage layout --split val --stress --unet models/unet_v1.pt   # faded / broken rules
lre tessdata                                              # Tesseract hin + eng models for the baseline
lre evaluate --stage baseline --split test --workers 6    # template + Tesseract baseline (Table 11 row 1)
lre evaluate --stage validation --split test              # Phase 4 check: rules on ground-truth fields
lre build-lines --split train   # (also val, test)        # 64 px line crops for the CRNN
lre train-crnn --stage B --epochs 15                      # CRNN on SynJama lines (stage A needs IIIT-HW-Dev / IAM)
lre train-crf                                             # CRF owner-cell tagger
lre evaluate --stage proposed --split train --limit 150   # CRNN system output on train pages (scorer training data)
lre evaluate --stage proposed --split val                 # Proposed A on val (theta / calibration data)
lre evaluate --stage proposed --split test                # Proposed A (Table 11 row 2)
lre evaluate --stage proposed_b --split test              # Proposed B: + GBM scorer + calibrated routing (row 3)
lre evaluate --split test                                 # (Phase 7) writes results/metrics.json, results/tables/
```

## 5. Train from scratch

On Google Colab: open `notebooks/04_colab_training.ipynb` (T4 GPU) and run the cells in order;
it regenerates the data, trains the CRNN and evaluates Proposed A / B against the baseline.
Locally the same commands work (see section 4); the CRNN needs a GPU to train in reasonable time.

## 6. Build status

| Phase | Goal | Status |
|---|---|---|
| 0 Scaffold | runnable empty project | done |
| 1 Data | SynJama generator | done (public datasets: manual download, see data/README_data.md) |
| 2 Preprocess + layout | pages become cell crops | done: classical chain + U-Net-lite fallback (`lre train-unet`, `lre evaluate --stage layout [--stress --unet ...]`) |
| 3 Baseline | template + Tesseract baseline, evaluation harness | done (`lre tessdata`, `lre evaluate --stage baseline`) |
| 4 Validation | knowledge base and rule engine | done (`lre validate`, `lre evaluate --stage validation`) |
| 5 AI extractor | CRNN, CRF, scorer, calibration | code done; CRNN stage B trained locally (stage A / C need IIIT-HW-Dev, IAM, HandFilled) |
| 6 Reviewer app | Streamlit reviewer | done (`lre run` -> `results/lre.db` -> `streamlit run app/reviewer.py`) |
| 7 Reports | final numbers | not started |

## 7. Licences and data privacy

* Code: ours. SynJama pages are synthetic; all names are fictitious and were never taken from real records.
* Fonts: Kalam, Amita, Tillana, Hind (SIL Open Font License 1.1, Google Fonts).
* IAM is **non-commercial research only**: any deployed model must be retrained without it.
* DHCD: CC BY 4.0. IIIT-HW-Dev / IIIT-INDIC-HW-WORDS: research use, cite the papers.
* The pipeline runs offline; names are masked in logs; caste identifiers are never extracted.
