# Data

Nothing in `data/raw/`, `data/synjama/` or `data/handfilled/` is committed to git.

| Folder | Content | How to get it |
|---|---|---|
| `data/fonts/` | OFL handwriting-style Devanagari fonts | `lre fonts` |
| `data/sample/` | 5 synthetic pages + `register.csv` for the demo | `lre synth --split sample --pages 5 --out data/sample` |
| `data/synjama/{train,val,test}/` | SynJama, 1,000 pages split by village | `lre synth --pages 1000 --out data/synjama --workers 4` |
| `data/raw/iam/` | IAM Handwriting DB 3.0 | manual, below |
| `data/raw/dhcd/` | Devanagari Handwritten Character Dataset | manual or scripted, below |
| `data/raw/iiit-hw-dev/` | IIIT-HW-Dev word images | manual, below |
| `data/handfilled/` | consented volunteer pages (report D.7) | collected by the team |

## Public datasets

**IAM Handwriting Database 3.0** (non-commercial research only). Register at
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database, then download `words.tgz`,
`lines.tgz` and `ascii.tgz` into `data/raw/iam/` and extract them there. Use the standard
writer-independent split (Aachen partition).

**DHCD** (CC BY 4.0, doi:10.24432/C5XS53). Download the zip from the UCI repository page
(https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) and extract
to `data/raw/dhcd/` so that `data/raw/dhcd/Train/digit_0/` exists.

**IIIT-HW-Dev** (research use). Request access on
https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data, download version 1
(about 1.8 GB) into `data/raw/iiit-hw-dev/` and keep the dataset's own train / val / test lists.
Read the Readme in the download and cite Dutta et al., DAS 2018.

**IIIT-INDIC-HW-WORDS (Gurmukhi)**, optional, same page as above.

## SynJama files

Per split folder:

* `pages/<page_id>.png` - degraded page (300 dpi, 4200 x 3000 px by default).
* `gt/<page_id>.json` - ground truth. Boxes are in the **deskewed frame**: the scanned image
  rotated back by `skew_deg` about its centre. `transform.matrix_form_to_scan` maps the clean
  form frame to the scanned image if you need boxes in the scan itself. Each holding has its true
  field values (report Table 4), a `row_bbox`, and 12 `cells`, each with its text `lines`
  (text + box): these are the CRNN training crops.
* `register.csv` - the digital register for every village in the split (one row per khewat),
  with injected contradictions and benign variants.
* `labels.csv` - every injected change: `kind` is `contradiction` (register disagrees with the
  page, expected rule in `rules`), `intra` (the page itself is inconsistent, R1/R2), `benign`
  (must NOT be flagged) or `renumber` (khewat renumbered: the linker must fall back to khasra
  overlap). `holding_id` is empty for holdings that are in the register but on no generated page.
* `holdings_gt.jsonl` - the true holdings as written on the pages, one per line.
* `manifest.json` - counts, seed, config hash, fonts and label counts.

The test split is rendered with fonts that never appear in train / val (`synth.fonts.test` in
config.yaml) and its own seed stream.
