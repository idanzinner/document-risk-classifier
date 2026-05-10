# Hebrew PDF Hallucination-Risk Classifier

A document-image classifier that assigns each page of a Hebrew PDF one of three
risk categories for downstream LLM-based extraction:

| Category | Meaning |
|---|---|
| `safe_for_extraction` | Page is clean and structured — low hallucination risk |
| `review` | Page is borderline — route to human review |
| `high_hallucination_risk` | Dense handwriting, poor scan, or complex layout — high risk |

---

## Goal

PDF pages fed to an LLM for Hebrew text extraction vary widely in quality.
Some pages contain dense handwriting, degraded scans, or complex layouts that
cause the LLM to hallucinate.  This classifier acts as a pre-filter:
pages flagged as `high_hallucination_risk` are blocked from extraction (or
routed to a human), reducing downstream hallucination in the extracted output.

The model is trained on rendered page images (224×224 greyscale PNG) using
binary `BCEWithLogitsLoss`.  Ternary output is produced post-hoc via
temperature-scaled calibration and threshold optimisation that targets a
false-safe rate of ≤ 5 %.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## Usage

### Local setup (run once)

Open `notebooks/00_gcp_setup.ipynb` and run all cells. It validates the directory
structure, cleans labels, generates splits, renders PDFs to `data/rendered_pages/`,
and spot-checks the output.

### Building the training dataset from scratch (Phase 6 pipeline)

```bash
# 1. Enumerate all source folders and build metadata.csv
python scripts/build_metadata_v2.py

# 2. Render all PDFs to 224×224 grayscale PNGs (skip-existing)
python scripts/render_all_pages.py

# 3. Generate grouped train/val/test splits (grouped by source_doc_stem)
python scripts/regenerate_splits.py

# 4. Annotate rubric scores via Claude vision API (incremental)
python scripts/annotate_rubric.py
```

### Training

```bash
# ResNet50
python -m src.train.train_baseline --config configs/baseline.yaml

# ViT-Base
python -m src.train.train_baseline --config configs/baseline.yaml --model vit_base_patch16_224

# DiT (3-stage: head-only → top-2 blocks → full fine-tune)
python -m src.train.train_dit --config configs/dit.yaml
```

After training, open the corresponding notebook (04 or 05) in **eval-only mode** —
load the checkpoint and run all cells to see metrics and plots.

### Cost-weighted threshold recalibration

```bash
# Default: FN:FP = 10:1  (missing risky is 10× worse than flagging safe)
python scripts/retune_thresholds.py

# Softer ratio
python scripts/retune_thresholds.py --fn-cost 5 --fp-cost 1
```

### External validation set inference

```bash
python scripts/_run_validation_inference.py
# Outputs: validation_report/index.html + validation_report/report_data.json
```

### Regenerate reports

```bash
# Must regenerate validation_report/ first (see above)
python scripts/build_finetune_report.py       # → reports/finetune_report.html
python scripts/build_mixed_eval_report.py     # → reports/mixed_eval_report.html
```

### Run inference on new PDFs

```bash
python -m src.inference.predict \
    --config configs/inference.yaml \
    --pdf_path path/to/document.pdf \
    --page_num 1
```

### Special-font OCR (Apple Silicon only)

```bash
# 1. Render source PDFs at 300 DPI for OCR
python scripts/split_pdfs_to_pages.py

# 2. Open notebooks/09_special_font_ocr.ipynb and run all cells
#    (requires mlx-vlm + Gemma 4 26B 4-bit, ~36 GB unified memory)
```

---

## Repository Structure

```
for_tal/
├── data/
│   ├── handwritten/                       # source PDFs — handwritten questionnaires
│   ├── handwritten_and_questioniers/      # source PDFs — mixed handwritten + typed
│   ├── handwritten_edge_cases/            # source PDFs — edge-case handwritten pages
│   ├── regular_forms/                     # source PDFs — standard typed forms
│   ├── regular_forms_edge_cases/          # source PDFs — edge-case regular forms (was spaciel_font)
│   ├── spaciel_font/                      # source PDFs — special-font scans (OCR track)
│   ├── validation_set/                    # external validation PDFs (400 docs)
│   ├── rendered_pages/                    # 224×224 grayscale PNGs from render_pdf.py
│   ├── splits/                            # train.csv / val.csv / test.csv (generated)
│   ├── splits_v1/, splits_v2_precorrection/  # pre-correction split backups
│   ├── metadata.csv                       # master page-level index + labels (1873 rows)
│   ├── metadata_v1.csv                    # Phase 5 backup (1014 rows)
│   ├── metadata_v2_precorrection.csv      # pre-Phase-6d backup (1873 rows)
│   ├── labels_binary.csv                  # original binary annotation file
│   ├── labels_binary_clean.csv            # cleaned binary labels (schema-conformant)
│   ├── labels_rubric.csv                  # full D/H/S/L rubric annotations
│   ├── rubric_checkpoint.jsonl            # Claude vision API annotation checkpoint
│   ├── validation_results_full.csv        # per-doc external validation output
│   ├── handwritten.xlsx                   # ground-truth source spreadsheet
│   ├── regular_forms.xlsx                 # ground-truth source spreadsheet
│   ├── handwritten_validation_set.xlsx    # external validation ground truth (risky)
│   └── regular_documents_validation_set.xlsx  # external validation ground truth (safe)
├── src/
│   ├── data/                              # rendering, dataset, split utilities
│   ├── models/                            # ResNet, ViT, DiT classifiers + calibrator
│   ├── train/                             # training loops + evaluation
│   ├── inference/                         # prediction entry point + service schema
│   └── utils/                             # metrics, logging, visualization, device helpers
├── scripts/
│   ├── clean_labels.py                    # Phase 0 — label cleaning + metadata bootstrap
│   ├── build_metadata_v2.py               # Phase 6 — multi-folder metadata builder
│   ├── render_all_pages.py                # Phase 6 — batch PDF→PNG renderer
│   ├── regenerate_splits.py               # Phase 6 — grouped train/val/test split generator
│   ├── annotate_rubric.py                 # Phase 4 — Claude vision API rubric annotator
│   ├── retune_thresholds.py               # Phase 6c — cost-weighted threshold recalibration
│   ├── split_pdfs_to_pages.py             # Phase 5d — 300 DPI per-page renderer for OCR
│   ├── build_finetune_report.py           # Phase 6b — regenerates reports/finetune_report.html
│   ├── build_mixed_eval_report.py         # — regenerates reports/mixed_eval_report.html
│   ├── _run_validation_inference.py       # Phase 5c — regenerates validation_report/
│   ├── fix_spaciel_font_to_edge_cases.py  # Phase 6d — one-shot metadata relabel
│   └── drive_to_gcs.py                    # optional GCS upload (not part of local workflow)
├── notebooks/
│   ├── 00_gcp_setup.ipynb                 # local setup: dirs, label cleaning, splits, rendering
│   ├── 01_data_audit.ipynb                # PDF discovery + metadata construction
│   ├── 02_rendering_checks.ipynb          # DPI / aspect-ratio / cache validation
│   ├── 03_label_consistency.ipynb         # rubric distribution + label consistency
│   ├── 04_baseline_training.ipynb         # ResNet50 + ViT-Base evaluation (eval-only)
│   ├── 05_dit_training.ipynb              # DiT 3-stage evaluation (eval-only)
│   ├── 06_calibration_eval.ipynb          # calibration curves + threshold sensitivity
│   ├── 07_inference_demo.ipynb            # end-to-end PDF → risk-category demo
│   ├── 08_validation_inference.ipynb      # external validation set (400 docs, 3 models)
│   └── 09_special_font_ocr.ipynb          # Gemma 4 OCR for special-font scans
├── configs/
│   ├── baseline.yaml                      # ResNet/ViT training config
│   ├── dit.yaml                           # DiT staged training config
│   └── inference.yaml                     # inference thresholds + model paths
├── eval/
│   ├── evnaluation.py                     # Gemini vs DiT evaluation script
│   └── mixed_results - Sheet1.csv         # 648-doc ground-truth evaluation CSV
├── reports/
│   ├── finetune_report.html               # Phase 6 interactive fine-tune report
│   ├── mixed_eval_report.html             # Gemini vs DiT comparison report
│   └── _finetune_report_template.html     # HTML/JS template for build_finetune_report.py
├── validation_report/
│   ├── index.html                         # external validation interactive report
│   └── report_data.json                   # per-model plot data for the report
├── plots/                                 # saved training curve / metric PNGs
├── docs/
│   ├── PROJECT_STATUS.md                  # pipeline progress and cross-step knowledge
│   ├── INTERFACES.md                      # data schemas and module API contracts
│   └── FINETUNE_REPORT.md                 # plain-text fine-tune summary for Slack/email
└── requirements.txt
```

---

## Reports

Three self-contained HTML reports are included for sharing results without any
server or local assets:

- [`reports/finetune_report.html`](reports/finetune_report.html) — interactive
  Phase 6/6c/6d fine-tune report with tabs for training curves, held-out test
  metrics, external validation (400 docs), and Phase 5 vs Phase 6 FSR comparison.
  See also the plain-text companion [`docs/FINETUNE_REPORT.md`](docs/FINETUNE_REPORT.md).
- [`reports/mixed_eval_report.html`](reports/mixed_eval_report.html) — Gemini
  (text-based LLM) vs DiT (fine-tuned vision classifier) comparison on a 648-doc
  evaluation set with ground-truth labels.
- [`validation_report/index.html`](validation_report/index.html) — per-model
  external validation report with a model-selector dropdown.

---

## Key Design Decisions

- **Grouped splits**: pages from the same institution are kept together in one
  split to prevent data leakage.
- **Binary training, ternary output**: the model outputs a single logit;
  the ternary labels are derived via calibrated thresholds selected to meet
  the false-safe rate constraint.
- **Most important metric**: false-safe rate — risky pages predicted as safe.
  Target: ≤ 5 %.
- **Deterministic preprocessing**: rendering parameters are versioned in
  `configs/` and never changed after the first rendering run.
