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

### 1. Render PDFs to images

```bash
python -m src.data.render_pdf \
    --pdf_dir data/ \
    --output_dir data/rendered_pages/ \
    --dpi 150
```

### 2. Create train/val/test splits

```bash
python -m src.data.splits \
    --metadata data/metadata.csv \
    --output_dir data/splits/
```

### 3. Train baselines

```bash
python -m src.train.train_baseline --config configs/baseline.yaml
```

### 4. Train DiT (staged fine-tuning)

```bash
python -m src.train.train_dit --config configs/dit.yaml
```

### 5. Calibrate and evaluate

```bash
python -m src.train.evaluate \
    --config configs/dit.yaml \
    --checkpoint checkpoints/dit/best_model.pt
```

### 6. Run inference

```bash
python -m src.inference.predict \
    --config configs/inference.yaml \
    --pdf_path path/to/document.pdf \
    --page_num 1
```

---

## Repository Structure

```
for_tal/
├── data/
│   ├── pdf_s - שאלונים וכתב יד /   # source PDFs — questionnaires + handwriting
│   ├── מסמכים רגילים /              # source PDFs — regular typed documents
│   ├── rendered_pages/              # output PNGs from render_pdf.py
│   ├── splits/                      # train.csv / val.csv / test.csv
│   ├── metadata.csv                 # master page-level index + labels
│   ├── labels_binary.csv            # minimal binary annotation file
│   └── labels_rubric.csv            # full D/H/S/L rubric annotation file
├── src/
│   ├── data/                        # rendering, dataset, split utilities
│   ├── models/                      # ResNet, ViT, DiT classifiers + calibrator
│   ├── train/                       # training loops + evaluation
│   ├── inference/                   # prediction entry point + service schema
│   └── utils/                       # metrics, logging, visualization
├── configs/
│   ├── baseline.yaml                # ResNet/ViT training config
│   ├── dit.yaml                     # DiT staged training config
│   └── inference.yaml               # inference thresholds + model paths
├── notebooks/                       # EDA, training, and analysis notebooks
├── docs/
│   ├── PROJECT_STATUS.md            # pipeline progress and cross-step knowledge
│   └── INTERFACES.md                # data schemas and module API contracts
└── requirements.txt
```

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
