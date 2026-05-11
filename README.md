# Hebrew PDF Hallucination-Risk Classifier

A document-image classifier that assigns each page of a Hebrew PDF one of three
risk categories for downstream LLM-based text extraction:

| Category | Meaning |
|---|---|
| `safe_for_extraction` | Page is clean and structured — low hallucination risk |
| `review` | Page is borderline — route to human review |
| `high_hallucination_risk` | Dense handwriting, poor scan, or complex layout — high risk |

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Environment Setup](#3-environment-setup)
4. [Data Organisation](#4-data-organisation)
5. [Data Pipeline](#5-data-pipeline)
6. [Simple Two-Folder Pipeline (ADR-0001)](#6-simple-two-folder-pipeline-adr-0001)
7. [Training](#7-training)
8. [Checkpoints and Saved Artefacts](#8-checkpoints-and-saved-artefacts)
9. [Calibration and Thresholds](#9-calibration-and-thresholds)
10. [Evaluation](#10-evaluation)
11. [Inference on New PDFs](#11-inference-on-new-pdfs)
12. [Results Summary](#12-results-summary)
13. [Reports](#13-reports)
14. [Special-Font OCR Track](#14-special-font-ocr-track)
15. [Config Reference](#15-config-reference)
16. [Repository Structure](#16-repository-structure)
17. [Known Gotchas](#17-known-gotchas)
18. [Architecture Decision Records](#18-architecture-decision-records)

---

## 1. Motivation

PDF pages fed to an LLM for Hebrew text extraction vary widely in quality.
Pages with dense handwriting, degraded scans, or complex layouts cause the LLM to
hallucinate — fabricating names, dates, and amounts that are not in the document.

This classifier acts as a **pre-filter**: pages flagged as
`high_hallucination_risk` are blocked from extraction (or routed to a human
reviewer), reducing downstream hallucination in the extracted output.

The model is trained on rendered page images (224×224 grayscale PNG) using
binary `BCEWithLogitsLoss`. Ternary output is produced post-hoc via
temperature-scaled calibration and cost-weighted threshold optimisation.

**Most important metric:** false-safe rate (FSR) — risky pages that are
predicted safe. Missing a risky document is treated as 10× worse than
incorrectly flagging a safe one, so the decision threshold τ* minimises
`cost(τ) = 10·FN(τ) + 1·FP(τ)`.

---

## 2. Architecture Overview

Three models are implemented, all sharing the same binary training and ternary
output logic:

### ResNet50 / EfficientNet-B0 (`src/models/resnet_baseline.py`)

- timm-based CNN backbones, ImageNet pretrained
- Single linear head: 512 → 1 logit
- Grayscale input (mode L) is repeated to 3 channels in `forward()`
- Training: AdamW, lr=1e-4, early stopping patience=5

### ViT-Base (`src/models/vit_baseline.py`)

- `vit_base_patch16_224` from timm, ImageNet pretrained
- Same single-logit head pattern
- Tends to converge very fast on small datasets (best epoch often 2–5)

### DiT (`src/models/dit_classifier.py`) — **recommended production model**

- HuggingFace `microsoft/dit-base` — a BEiT-style transformer pretrained on
  IIT-CDIP scanned document images
- Three-stage fine-tuning:
  - **Stage 1:** backbone frozen, head only (lr=1e-3)
  - **Stage 2:** top 2 transformer blocks unfrozen (lr=1e-5)
  - **Stage 3:** full model (lr=5e-6)
- Each stage has independent early stopping (patience=15)
- The best checkpoint is the global best across all three stages
- `freeze_backbone()`, `unfreeze_top_blocks(n)`, `unfreeze_all()` helpers
- Staged unfreezing accesses `self.backbone.encoder.layer[-n:]`

All models output a single logit `[B, 1]`. Calibrated probabilities and
ternary labels come from the post-training `TemperatureCalibrator`.

---

## 3. Environment Setup

```bash
# Requires Python ≥ 3.10
pip install -r requirements.txt
```

Key dependencies: `torch>=2.1`, `timm>=0.9`, `transformers>=4.35`,
`pymupdf>=1.23` (PDF rendering), `pydantic>=2.0` (inference schema),
`scipy>=1.11` (calibration), `plotly>=5.18` (reports).

**Device selection** is automatic via `src/utils/device.get_device("auto")`:
CUDA → MPS (Apple Silicon) → CPU. Override in any config with
`training.device: cpu`.

**Apple Silicon note:** do NOT set `torch.channels_last` — timm backbones
crash on the MPS backward pass at PyTorch 2.11. The training scripts use
`mps_sync()` after `loss.backward()` and `mps_empty_cache()` after each epoch
to prevent Metal memory pressure.

GCP packages (`google-cloud-storage`, `gdown`, etc.) are commented out in
`requirements.txt` — they are only needed for the optional
`scripts/drive_to_gcs.py` upload workflow.

---

## 4. Data Organisation

### 4.1 Source folder layout

| Folder | Label | Description |
|---|---|---|
| `data/regular_forms/` | 0 (safe) | Standard typed Hebrew forms (634 pages) |
| `data/regular_forms_edge_cases/` | 0 (safe) | Visually unusual but typed forms (191 pages, formerly `spaciel_font`) |
| `data/handwritten/` | 1 (risky) | Handwritten questionnaires (386 pages) |
| `data/handwritten_and_questioniers/` | 1 (risky) | Mixed handwritten + typed (650 pages) |
| `data/handwritten_edge_cases/` | 1 (risky) | Edge-case handwritten pages (12 pages) |

**Total corpus:** 1,873 pages — 825 safe (label=0) / 1,048 risky (label=1).

### 4.2 CSV files

| File | Rows | Purpose |
|---|---|---|
| `data/metadata.csv` | 1,873 | Master index — all schema columns, split assignments |
| `data/labels_binary.csv` | 1,873 | Original binary annotation (file_path, label_binary) |
| `data/labels_binary_clean.csv` | 1,873 | Schema-normalised binary labels |
| `data/labels_rubric.csv` | 2,066 | D/H/S/L rubric scores (extra rows from multi-run checkpoint) |
| `data/rubric_checkpoint.jsonl` | 1,873 | Claude vision API annotation checkpoint for incremental runs |
| `data/validation_results_full.csv` | 400 | Per-document results from external validation inference |

**`data/metadata.csv` column schema:**

| Column | Type | Description |
|---|---|---|
| `file_path` | str | PNG filename (relative to `data/rendered_pages/`) — stored with `.pdf` extension, resolved to `.png` at dataset load time |
| `page_num` | int | 1-indexed page number within the source PDF |
| `institution` | str | `regular_docs` or `questionnaires` |
| `template_family` | str | Document template family (e.g. `regular_form`, `questionnaire_uuid`) |
| `label_binary` | int | `0` = safe, `1` = risky |
| `D` | int | Density: 0–3 (higher = less dense / easier to extract) |
| `H` | int | Handwriting: 0–3 (higher = more handwriting) |
| `S` | int | Scan quality: 0–3 (higher = worse scan) |
| `L` | int | Layout complexity: 0–3 (higher = more complex) |
| `risk_score` | int | `(3−D) + H + S + L` — range 0–12 |
| `split` | str | `train`, `val`, or `test` |
| `source_folder` | str | Source folder name (e.g. `regular_forms`, `handwritten`) |
| `source_doc_stem` | str | Source PDF stem — the split grouping key (prevents page leakage) |
| `is_edge_case` | bool | True for `regular_forms_edge_cases` pages |

### 4.3 Rendered images

All PDFs are pre-rendered to **224×224 grayscale PNG** at 150 DPI and stored
in `data/rendered_pages/`. This directory is gitignored (large) — regenerate
with `scripts/render_all_pages.py`.

Rendering is deterministic: same PDF + same config always produces the same PNG.
The parameters (DPI, size, grayscale) are versioned in `configs/baseline.yaml`
under `data:` and must never change once rendering has started.

---

## 5. Data Pipeline

Run these steps in order when setting up from scratch.

### Step 1 — Build metadata (first time only, or after adding new source folders)

```bash
python scripts/build_metadata_v2.py
```

Enumerates all 5 source folders, emits `data/metadata.csv` with all schema
columns. Backs up the previous metadata to `data/metadata_v1.csv` (or
`metadata_v2_precorrection.csv`) before overwriting.

### Step 2 — Render PDFs to PNGs

```bash
python scripts/render_all_pages.py
```

Renders every PDF in every source folder to `data/rendered_pages/` at
224×224 grayscale. Skip-existing is on by default — safe to re-run after
adding new PDFs.

**Note:** Pages from `data/regular_forms_edge_cases/` (formerly `spaciel_font`)
are sourced from pre-existing 300 DPI PNGs in `data/spaciel_font_pages/` and
downsampled — the script handles this automatically.

### Step 3 — Generate splits

```bash
python scripts/regenerate_splits.py
```

Groups pages by `source_doc_stem` (unique source PDF), then assigns whole groups
to train/val/test (70/15/15). Groups with fewer than 3 pages always go to train.
Saves `data/splits/train.csv`, `val.csv`, `test.csv`.

**Why source_doc_stem?** A single source PDF may contribute hundreds of pages.
If page 1 is in train and page 2 is in val, the model leaks visual features
across splits. Grouping by source document prevents this entirely.

### Step 4 — Annotate rubric scores (optional, already done)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/annotate_rubric.py
```

Sends each rendered PNG to Claude (`claude-sonnet-4-5`) via the vision API and
receives D/H/S/L scores. Results are written incrementally to
`data/rubric_checkpoint.jsonl` — interrupted runs resume automatically.
Outputs `data/labels_rubric.csv` and updates `data/metadata.csv`.

**Rubric already populated** for all 1,873 pages. Re-run only when new pages
are added.

### Shortcut — notebook orchestration

For a guided local setup, open `notebooks/00_gcp_setup.ipynb` and run all cells.
It validates directories, calls `clean_labels.py`, generates splits, renders
PDFs, and spot-checks the output.

---

## 6. Simple Two-Folder Pipeline (ADR-0001)

A **parallel** ingestion path that consumes only two top-level folders —
`data/safe/` and `data/risky/` — instead of the five legacy source folders
described above. The legacy pipeline is left fully intact; this one writes to
disjoint paths so both can coexist.

The decision, rejected alternatives, and contracts are recorded in
[`docs/decisions/0001-simple-two-folder-ingestion.md`](docs/decisions/0001-simple-two-folder-ingestion.md).

### 6.1 When to use it

| You want… | Use… |
|---|---|
| Document-level holdout, rich provenance (`source_doc_stem`, `template_family`, `institution`, `is_edge_case`) | Legacy pipeline (Section 5) |
| A fresh classifier from raw PDFs with the simplest possible folder layout | Simple pipeline (this section) |
| To reproduce Phase 5/6/6c/6d artefacts | Legacy pipeline (Section 5) |

### 6.2 Folder layout

```
data/
├── safe/                    # label_binary = 0
│   ├── foo.pdf
│   └── nested/bar.pdf       # arbitrary nesting allowed
└── risky/                   # label_binary = 1
    └── baz.pdf
```

PDFs may be nested arbitrarily under either root. Multi-page PDFs are expanded
to one row per page at ingest time.

### 6.3 Pipeline

Run in order. Stage 4 (rubric annotation) is optional.

```bash
# 1. Enumerate PDFs → one row per page → data/metadata_simple.csv
python scripts/build_metadata_simple.py

# 2. Render every page → data/rendered_pages_simple/*.png  (224x224 grayscale)
python scripts/render_pages_simple.py

# 3. (Optional) Annotate D/H/S/L/risk_score via Claude vision
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/annotate_rubric_simple.py --concurrency 10
# or, without an API key, run with synthetic scores:
python scripts/annotate_rubric_simple.py --dry-run

# 4. Stratified random 70/15/15 train/val/test split (by label_binary)
python scripts/make_splits_simple.py
```

All four scripts accept `--help` for path overrides. Re-runs are safe:
`build_metadata_simple` overwrites the metadata CSV deterministically,
`render_pages_simple` skips PNGs that already exist (`--force` to override),
and `annotate_rubric_simple` resumes from
`data/rubric_simple_checkpoint.jsonl`.

### 6.4 Metadata schema (`data/metadata_simple.csv`)

| Column | Description |
|---|---|
| `file_path` | Slug-based virtual filename ending in `.pdf` (resolves to a `.png` in `rendered_pages_simple/`) |
| `page_num` | 1-indexed page number within the source PDF |
| `label_binary` | `0` (safe) or `1` (risky) |
| `source_pdf` | Original relative path from `data/` — kept for traceability |
| `num_pages` | Total page count of the source PDF (denormalised) |
| `split` | `train`, `val`, or `test` (populated by `make_splits_simple.py`) |
| `D`, `H`, `S`, `L` | Rubric scores 0–3 (init `-1`, populated by `annotate_rubric_simple.py`) |
| `risk_score` | `(3 − D) + H + S + L`, range 0–12 (init `-1`) |

### 6.5 Unique-filename contract

`HallucinationRiskDataset` flattens rendered PNG paths to basename, so each
row's `file_path` is a globally unique slug built from the PDF's relative path:

```
data/safe/sub/foo.pdf  page 1
    → file_path  "safe__sub__foo__page_001.pdf"
    → rendered   data/rendered_pages_simple/safe__sub__foo__page_001.png
```

`build_metadata_simple.py` refuses to ingest any of the legacy folder names
(`regular_forms`, `handwritten`, `spaciel_font`, …) to prevent accidental
misuse of the wrong pipeline.

### 6.6 Training against the simple pipeline

Reuses the existing training scripts unchanged — only the config changes:

```bash
python -m src.train.train_baseline --config configs/baseline_simple.yaml
python -m src.train.train_dit       --config configs/dit_simple.yaml
```

Checkpoints land in `checkpoints/{baseline,dit}_simple/` and logs in
`logs/{baseline,dit}_simple/`. Inference loads them the same way as the legacy
checkpoints (Section 11).

### 6.7 Isolation guarantees

The simple pipeline writes to **disjoint** paths and never mutates legacy
artefacts:

| Concept | Legacy path | Simple path |
|---|---|---|
| Metadata CSV | `data/metadata.csv` | `data/metadata_simple.csv` |
| Rendered PNGs | `data/rendered_pages/` | `data/rendered_pages_simple/` |
| Splits | `data/splits/` | `data/splits_simple/` |
| Rubric labels | `data/labels_rubric.csv` | `data/labels_rubric_simple.csv` |
| Rubric checkpoint | `data/rubric_checkpoint.jsonl` | `data/rubric_simple_checkpoint.jsonl` |
| Model checkpoints | `checkpoints/{baseline,dit}/` | `checkpoints/{baseline,dit}_simple/` |
| Logs | `logs/{baseline,dit}/` | `logs/{baseline,dit}_simple/` |

Zero changes to `src/` are required to use the simple pipeline — the unmodified
`HallucinationRiskDataset` consumes its artefacts via the new configs.

### 6.8 Limitations

- **Random per-page split** (stratified by `label_binary`), not document-grouped.
  Multi-page source PDFs may have their pages split across train/val/test. If
  page leakage matters, use the legacy pipeline.
- **No `template_family` / `institution` / `is_edge_case` columns** — the
  schema is deliberately minimal. Reports/notebooks that reference those
  columns target the legacy `metadata.csv` only.

---

## 7. Training

All training commands must be run from the **repository root**.

### 7.1 ResNet50

```bash
python -m src.train.train_baseline --config configs/baseline.yaml
```

Produces:
- `checkpoints/baseline/best_resnet50.pt`
- `checkpoints/baseline/calibrator_resnet50.pkl`
- `logs/baseline/baseline_resnet50.json`

### 7.2 EfficientNet-B0

```bash
python -m src.train.train_baseline \
    --config configs/baseline.yaml \
    --model efficientnet_b0
```

The `--model` flag overrides `cfg["model"]["name"]` at runtime without editing
the YAML. Any timm model name works here.

### 7.3 ViT-Base

```bash
python -m src.train.train_baseline \
    --config configs/baseline.yaml \
    --model vit_base_patch16_224
```

Produces:
- `checkpoints/baseline/best_vit_base_patch16_224.pt`
- `checkpoints/baseline/calibrator_vit_base_patch16_224.pkl`
- `logs/baseline/baseline_vit_base_patch16_224.json`

ViT typically early-stops at epoch 2–5 on this dataset — that is expected and
genuine; it is not underfitting.

### 7.4 DiT (recommended)

```bash
python -m src.train.train_dit --config configs/dit.yaml
```

Produces:
- `checkpoints/dit/best_model.pt` — global best across all three stages
- `checkpoints/dit/calibrator.pkl`
- `logs/dit/dit.json`

The best checkpoint is always the one with the highest val F1 across all stages.
Stage 2 often outperforms stage 3 (the full fine-tune) due to early stopping.

### 7.5 What happens during training

For all models:

1. `HallucinationRiskDataset` is built for `train`, `val`, and `test` splits
2. `WeightedRandomSampler` upsamples the minority class each epoch
3. `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` further compensates for imbalance
4. Gradient clipping at `max_norm=1.0` on every batch
5. After each epoch: a `TemperatureCalibrator` is fit on the current val logits,
   thresholds are selected targeting FSR ≤ 5%, val F1 is computed
6. Best val F1 triggers a checkpoint save
7. Early stopping fires when `patience` consecutive epochs show no improvement
8. After training: the best checkpoint is reloaded and the model is evaluated on
   **all three splits** (train-no-aug, val, test) with the final calibrator.
   The checkpoint is then expanded with all logits/labels/metrics for downstream
   analysis in the evaluation notebooks.

### 7.6 Training augmentations

Applied only to the training split when `data.augmentation: true` in config:

- `RandomRotation(±3°)` — mimics scan skew
- `ColorJitter(brightness=0.2, contrast=0.2)` — scan exposure variation
- `GaussianBlur(kernel=3, σ=0.1–1.0)` — scan focus blur
- `RandomPerspective(distortion=0.05, p=0.3)` — slight physical warping

### 7.7 Overriding config values

The simplest override is `--model` (for baseline). For other values, edit the
YAML directly or pass a second YAML that overrides specific keys:

```bash
# Train with CPU only (e.g. for debugging)
# Edit configs/baseline.yaml: training.device: cpu

# Use a fixed pos_weight instead of auto
# Edit configs/baseline.yaml: training.pos_weight: 2.5
```

---

## 8. Checkpoints and Saved Artefacts

### 8.1 Checkpoint format

Every `.pt` checkpoint is a Python dict saved with `torch.save`. After the
final evaluation pass the dict contains:

| Key | Type | Description |
|---|---|---|
| `epoch` | int | Epoch of the best val F1 (baseline) or `"stage{n}_epoch{k}"` (DiT) |
| `model_state_dict` | OrderedDict | Model weights |
| `model_name` | str | timm or HuggingFace model identifier |
| `val_f1` | float | Best validation F1 |
| `temperature` | float | Fitted temperature scalar |
| `thresholds` | dict | `{'T_low': float, 'T_high': float}` |
| `config` | dict | Full YAML config used for this run |
| `train_logits` | np.ndarray | Raw logits on the training set (no augment) |
| `train_labels` | np.ndarray | Ground-truth labels for training set |
| `train_metrics` | dict | Computed metrics on training set |
| `val_logits` | np.ndarray | Raw logits on validation set |
| `val_labels` | np.ndarray | Ground-truth labels for validation set |
| `val_metrics` | dict | Computed metrics on validation set |
| `test_logits` | np.ndarray | Raw logits on test set |
| `test_labels` | np.ndarray | Ground-truth labels for test set |
| `test_metrics` | dict | Computed metrics on test set |

### 8.2 Loading a checkpoint manually

```python
import torch, pickle

ckpt = torch.load("checkpoints/dit/best_model.pt", map_location="cpu", weights_only=False)

# Access stored results without rerunning inference
test_logits  = ckpt["test_logits"]   # np.ndarray [N]
test_labels  = ckpt["test_labels"]   # np.ndarray [N]
test_metrics = ckpt["test_metrics"]  # dict
thresholds   = ckpt["thresholds"]    # {'T_low': float, 'T_high': float}
temperature  = ckpt["temperature"]   # float

# Rebuild the model
from src.models.dit_classifier import DiTClassifier
model = DiTClassifier()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load the calibrator
with open("checkpoints/dit/calibrator.pkl", "rb") as fh:
    calibrator = pickle.load(fh)
```

### 8.3 Calibrator format

Calibrators are pickled as full `TemperatureCalibrator` objects (not dicts).
`load_pipeline()` handles both the object and the legacy dict format automatically.

Do **not** use `calibrator.load(path)` directly on a pkl saved as an object —
use `pickle.load()` instead.

### 8.4 Threshold backup

Every run of `scripts/retune_thresholds.py` creates a timestamped backup:

```
checkpoints/thresholds_backup_<YYYYMMDDTHHMMSSZ>/
    best_resnet50.pt
    calibrator_resnet50.pkl
    best_vit_base_patch16_224.pt
    calibrator_vit_base_patch16_224.pkl
    best_model.pt          (DiT)
    calibrator.pkl         (DiT)
```

Roll back by copying the desired files back to `checkpoints/baseline/` or
`checkpoints/dit/`.

---

## 9. Calibration and Thresholds

### 9.1 Temperature scaling

After training, a scalar temperature `T` is fit by minimising binary
cross-entropy on the validation logits:

```
calibrated_prob = sigmoid(logit / T)
```

- `T > 1` softens overconfident predictions (ViT: T ≈ 2.3, DiT: T ≈ 3.7)
- `T < 1` sharpens underconfident predictions (ResNet: T ≈ 0.9)

The calibrator is fit once per training run (on val logits) and applied to
all splits thereafter.

### 9.2 Threshold selection

Two thresholds produce three output bands:

```
prob < T_low  →  safe_for_extraction
T_low ≤ prob ≤ T_high  →  review
prob > T_high →  high_hallucination_risk
```

**FSR-target method (used during training):**
`T_low` is the highest threshold at which FSR ≤ 5% on the validation set.
`T_high = (T_low + 1) / 2`.

**Cost-weighted method (used in production — Phase 6c):**
τ* minimises `cost(τ) = 10·FN(τ) + 1·FP(τ)` on the validation split.
This is the method stored in the checkpoints after `retune_thresholds.py`.

### 9.3 Retuning thresholds

```bash
# Default: FN is 10× more expensive than FP
python scripts/retune_thresholds.py

# Softer cost ratio
python scripts/retune_thresholds.py --fn-cost 5 --fp-cost 1
```

This updates `checkpoint["thresholds"]` and `calibrator.t_low` / `calibrator.t_high`
in all three model checkpoints simultaneously, and writes a timestamped backup.

**Important:** τ* must be written to **both** the checkpoint dict and the
calibrator object. `retune_thresholds.py` does both. Touching only one will
cause inference to use inconsistent thresholds.

---

## 10. Evaluation

### 10.1 Evaluation notebooks (eval-only — no retraining)

| Notebook | Models | What it shows |
|---|---|---|
| `notebooks/04_baseline_training.ipynb` | ResNet50, ViT-Base | Loss curves, confusion matrices, ROC/PR curves, calibration plots, per-institution metrics, error slice analysis |
| `notebooks/05_dit_training.ipynb` | DiT | Same as above, plus per-stage parameter counts and training curve breakdowns |
| `notebooks/06_calibration_eval.ipynb` | All | Calibration reliability diagrams, threshold sensitivity, FSR vs review-rate trade-off, false-safe case details |

These notebooks **do not retrain**. They load the checkpoint, read stored
logits/labels from the checkpoint dict, and replot everything. Run all cells top
to bottom — a fresh kernel is sufficient.

### 10.2 Standalone evaluator

```bash
python -m src.train.evaluate \
    --config configs/dit.yaml \
    --checkpoint checkpoints/dit/best_model.pt
```

Writes:
- `eval_output/eval_summary.json` — full metric dict
- `eval_output/error_analysis.csv` — per-page false-safe and false-risky cases

### 10.3 Metrics computed

All evaluation surfaces the same metric set via `src/utils/metrics.py`:

| Metric | Description |
|---|---|
| `f1` | Macro F1 across safe/risky |
| `precision_safe` | Precision on safe-class predictions |
| `recall_risky` | Recall on risky-class predictions (= 1 − FSR) |
| `false_safe_rate` | Fraction of truly risky pages predicted safe — **primary metric** |
| `review_rate` | Fraction of pages routed to the review band |
| `roc_auc` | Area under the ROC curve |
| `pr_auc` | Area under the Precision-Recall curve |
| `ece` | Expected Calibration Error (10 bins) |

Per-institution breakdowns of F1, recall, and FSR are logged to console and
available in the evaluation notebooks.

### 10.4 External validation set (400 documents)

```bash
python scripts/_run_validation_inference.py
```

Runs all three models on:
- `data/validation_set/handwritten/` — 94 risky documents
- `data/validation_set/regular_documents/` — 306 safe documents

Outputs:
- `validation_report/report_data.json` — per-model per-doc arrays
- `validation_report/index.html` — standalone interactive HTML report

**Must rerun `_run_validation_inference.py` after any threshold or model change**
before regenerating `reports/finetune_report.html`, otherwise the report builder
will error with a clear message about missing `per_doc` arrays.

---

## 11. Inference on New PDFs

### 11.1 Python API (recommended)

```python
from src.inference.predict import load_pipeline, predict_single, predict_batch
from src.inference.service_schema import PredictionRequest

# Load model + calibrator + thresholds from disk
model, calibrator, thresholds, device = load_pipeline(
    checkpoint_path="checkpoints/dit/best_model.pt",
    calibrator_path="checkpoints/dit/calibrator.pkl",
    model_type="dit",           # one of: 'resnet50', 'efficientnet_b0', 'vit', 'dit'
    config_path="configs/dit.yaml",
    device="auto",              # 'auto' selects CUDA → MPS → CPU
)

# Single PDF (renders page 1 on the fly)
response = predict_single(
    pdf_path="path/to/document.pdf",
    model=model,
    calibrator=calibrator,
    thresholds=thresholds,
    dpi=150,
    device=device,
)
print(response.risk_category)   # 'safe_for_extraction' | 'review' | 'high_hallucination_risk'
print(response.confidence)      # calibrated probability [0, 1]
print(response.raw_logit)       # raw model logit

# Batch of pre-rendered PNGs
requests = [
    PredictionRequest(file_path="page_0001.pdf", page_num=1),
    PredictionRequest(file_path="page_0002.pdf", page_num=2),
]
responses = predict_batch(
    requests=requests,
    model=model,
    calibrator=calibrator,
    thresholds=thresholds,
    rendered_dir="data/rendered_pages",
    device=device,
    batch_size=32,
)
```

`load_pipeline` reads `model_name` from the checkpoint and overrides the config
automatically — passing `configs/baseline.yaml` with `model_type="vit"` works
correctly without manually editing the YAML.

### 11.2 CLI — batch inference over a directory

```bash
python -m src.inference.predict \
    --checkpoint checkpoints/dit/best_model.pt \
    --calibrator checkpoints/dit/calibrator.pkl \
    --model_type dit \
    --config configs/dit.yaml \
    --input_dir data/rendered_pages \
    --output_json inference_results/predictions.json \
    --batch_size 32 \
    --device auto
```

The output JSON is a list of objects with keys:
`file_path`, `page_num`, `risk_category`, `confidence`, `raw_logit`.

### 11.3 Response schema

```python
class PredictionResponse(BaseModel):
    file_path: str
    page_num: int
    risk_category: RiskCategory      # 'safe_for_extraction' | 'review' | 'high_hallucination_risk'
    confidence: float                # calibrated probability [0, 1]
    raw_logit: float                 # raw model output before calibration

    @property
    def is_safe(self) -> bool: ...
    @property
    def needs_review(self) -> bool: ...
    @property
    def is_high_risk(self) -> bool: ...
```

### 11.4 Routing logic

```
confidence < T_low  → safe_for_extraction   (proceed with LLM extraction)
T_low ≤ conf ≤ T_high → review              (route to human reviewer)
confidence > T_high → high_hallucination_risk (block from extraction)
```

With the Phase 6d cost-weighted thresholds (FN:FP = 10:1), τ* is in the range
0.02–0.05, meaning the model is very conservative — nearly all uncertain pages
are routed to review rather than declared safe.

### 11.5 End-to-end demo

See `notebooks/07_inference_demo.ipynb` for a walkthrough that loads ResNet50,
renders raw PDFs on the fly, and prints a risk summary per page.

---

## 12. Results Summary

### Held-out test set (n=284, source-doc grouped — Phase 6d)

| Model | F1 | ROC-AUC | Temperature | τ* | FN @ τ* |
|---|---|---|---|---|---|
| ResNet50 | 0.971 | 0.999 | 0.932 | 0.050 | **0** |
| ViT-Base | 0.953 | 0.986 | 2.353 | 0.032 | **0** |
| DiT | **0.978** | **0.999** | 3.749 | 0.036 | **0** |

All three models achieve **0 false-negatives** on the held-out test set at τ*.

### External validation set (n=400: 94 risky + 306 safe — Phase 6d, at τ*)

| Model | FSR @ τ* | Accuracy | TN / FP / FN / TP |
|---|---|---|---|
| ResNet50 | 2.1% | 55.8% | 139 / 167 / 2 / 92 |
| ViT-Base | 3.2% | 62.0% | 177 / 129 / 3 / 91 |
| **DiT** | **1.1%** | **71.2%** | 215 / 91 / 1 / 93 |

DiT leads on both FSR and accuracy on the external set. The accuracy drop vs
the test set (71% vs 97%) reflects domain shift — the conservative τ* routes
many borderline safe pages to review, which is the intended behaviour.

**Comparison with Phase 5 (pre-expansion, at 0.5 threshold):**

| Model | Phase 5 FSR | Phase 6d FSR | Improvement |
|---|---|---|---|
| ResNet50 | 36.2% | 2.1% | −34.1 pp |
| ViT-Base | 50.0% | 3.2% | −46.8 pp |
| DiT | 30.9% | 1.1% | −29.8 pp |

The improvement comes from three sources: expanded training corpus (1,873 vs
1,014 pages, 4 additional source folders), cost-weighted threshold selection
(Phase 6c), and corrected labels for 191 pages (Phase 6d).

---

## 13. Reports

Three self-contained HTML reports are committed to the repository (Plotly loaded
from CDN — no server required, just open in a browser):

| File | Size | Contents |
|---|---|---|
| [`reports/finetune_report.html`](reports/finetune_report.html) | ~261 KB | Tabs: Overview / Training Curves / Held-out Test / External Validation / Findings. Model selector on Test and External tabs. Phase 5 vs 6 FSR comparison with 15% target line. |
| [`reports/mixed_eval_report.html`](reports/mixed_eval_report.html) | ~72 KB | Gemini (text LLM) vs DiT comparison on 648 ground-truth docs. Tabs: Overview / Detailed Metrics / Error Analysis / Model Agreement / Findings. |
| [`validation_report/index.html`](validation_report/index.html) | ~132 KB | External validation set (400 docs) with model-selector dropdown. |

Plain-text summary: [`docs/FINETUNE_REPORT.md`](docs/FINETUNE_REPORT.md) — suitable for Slack or email.

### Regenerating reports

```bash
# 1. Run external validation inference first (needed by finetune_report builder)
python scripts/_run_validation_inference.py

# 2. Rebuild the fine-tune report
python scripts/build_finetune_report.py        # → reports/finetune_report.html

# 3. Rebuild the Gemini vs DiT comparison
python scripts/build_mixed_eval_report.py      # → reports/mixed_eval_report.html
```

---

## 14. Special-Font OCR Track

An independent pipeline for scanned Hebrew PDFs that use a non-standard font
(unreadable by standard OCR tools). Lives under `data/spaciel_font/` and runs
entirely locally on Apple Silicon using `mlx-vlm` and Gemma 4 26B (4-bit).

**Hardware requirement:** ~36 GB unified memory (Apple M-series). Will not run
on x86 or non-MLX hardware.

### Step 1 — Render PDFs at high resolution

```bash
python scripts/split_pdfs_to_pages.py
```

Renders every PDF under `data/spaciel_font/` to per-page PNGs at **300 DPI**
(higher than the 150 DPI used for the risk classifier — OCR is resolution-sensitive).
Outputs go to `data/spaciel_font_pages/<stem>_page_NNN.png` plus a
`manifest.csv` (columns: `page_image`, `source_pdf`, `page_num`, `status`).
Skips already-rendered pages.

### Step 2 — Run Gemma 4 OCR

Open `notebooks/09_special_font_ocr.ipynb` and run all cells.

On first run, `mlx-vlm.load` downloads `mlx-community/gemma-4-26b-a4b-it-4bit`
(~15 GB) and caches it in `~/.cache/huggingface/`. Runtime is 5–15 s/page.

For each page, two prompts are run:
1. **OCR prompt** — extracts all visible text, preserving RTL Hebrew reading order
   and line breaks (max 2048 tokens)
2. **Structured prompt** — attempts to extract key fields as JSON
   (name, ID, date, address, document type, institution)

Results are appended to `data/spaciel_font_pages/ocr_results.json` after every
page so interrupted runs resume cleanly. The notebook keys resume state on
`page_image` filenames already present in `ocr_results.json`.

### Step 3 — Export to CSV

Run the export cell in `notebooks/09_special_font_ocr.ipynb`. Produces
`data/spaciel_font_pages/ocr_results.csv` (UTF-8-sig for Hebrew compatibility
in Excel).

**Note:** The `data/spaciel_font/` and `data/spaciel_font_pages/` directories
must **not** be renamed. Both the OCR pipeline and the training renderer are
keyed on those names. Only the metadata labels changed in Phase 6d; the
on-disk directories are untouched.

---

## 15. Config Reference

### `configs/baseline.yaml` (ResNet / ViT)

```yaml
model:
  name: resnet50              # timm model name; override with --model flag
  pretrained: true
  num_classes: 1

training:
  epochs: 30
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  early_stopping_patience: 5
  loss: bce_with_logits
  device: auto                # "auto" | "cpu" | "mps" | "cuda"
  pos_weight: auto            # "auto" (n_neg/n_pos) | float | null
  use_weighted_sampler: true  # oversample minority class

data:
  dpi: 150
  image_size: 224
  grayscale: true
  augmentation: true
  metadata_csv: "data/metadata.csv"
  rendered_dir: "data/rendered_pages"

splits:
  strategy: grouped_holdout
  group_col: source_doc_stem  # group key — prevents page leakage
  stratify_col: label_binary
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

output:
  checkpoint_dir: "checkpoints/baseline"
  log_dir: "logs/baseline"
```

### `configs/dit.yaml` (DiT)

Same structure, with stage-specific training fields instead of a single `epochs`:

```yaml
training:
  stage1_epochs: 30    stage1_lr: 1.0e-3   # head only
  stage2_epochs: 30    stage2_lr: 1.0e-5   # top-2 blocks
  stage3_epochs: 30    stage3_lr: 5.0e-6   # full model
  batch_size: 16
  early_stopping_patience: 15
  ...
output:
  checkpoint_dir: "checkpoints/dit"
  log_dir: "logs/dit"
```

### `configs/baseline_simple.yaml` and `configs/dit_simple.yaml`

Mirror the structure above but point at the simple two-folder pipeline
artefacts (see Section 6):

```yaml
data:
  metadata_csv: "data/metadata_simple.csv"
  rendered_dir: "data/rendered_pages_simple"

splits:
  strategy: stratified_random   # random per-page (no source_doc_stem grouping)
  group_col: null
  stratify_col: label_binary
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

output:
  checkpoint_dir: "checkpoints/baseline_simple"   # or "checkpoints/dit_simple"
  log_dir: "logs/baseline_simple"                 # or "logs/dit_simple"
```

### `configs/inference.yaml`

Used by `scripts/_run_validation_inference.py` and the CLI. Specifies model
paths and fallback thresholds (overridden by stored calibrator thresholds when
available):

```yaml
model:
  checkpoint_path: "checkpoints/dit/best_model.pt"
  calibrator_path: "checkpoints/dit/calibrator.pkl"

thresholds:
  safe_upper: 0.3      # T_low fallback (used only if calibrator has no stored thresholds)
  risky_lower: 0.7     # T_high fallback
```

---

## 16. Repository Structure

```
for_tal/
├── data/
│   ├── handwritten/                       # source PDFs — handwritten questionnaires (legacy pipeline)
│   ├── handwritten_and_questioniers/      # source PDFs — mixed handwritten + typed (legacy pipeline)
│   ├── handwritten_edge_cases/            # source PDFs — edge-case handwritten pages (legacy pipeline)
│   ├── regular_forms/                     # source PDFs — standard typed forms (legacy pipeline)
│   ├── regular_forms_edge_cases/          # source PDFs — edge-case regular forms (was spaciel_font)
│   ├── spaciel_font/                      # source PDFs — special-font scans (OCR track only)
│   ├── safe/                              # source PDFs — simple pipeline, label=0 (ADR-0001)
│   ├── risky/                             # source PDFs — simple pipeline, label=1 (ADR-0001)
│   ├── validation_set/                    # external validation PDFs (400 docs)
│   ├── rendered_pages/                    # 224×224 grayscale PNGs (legacy, gitignored)
│   ├── rendered_pages_simple/             # 224×224 grayscale PNGs (simple pipeline, gitignored)
│   ├── splits/                            # train.csv / val.csv / test.csv (legacy, gitignored)
│   ├── splits_simple/                     # train.csv / val.csv / test.csv (simple pipeline, gitignored)
│   ├── splits_v1/, splits_v2_precorrection/  # pre-correction split backups
│   ├── metadata.csv                       # master index — legacy pipeline (1873 rows)
│   ├── metadata_simple.csv                # master index — simple pipeline (ADR-0001)
│   ├── metadata_v1.csv                    # Phase 5 backup (1014 rows)
│   ├── metadata_v2_precorrection.csv      # pre-Phase-6d backup (1873 rows)
│   ├── labels_binary.csv                  # original binary annotation
│   ├── labels_binary_clean.csv            # schema-normalised binary labels
│   ├── labels_rubric.csv                  # D/H/S/L rubric scores (legacy pipeline)
│   ├── labels_rubric_simple.csv           # D/H/S/L rubric scores (simple pipeline)
│   ├── rubric_checkpoint.jsonl            # Claude vision API checkpoint (legacy)
│   ├── rubric_simple_checkpoint.jsonl     # Claude vision API checkpoint (simple)
│   ├── validation_results_full.csv        # per-doc external validation output
│   ├── handwritten.xlsx                   # ground-truth source spreadsheet
│   ├── regular_forms.xlsx                 # ground-truth source spreadsheet
│   ├── handwritten_validation_set.xlsx    # external validation ground truth (risky)
│   └── regular_documents_validation_set.xlsx
├── src/
│   ├── data/
│   │   ├── render_pdf.py                  # PDF → PIL Image renderer (PyMuPDF)
│   │   ├── dataset.py                     # HallucinationRiskDataset (PyTorch Dataset)
│   │   └── splits.py                      # grouped split + k-fold utilities
│   ├── models/
│   │   ├── resnet_baseline.py             # ResNetClassifier (timm)
│   │   ├── vit_baseline.py                # ViTClassifier (timm)
│   │   ├── dit_classifier.py              # DiTClassifier (HuggingFace)
│   │   └── calibrator.py                  # TemperatureCalibrator
│   ├── train/
│   │   ├── train_baseline.py              # ResNet/ViT training loop
│   │   ├── train_dit.py                   # DiT 3-stage training loop
│   │   └── evaluate.py                    # standalone evaluator + error analysis
│   ├── inference/
│   │   ├── service_schema.py              # Pydantic v2 request/response schemas
│   │   └── predict.py                     # load_pipeline, predict_single, predict_batch, CLI
│   └── utils/
│       ├── metrics.py                     # F1, FSR, ROC-AUC, PR-AUC, ECE, per-institution
│       ├── logging.py                     # TrainingLogger (JSON), ErrorAnalysisLogger (CSV)
│       ├── visualization.py               # 7 plot functions (confusion, ROC, PR, calibration…)
│       └── device.py                      # get_device, prepare_model, mps_sync, mps_empty_cache
├── scripts/
│   ├── clean_labels.py                    # label cleaning + metadata bootstrap (Phase 0)
│   ├── build_metadata_v2.py               # 5-folder metadata builder (legacy, Phase 6)
│   ├── render_all_pages.py                # batch PDF → PNG renderer (legacy, Phase 6)
│   ├── regenerate_splits.py               # grouped split generator (legacy, Phase 6)
│   ├── annotate_rubric.py                 # Claude vision rubric annotator (legacy, Phase 4)
│   ├── build_metadata_simple.py           # 2-folder metadata builder (simple, ADR-0001)
│   ├── render_pages_simple.py             # batch PDF → PNG renderer (simple, ADR-0001)
│   ├── make_splits_simple.py              # stratified random split (simple, ADR-0001)
│   ├── annotate_rubric_simple.py          # Claude vision rubric annotator (simple, ADR-0001)
│   ├── retune_thresholds.py               # cost-weighted threshold recalibration (Phase 6c)
│   ├── split_pdfs_to_pages.py             # 300 DPI renderer for OCR (Phase 5d)
│   ├── build_finetune_report.py           # regenerates reports/finetune_report.html
│   ├── build_mixed_eval_report.py         # regenerates reports/mixed_eval_report.html
│   ├── _run_validation_inference.py       # regenerates validation_report/
│   ├── fix_spaciel_font_to_edge_cases.py  # one-shot metadata relabel (Phase 6d)
│   └── drive_to_gcs.py                    # optional GCS upload (not part of local workflow)
├── notebooks/
│   ├── 00_gcp_setup.ipynb                 # local setup orchestrator
│   ├── 01_data_audit.ipynb                # PDF discovery + metadata construction
│   ├── 02_rendering_checks.ipynb          # DPI / aspect-ratio / cache validation
│   ├── 03_label_consistency.ipynb         # rubric distribution + label consistency
│   ├── 04_baseline_training.ipynb         # ResNet50 + ViT-Base eval (eval-only)
│   ├── 05_dit_training.ipynb              # DiT 3-stage eval (eval-only)
│   ├── 06_calibration_eval.ipynb          # calibration curves + threshold sensitivity
│   ├── 07_inference_demo.ipynb            # end-to-end PDF → risk-category demo
│   ├── 08_validation_inference.ipynb      # external validation (400 docs, 3 models)
│   └── 09_special_font_ocr.ipynb          # Gemma 4 OCR for special-font scans
├── configs/
│   ├── baseline.yaml                      # ResNet/ViT — legacy 5-folder pipeline
│   ├── dit.yaml                           # DiT — legacy 5-folder pipeline
│   ├── baseline_simple.yaml               # ResNet/ViT — simple 2-folder pipeline (ADR-0001)
│   ├── dit_simple.yaml                    # DiT — simple 2-folder pipeline (ADR-0001)
│   └── inference.yaml
├── eval/
│   ├── evnaluation.py                     # Gemini vs DiT evaluation script
│   └── mixed_results - Sheet1.csv         # 648-doc ground-truth evaluation CSV
├── reports/
│   ├── finetune_report.html               # Phase 6 interactive report
│   ├── mixed_eval_report.html             # Gemini vs DiT comparison report
│   └── _finetune_report_template.html     # HTML/JS template
├── validation_report/
│   ├── index.html
│   └── report_data.json
├── plots/                                 # saved metric PNGs (baseline/, dit/)
├── eval_output/                           # standalone evaluator outputs (gitignored)
├── checkpoints/                           # model weights (gitignored)
├── logs/                                  # training logs (gitignored)
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── INTERFACES.md
│   ├── FINETUNE_REPORT.md
│   └── decisions/
│       └── 0001-simple-two-folder-ingestion.md   # ADR (Section 6)
└── requirements.txt
```

---

## 17. Known Gotchas

- **`file_path` in metadata.csv stores `.pdf` extensions**, not `.png`.
  `HallucinationRiskDataset._build_file_index()` resolves this automatically
  at dataset load time. Do not rename rendered PNGs.

- **Hebrew filenames throughout.** All path handling uses `pathlib.Path` — never
  use string concatenation for paths. Files with Hebrew names are valid on macOS
  and Linux; Windows may need WSL.

- **Do not force `channels_last` on MPS.** timm backbones use internal `.view()`
  operations that crash on MPS backward pass when the tensor is in NHWC format.
  PyTorch ≥ 2.11 handles the conversion transparently.

- **`data/spaciel_font/` and `data/spaciel_font_pages/` must not be renamed.**
  The OCR pipeline (`notebooks/09_special_font_ocr.ipynb`) and the
  `render_all_pages.py` renderer both key on those directory names. Only the
  metadata labels changed in Phase 6d; the filesystem is unchanged.

- **DiT best checkpoint may be stage 2, not stage 3.** The training script saves
  the global best across all stages. Stage 2 (top-2 blocks) often converges
  earlier than stage 3 with patience=15. Always use `best_model.pt`.

- **`load_pipeline()` requires both `model_type` and `config_path`.** When
  `model_type='vit'` and `config_path='configs/baseline.yaml'` (which says
  `resnet50`), the pipeline reads `model_name` from the checkpoint to override
  the config before building the model.

- **Calibrator pickles are whole objects, not dicts.** Use `pickle.load()`,
  not `calibrator.load(path)`. `load_pipeline()` handles both formats.

- **Retune thresholds before regenerating reports.** `build_finetune_report.py`
  reads `per_doc` arrays from `validation_report/report_data.json`.
  Re-run `scripts/_run_validation_inference.py` after any threshold or model
  change to keep those arrays in sync.

- **`data/metadata.csv` encoding: UTF-8, no BOM.** `labels_binary_clean.csv`
  uses UTF-8-sig (BOM). Always open `metadata.csv` with
  `encoding="utf-8"` — the BOM-less encoding avoids a `KeyError` on
  `\ufefffile_path` in scripts that read it with pandas.

- **`pos_weight: auto` computes `n_neg / n_pos` from the training split at
  runtime.** Do not set it to a hard-coded value unless you have a specific
  reason to deviate from the class-frequency ratio.

- **Simple pipeline (Section 6) splits at the page level, not the document
  level.** Multi-page source PDFs may have their pages split across
  train/val/test. Use the legacy pipeline when document-level holdout
  matters.

- **Simple pipeline file paths must remain disjoint from the legacy
  pipeline.** Never set `data.metadata_csv: data/metadata.csv` in
  `configs/*_simple.yaml`, or vice versa — the schemas and splitting
  strategies are incompatible and silently mixing them will produce wrong
  metrics. See ADR-0001 for the full contract.

- **`anthropic` is not declared in `requirements.txt`** but is imported by
  both `scripts/annotate_rubric.py` and `scripts/annotate_rubric_simple.py`.
  Install separately (`pip install anthropic`) before running either, or use
  `--dry-run` on the simple annotator for plumbing tests.

---

## 18. Architecture Decision Records

Architectural decisions live under `docs/decisions/` as MADR-style ADRs.
They are append-only: to change a decision, write a new ADR with
`supersedes: [ADR-NNNN]` rather than editing the existing one.

| ID | Title | Status |
|---|---|---|
| [ADR-0001](docs/decisions/0001-simple-two-folder-ingestion.md) | Add parallel simple two-folder ingestion path (safe/ + risky/) | accepted |

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the per-phase
status board and the open-task queue.
