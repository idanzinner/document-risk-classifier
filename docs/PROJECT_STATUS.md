# Project Status — Hebrew PDF Hallucination-Risk Classifier

## What Has Been Done

### Phase 0: Scaffolding (Completed)
- Repository skeleton created with full directory structure matching the plan
- `requirements.txt` with all 14 dependencies (torch, torchvision, timm, transformers, pymupdf, Pillow, scikit-learn, pandas, numpy, matplotlib, pyyaml, tqdm, pydantic, albumentations, scipy)
- Interface contract defined in `docs/INTERFACES.md` — all function signatures, class APIs, data schemas
- Config YAMLs created: `configs/baseline.yaml`, `configs/dit.yaml`, `configs/inference.yaml`
- Metadata CSV stubs created with correct schemas: `data/metadata.csv`, `data/labels_binary.csv`, `data/labels_rubric.csv`
- `README.md` with setup, usage, and project overview

### Phase 1: Core Modules (Completed)

**Agent 1 — Data Pipeline (`src/data/`)**
- `render_pdf.py`: PyMuPDF renderer with aspect-ratio preservation, white padding, deterministic output, batch rendering with skip-existing cache
- `dataset.py`: `HallucinationRiskDataset` — loads from metadata CSV, scan-realistic augmentations (rotation/blur/contrast/perspective), grayscale-aware normalization
- `splits.py`: Grouped splitting by institution (70/15/15), 5-fold grouped CV, save/load to disk

**Agent 2 — Model Architectures (`src/models/`)**
- `resnet_baseline.py`: `ResNetClassifier` — timm-based ResNet50/EfficientNet-B0, ImageNet pretrained, single-logit binary head, grayscale guard
- `vit_baseline.py`: `ViTClassifier` — timm-based ViT-Base, same pattern
- `dit_classifier.py`: `DiTClassifier` — HuggingFace `microsoft/dit-base`, staged training helpers (`freeze_backbone`, `unfreeze_top_blocks`, `unfreeze_all`)
- `calibrator.py`: `TemperatureCalibrator` — temperature scaling via scipy NLL minimization, threshold selection targeting false-safe rate ≤ 5%

**Agent 3 — Metrics and Utilities (`src/utils/`)**
- `metrics.py`: Full metric suite — F1, precision_safe, recall_risky, false_safe_rate, review_rate, ROC-AUC, PR-AUC, ECE; per-institution breakdowns
- `visualization.py`: 7 plot functions — confusion matrix, ROC, PR, calibration, per-institution bars, error slice summary, score distribution (all non-interactive, save-to-file)
- `logging.py`: `TrainingLogger` (JSON epoch history), `ErrorAnalysisLogger` (CSV per-sample with UTF-8-sig for Hebrew filenames)

**Agent 4 — Training and Evaluation (`src/train/`)**
- `train_baseline.py`: Full training loop — BCEWithLogitsLoss, AdamW, early stopping, checkpoint saving, per-epoch calibration, `TrainingLogger` JSON output
- `train_dit.py`: Three-stage DiT training orchestrator — per-stage early stopping, global best selection, `TrainingLogger` JSON output
- `evaluate.py`: Standalone evaluator — 7 error slices, `ErrorAnalysisLogger` for all false-safe/false-risky cases, `eval_summary.json` output

### Phase 2: Integration Modules and Notebooks (Completed)

**Agent 5 — Inference Service (`src/inference/`)**
- `service_schema.py`: Pydantic v2 schemas — `RiskCategory` enum, `PredictionRequest`, `PredictionResponse` (with `is_safe`/`needs_review`/`is_high_risk` properties), `BatchPredictionRequest`, `BatchPredictionResponse`
- `predict.py`: `load_pipeline`, `predict_batch`, `predict_single` — full batch inference pipeline, Hebrew-safe pathlib handling, CLI entrypoint

**Agent 6 — Data Notebooks (`notebooks/01-03`)**
- `01_data_audit.ipynb`: PDF discovery across both Hebrew subdirectories, metadata DataFrame construction, filename analysis, duplicate checks
- `02_rendering_checks.ipynb`: DPI comparison, grayscale/RGB comparison, aspect-ratio verification, batch timing, cache validation
- `03_label_consistency.ipynb`: Risk score formula demo, threshold classification, borderline set definition, contradiction checker, rubric distribution simulation

**Agent 7 — Training Notebooks (`notebooks/04-06`)**
- `04_baseline_training.ipynb`: ResNet + ViT training with loss curves, calibration, metrics comparison table
- `05_dit_training.ipynb`: Three-stage DiT training with param count inspection, per-stage plots, diagnostic set evaluation
- `06_calibration_eval.ipynb`: Calibration curve analysis, threshold sensitivity plot, full test metrics, per-institution breakdown, false-safe error analysis, production readiness checklist

### Phase 3: GCP Alignment (Completed)

**Goal:** Make the pipeline runnable on Vertex AI Workbench / Colab Enterprise using GCS as the storage backend. Zero changes to `src/` modules — GCSFuse mounts the bucket at `/gcs/<bucket>/` so all `pathlib.Path` code works unchanged.

**Key decisions:**
- GCS strategy: GCSFuse (not gcsfs/direct SDK) — all existing file I/O works via mounted path
- Compute: Vertex AI Workbench / Colab Enterprise (notebook-driven, no Vertex AI Training Jobs needed for now)
- Data migration: Google Drive → GCS via `gdown` + `google-cloud-storage`
- Split grouping: changed from `institution` to `template_family` — see gotchas below

**Artifacts produced:**
- `scripts/clean_labels.py`: Cleans `data/labels_binary.csv` → `data/labels_binary_clean.csv` + `data/metadata.csv`
  - Fixes column names, converts text labels to int, extracts `page_num`, derives `institution`, stubs rubric columns as -1
- `scripts/drive_to_gcs.py`: Downloads all 1014 PDFs from Google Drive (via `gdown`) and uploads to `gs://<BUCKET>/raw_pdfs/`; also uploads CSVs to `gs://<BUCKET>/data/`
- `notebooks/00_gcp_setup.ipynb`: GCP auth, bucket validation, split generation, full render pipeline run, spot-check visualizations
- `configs/baseline.yaml`, `configs/dit.yaml`, `configs/inference.yaml`: Updated with GCS-mounted paths and `gcs.bucket_name` field
- `requirements.txt`: Added `google-cloud-storage>=2.10.0`, `gcsfs>=2023.6.0`, `gdown>=5.0.0`

**GCS bucket structure:**
```
gs://YOUR_BUCKET_NAME/
  raw_pdfs/           # 1014 single-page PDFs (flat, no Hebrew folder names)
  rendered_pages/     # 1014 PNGs at 224x224 grayscale (populated by render_all)
  data/
    metadata.csv      # master index (all schema columns, rubric stubs -1)
    labels_binary_clean.csv
    labels_binary.csv
    splits/
      train.csv
      val.csv
      test.csv
  checkpoints/
    baseline/
    dit/
  logs/
    baseline/
    dit/
  inference_results/  # output from predict.py
```

---

## Remaining Pipeline Steps (Prioritized)

### Next — Must Do Before Training

1. **Create GCS bucket** — create a GCS bucket and update `YOUR_BUCKET_NAME` in all three `configs/*.yaml` files.

2. **Migrate PDFs to GCS** — run `python scripts/drive_to_gcs.py --bucket YOUR_BUCKET_NAME`. Requires Drive files to be set to "Anyone with the link can view".

3. **Run setup notebook** — open `notebooks/00_gcp_setup.ipynb` on Vertex AI Workbench, set `BUCKET_NAME`, and run all cells to validate data and render images.

4. **Populate rubric scores** — fill in D/H/S/L columns in `metadata.csv` for at least the borderline cases (risk_score 4-6) before DiT training. The `data/labels_rubric.csv` is still empty.

5. **Verify split strategy** — `template_family` is now used as the grouping key. With only 3 template families in the current data, kfold CV is recommended. Revisit `splits.strategy` in configs.

### Phase 4 — Model Training

6. Run `notebooks/04_baseline_training.ipynb` for ResNet + ViT baselines
7. Run `notebooks/05_dit_training.ipynb` for DiT fine-tuning
8. Run `notebooks/06_calibration_eval.ipynb` for final calibration and threshold selection

### Phase 5 — Production

9. Run `src/train/evaluate.py` on held-out test set
10. Deploy `src/inference/predict.py` as batch inference service
11. Set up review queue integration per confidence routing logic

### Phase 3 — Model Training

5. Run `notebooks/04_baseline_training.ipynb` for ResNet + ViT baselines
6. Run `notebooks/05_dit_training.ipynb` for DiT fine-tuning
7. Run `notebooks/06_calibration_eval.ipynb` for final calibration and threshold selection

### Phase 4 — Production

8. Run `src/train/evaluate.py` on held-out test set
9. Deploy `src/inference/predict.py` as batch inference service
10. Set up review queue integration per confidence routing logic

---

## Cross-Step Knowledge

### Data

- PDFs live in two subdirectories under `data/` locally, and in `gs://YOUR_BUCKET/raw_pdfs/` on GCS (flat, no Hebrew folder names)
  - `pdf_s - שאלונים וכתב יד /` — questionnaires and handwriting (label=1, risky)
  - `מסמכים רגילים /` — regular documents (label=0, safe)
- 1014 page-split PDFs with Hebrew filenames, all with `_page_XXXX` suffix
- Labels populated: 634 safe ("no risk"), 380 risky ("high risk") — see `data/labels_binary_clean.csv`
- Rubric scores (D/H/S/L) still pending — `data/metadata.csv` has stub values of -1

### GCP Setup

- **Compute:** Vertex AI Workbench or Colab Enterprise (notebook-based)
- **Storage:** GCS bucket — update `YOUR_BUCKET_NAME` in all `configs/*.yaml` before running
- **GCSFuse:** On Vertex AI Workbench the bucket mounts at `/gcs/<bucket_name>/` automatically. All `pathlib.Path` code in `src/` works unchanged through this mount.
- **Authentication:** VM service account on Workbench; `google.colab.auth.authenticate_user()` on Colab Enterprise
- **Migration script:** `python scripts/drive_to_gcs.py --bucket YOUR_BUCKET_NAME` (needs Drive files set to public link access)
- **Setup notebook:** `notebooks/00_gcp_setup.ipynb` — validates everything before training

### Engineering Constraints

- All preprocessing must be deterministic — `render_pdf.py` is stateless and hash-stable
- Rendering settings versioned via `configs/` YAMLs
- Splits saved to GCS at `gs://BUCKET/data/splits/` — never regenerate casually
- Calibrator and thresholds saved alongside the model checkpoint in GCS
- Evaluation must always report per-institution metrics
- Hebrew filenames handled via `pathlib` throughout — do not use string concatenation for paths

### Model Strategy

- DiT (`microsoft/dit-base`) is the target production model
- ResNet50 and ViT-Base are sanity-check baselines (Stage 1 milestones)
- Binary training with `BCEWithLogitsLoss`; ternary output via calibrated thresholds
- Most important metric: **false-safe rate** — safe predictions that are actually hallucination-risk
- Thresholds selected conservatively (low false-safe rate > raw F1)

### Known Gotchas

- **CRITICAL — Label/institution correlation:** All 634 "no risk" files are from `regular_docs`, all 380 "high risk" from `questionnaires`. Grouped splits by `institution` would put all of one class in train and the other in test. Use `template_family` as the group key instead, and verify splits have mixed labels.
- DiT is a BeitModel under the hood — staged unfreezing accesses `self.backbone.encoder.layer[-n:]`
- Grayscale input (mode 'L') needs to be repeated to 3 channels for all models (handled in each model's `forward`)
- Class imbalance: 634 safe vs 380 risky (~1.67:1). Monitor if it worsens after rubric annotation; consider focal loss if skew > 4:1
- Only 3 template families detected in current data (`regular_form`, `questionnaire_uuid`, `questionnaire_named`) — with so few groups, kfold CV is preferred over 70/15/15
- `ErrorAnalysisLogger` uses UTF-8-sig encoding to handle Hebrew filenames in Excel/CSV viewers
- `TrainingLogger` writes JSON history to GCS `logs/` — useful for comparing runs programmatically
- `gdown` downloads require Drive files to be set to "Anyone with the link can view" — verify sharing settings if downloads fail
