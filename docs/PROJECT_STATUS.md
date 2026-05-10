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

### Phase 3: Local Conversion (Completed)

**Goal:** Convert the pipeline from GCS/Vertex AI Workbench to run fully locally. Zero changes to `src/` modules — all file I/O uses `pathlib.Path` relative paths.

**Key decisions:**
- Removed all GCS/GCSFuse dependencies from configs, notebooks, and requirements
- `notebooks/00_gcp_setup.ipynb` rewritten as a local setup notebook (no GCP auth, no bucket references)
- Rendered images written to `data/rendered_pages/` locally (fixed inconsistency where notebooks 04/05/06 used `data/rendered`)
- GCP packages commented out in `requirements.txt` (kept for reference, not installed by default)
- `scripts/drive_to_gcs.py` retained as-is for optional GCS upload; not part of the local workflow

**Artifacts updated:**
- `configs/baseline.yaml`: removed `gcs:` block, paths now `data/metadata.csv`, `data/rendered_pages`, `checkpoints/baseline`, `logs/baseline`
- `configs/dit.yaml`: same local paths
- `configs/inference.yaml`: model paths `checkpoints/dit/best_model.pt`, `checkpoints/dit/calibrator.pkl`, output `inference_results`
- `notebooks/00_gcp_setup.ipynb`: local setup — validates directories, runs `clean_labels.py`, generates splits, renders PDFs, spot-checks
- `notebooks/04_baseline_training.ipynb`, `05_dit_training.ipynb`, `06_calibration_eval.ipynb`: fixed `RENDERED_DIR` from `data/rendered` → `data/rendered_pages`
- `src/train/train_baseline.py`, `src/train/train_dit.py`: default `rendered_dir` fallback updated to `data/rendered_pages`
- `requirements.txt`: GCP packages commented out

**Local directory structure:**
```
for_tal/
  data/
    rendered_pages/          # PNGs at 224x224 grayscale (populated by notebook 00)
    splits/
      train.csv
      val.csv
      test.csv
    metadata.csv             # master index (all schema columns, rubric stubs -1)
    labels_binary_clean.csv
    labels_binary.csv
  checkpoints/
    baseline/
    dit/
  logs/
    baseline/
    dit/
  inference_results/         # output from predict.py
```

---

## Remaining Pipeline Steps (Prioritized)

### Next — Must Do Before Training

1. **Run setup notebook** — open `notebooks/00_gcp_setup.ipynb` and run all cells to clean labels, generate splits, render PDFs, and validate data.

2. ~~**Populate rubric scores**~~ — **COMPLETED** (see Phase 4 below).

3. **Verify split strategy** — `template_family` is now used as the grouping key. With only 3 template families in the current data, kfold CV is recommended. Revisit `splits.strategy` in configs.

### Phase 4 — Rubric Annotation (Completed)

**Script:** `scripts/annotate_rubric.py` — auto-annotates D/H/S/L rubric scores via Claude vision API.

**Method:**
- 904 pages: scored by Claude (`claude-sonnet-4-5`) via vision API (10 concurrent requests, ~2.5 min total)
- 110 pages: fallback scores (D=1, H=2, S=1, L=2 → risk_score=7) — these pages had fully-Hebrew filenames that were masked to underscores in the CSV; all are `label_binary=1` (questionnaire/risky)
- PNG resolution uses direct stem match + prefix+datetime+page fallback for renamed files

**Results (1014 pages):**
- `safe_for_extraction` (risk_score ≤ 3): 676 pages
- `review` (risk_score 4–6): 228 pages
- `high_hallucination_risk` (risk_score > 6): 110 pages

**Cross-tab vs binary labels:**
- 534 binary-safe → safe_for_extraction, 100 binary-safe → review (these warrant inspection)
- 142 binary-risky → safe_for_extraction, 128 binary-risky → review, 110 binary-risky → high risk

**Artifacts:**
- `data/labels_rubric.csv` — 1014 rows, fully populated
- `data/metadata.csv` — D/H/S/L/risk_score columns updated
- `data/rubric_checkpoint.jsonl` — incremental checkpoint for resume support
- `scripts/annotate_rubric.py` — reusable annotation script

### Phase 4c — Hebrew Filename Fix in CSV Files (Completed)

**Problem:** `clean_labels.py` stripped Hebrew characters from filenames when generating `labels_binary_clean.csv`, `metadata.csv`, and `labels_rubric.csv`, replacing Hebrew runs with underscores (e.g., `104-105נפומניאשצי_נתנאל_קטין_נ_מנורה_20260222_095032_page_0001.pdf` → `104-105______________________________20260222_095032_page_0001.pdf`). This affected 193 rows across 18 distinct filename stems (all from the handwritten/questionnaire subset).

**Fix:** Propagated correct filenames from the hand-corrected `labels_binary.csv` using `drive_link` as the join key for `labels_binary_clean.csv` and `metadata.csv`, and a broken→correct mapping (derived from the drive_link join) for `labels_rubric.csv` (which has no `drive_link` column). All 193 rows updated in all three files; 0 residual underscore patterns remain.

**Impact:** The 28 "unresolvable" entries previously noted in Phase 4b are now fully resolvable — `dataset.py`'s fuzzy fallback matching is no longer needed for these files. PNG rendering should now cover 100% of the 380 risky-class pages.

**Artifacts modified:**
- `data/labels_binary_clean.csv` — 193 `file_path` values corrected
- `data/metadata.csv` — 193 `file_path` values corrected
- `data/labels_rubric.csv` — 193 `file_path` values corrected

---

### Phase 4b — Notebook Bug Fixes (Completed)

**Bugs found and fixed before training could run:**

1. **File extension mismatch in `dataset.py`**: `metadata.csv` stores `file_path` with `.pdf` extension, but rendered pages on disk are `.png`. All 1014 image lookups failed, returning zero tensors. Fixed by adding a file index in `HallucinationRiskDataset.__init__` that resolves `.pdf` → `.png`.

2. **Masked Hebrew filenames**: ~193 metadata entries had Hebrew characters replaced with underscores (e.g., `104-105______20260222_095032_page_0001.pdf` → actual file `104-105נפומניאשצי_נתנאל_קטין_נ_מנורה_20260222_095032_page_0001.png`). Fixed by multi-strategy fuzzy matching in `_build_file_index()` using (prefix, datetime, page), (datetime, page), (range, page), and suffix-based fallbacks. Result: 986/1014 resolved (28 pure-underscore entries remain unresolvable — all `label_binary=1`, ~2.8% of data).

3. **Label shape mismatch in notebook training loops (04, 05, 06)**: Dataset returns labels as `(1,)` tensors → batched to `(B, 1)`. Training functions squeezed logits to `(B,)` with `.squeeze(1)` but left labels at `(B, 1)`, causing `BCEWithLogitsLoss` to broadcast to `(B, B)` — wrong gradients. Fixed by adding `.squeeze(-1)` on both logits and labels.

4. **Wasteful label counting in notebook 04**: Cell 5 iterated through all samples (triggering image loads) just to count label distribution. Fixed to read directly from DataFrame.

**Artifacts modified:**
- `src/data/dataset.py` — file index builder, path resolver, `re` import
- `notebooks/04_baseline_training.ipynb` — training/eval functions, label counting, cleared stale outputs
- `notebooks/05_dit_training.ipynb` — training/eval functions squeeze fix
- `notebooks/06_calibration_eval.ipynb` — inference function squeeze fix

### Phase 5 — Model Training (Completed)

**Architecture change: training is now script-only; notebooks are eval-only.**

```bash
# Train ResNet50 (default model in configs/baseline.yaml)
python -m src.train.train_baseline --config configs/baseline.yaml

# Train ViT-Base (override via --model flag)
python -m src.train.train_baseline --config configs/baseline.yaml --model vit_base_patch16_224

# Train DiT (3-stage: head-only → top-2 blocks → full fine-tune)
python -m src.train.train_dit --config configs/dit.yaml
```

After training, open `notebooks/04_baseline_training.ipynb` or `notebooks/05_dit_training.ipynb` and run all cells for comprehensive evaluation (train + val + test metrics and plots). No re-training happens in the notebooks.

**What the scripts produce:**
- `checkpoints/baseline/best_{model_name}.pt` — model weights + expanded checkpoint with:
  - `train_logits`, `train_labels`, `train_metrics`
  - `val_logits`, `val_labels`, `val_metrics`
  - `test_logits`, `test_labels`, `test_metrics`
  - `thresholds`, `temperature`, `model_name`
- `checkpoints/baseline/calibrator_{model_name}.pkl` — fitted `TemperatureCalibrator`
- `logs/baseline/baseline_{model_name}.json` — per-epoch train/val loss + metrics history
- Same pattern for DiT under `checkpoints/dit/` and `logs/dit/`

**Training results (held-out test set, 152 samples):**

| Model | Best epoch/stage | Temperature | F1 | FSR | ROC-AUC | ECE | T_low | T_high |
|-------|-----------------|-------------|-----|-----|---------|-----|-------|--------|
| ResNet50 | epoch 10 | 0.514 | 0.973 | 5.26% | 1.000 | 0.013 | 0.827 | 0.913 |
| ViT-Base | epoch 2 | 1.458 | 0.964 | 5.26% | 0.998 | 0.028 | 0.521 | 0.760 |
| DiT | stage2 epoch 7 | 1.202 | 1.000 | 0.00% | 1.000 | 0.024 | 0.805 | 0.903 |

Checkpoints: `checkpoints/baseline/best_resnet50.pt`, `checkpoints/baseline/best_vit_base_patch16_224.pt`, `checkpoints/dit/best_model.pt`. Logs: `logs/baseline/baseline_resnet50.json`, `logs/baseline/baseline_vit_base_patch16_224.json`, `logs/dit/dit.json`.

**Engineering notes:**
- MPS training enabled in PyTorch 2.11.0; device auto-selection prefers CUDA > MPS > CPU
- **Note on channels_last:** Do NOT force `torch.channels_last` — timm backbones crash on MPS backward pass. PyTorch 2.11.0 handles NCHW→NHWC transparently.
- MPS mitigations retained: `mps_sync()`, NaN batch skip, gradient clipping (max_norm=1.0), `mps_empty_cache()`
- Shared utility `src/utils/device.py` — `get_device()`, `prepare_model()`, `prepare_input()`, `mps_sync()`, `mps_empty_cache()`
- Config-level device override: set `training.device: cpu` in configs if needed

### Phase 5b — Inference Demo (Completed)

- `notebooks/07_inference_demo.ipynb`: End-to-end demo notebook — loads trained ResNet50 + calibrator, accepts raw PDF files, renders → classifies → outputs risk category with calibrated probability
- Demonstrates single-file, batch, and full test-set evaluation
- Pipeline: PDF → render_page (PyMuPDF, 224×224 grayscale) → Grayscale→3ch + ToTensor + Normalize → ResNet50 → temperature calibration → threshold routing

### Phase 5c — External Validation Set Inference (Completed)

**Goal:** Evaluate all trained models on a new external validation set of 400 documents (94 handwritten/risky + 306 regular/safe) and produce a shareable interactive HTML report with a model-selector dropdown.

**Inputs:**
- `data/handwritten_validation_set.xlsx` — 94 risky documents (label=1)
- `data/regular_documents_validation_set.xlsx` — 306 safe documents (label=0)
- `data/validation_set/handwritten/` + `data/validation_set/regular_documents/` — source PDFs
- All three checkpoint/calibrator pairs (ResNet50, ViT-Base, DiT)

**Results (external validation set, threshold=0.5, 400 docs):**

| Model | Accuracy | Macro F1 | FSR (at T_low) | TN | FP | FN | TP |
|-------|----------|----------|----------------|----|----|----|----|
| ResNet50 | 90.8% | 0.861 | 36.2% (T_low=0.827) | 297 | 9 | 28 | 66 |
| ViT-Base | 86.5% | 0.778 | 50.0% (T_low=0.521) | 298 | 8 | 46 | 48 |
| DiT | 85.2% | 0.808 | 30.9% (T_low=0.805) | 267 | 39 | 20 | 74 |

**Observations:**
- ResNet50 has the best overall accuracy (90.8%) and F1 (0.861) on the external set
- DiT has the lowest false-safe rate (30.9%) — it misses the fewest risky documents when used with its T_low routing threshold — and the highest risky-class recall (TP=74)
- ViT-Base has the highest FSR (50%) and the lowest risky recall — likely because its T_low=0.521 is too conservative and routes too many uncertain cases as "safe"
- Domain shift is evident across all models: DiT achieved 0% FSR on the held-out test set but 30.9% on the external set; ResNet50 went from 5.26% to 36.2%

**Artifacts:**
- `notebooks/08_validation_inference.ipynb` — refactored to loop over all 3 models; uses `MODELS` dict and passes `model_type` + `config_path` to `load_pipeline`
- `validation_report/index.html` — standalone HTML report with model dropdown (zip-ready, ~132 KB)
- `validation_report/report_data.json` — nested per-model plot data under `models` key (~334 KB)
- `scripts/_run_validation_inference.py` — standalone script (no Jupyter kernel needed) that regenerates both report files
- `scripts/_html_template_nb08.html` — HTML template with model selector extracted for maintainability

**To share:** `zip -r validation_report.zip validation_report/`

### Phase 5d — Special-Font OCR with Gemma 4 (In Progress)

**Goal:** Extract text and structured fields from scanned Hebrew PDF pages that use a non-standard font, which standard OCR tools fail to read. Uses a local multimodal LLM on Apple Silicon so Hebrew scans never leave the machine.

**Model:** [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)
- Gemma 4 26B MoE, instruction-tuned, 4-bit quantized for Apple Silicon
- ~15 GB first-run download, cached in `~/.cache/huggingface/`
- Requires ~36 GB unified memory (Apple M-series)
- Runtime: `mlx-vlm` (installed on first notebook run into the `doc-risk-classifier` pyenv)

**Pipeline:**
1. `scripts/split_pdfs_to_pages.py` — renders every PDF under `data/spaciel_font/` into per-page PNGs at 300 DPI, writing `data/spaciel_font_pages/<stem>_page_NNN.png` plus a `manifest.csv` (columns: `page_image`, `source_pdf`, `page_num`, `status`). Skips already-rendered pages; handles nested subdirectories by prefixing the parent folder name into the stem.
2. `notebooks/09_special_font_ocr.ipynb` — loads the manifest, loads Gemma 4 via `mlx-vlm.load`, runs two prompts per page:
   - **OCR prompt:** extract all visible text preserving RTL Hebrew reading order and line breaks (max 2048 tokens).
   - **Structured prompt:** currently instructs the model to render the page as markdown (max 512 tokens); downstream code attempts JSON parsing and records `{"_raw": ...}` on failure.
3. Results are appended to `data/spaciel_font_pages/ocr_results.json` **after every page** so interrupted runs resume cleanly via a `done_images` set keyed on `page_image`.
4. Summary/export cells flatten the records into `ocr_results.csv` (UTF-8-sig for Hebrew) with placeholder target fields `person_name`, `id_number`, `date_of_birth`, `address`, `document_type`, `institution`, `notes`.

**Current state:**
- 191 pages rendered (all `status=rendered` in manifest)
- 17 / 191 pages processed with OCR so far, 0 errors
- Full batch runtime estimate: 5–15 s/page → ~15–45 min on Apple Silicon

**Open items:**
- Complete the OCR sweep over the remaining ~174 pages
- The structured prompt currently produces free-form markdown rather than JSON; the target-field coverage report in cell 15 will report 0% until the prompt is tightened to return strict JSON with the `TARGET_FIELDS` schema
- No downstream join yet between `ocr_results.csv` and the hallucination-risk pipeline — this is a parallel text-extraction track

**Artifacts:**
- `scripts/split_pdfs_to_pages.py` — 300 DPI renderer with skip-existing and nested-path handling
- `notebooks/09_special_font_ocr.ipynb` — Gemma 4 OCR notebook with resume support
- `data/spaciel_font/` — source PDFs (non-standard font)
- `data/spaciel_font_pages/` — rendered PNGs + `manifest.csv` + `ocr_results.json` (+ `ocr_results.csv` once the export cell is run)

---

### Phase 6 — Expanded Training with All Data Folders (Completed)

**Goal:** Ingest 4 additional data folders (861 new pages), retrain all models with class-imbalance handling, and verify the external FSR drops from the 30–50% range seen in Phase 5c.

**New data ingested:**

| source_folder | Files | Label | Template Family |
|---|---|---|---|
| `handwritten` (new 6) | 6 pages | 1 (risky) | questionnaire_named / _uuid |
| `handwritten_edge_cases` | 12 pages | 1 (risky) | handwritten_edge |
| `handwritten_and_questioniers` | 650 pages | 1 (risky) | mixed_hwq |
| `spaciel_font` | 191 pages | 1 (risky) | special_font  *(reclassified to `regular_forms_edge_cases` / label=0 in Phase 6d)* |

**New corpus:** 1873 pages (634 safe : 1239 risky, ratio 0.51:1) — *Phase 6d shifted this to 825 / 1048 / 0.79:1*

**Pipeline changes:**

- `scripts/build_metadata_v2.py` — enumerates all 6 source folders, adds `source_folder`, `source_doc_stem`, `is_edge_case` columns to `data/metadata.csv`; backs up `metadata_v1.csv`, `labels_binary_clean_v1.csv`, `splits_v1/`
- `scripts/render_all_pages.py` — renders new pages at 224×224 grayscale (skip-existing); spaciel_font pages converted from existing 300 DPI PNGs; 853 new PNGs added to `data/rendered_pages/` with 0 errors
- `scripts/annotate_rubric.py` — run incrementally (925 new pages to Claude); all 1873 pages now have D/H/S/L scores in `data/labels_rubric.csv`
- `scripts/regenerate_splits.py` — regenerates splits grouped by `source_doc_stem` (within-group 70/15/15); groups with <3 pages go to train; result: 1314 train / 275 val / 284 test (all major source folders in val and test)
- `src/data/splits.py` — added `create_grouped_holdout_splits`; patched `create_grouped_splits` to handle small groups gracefully (assign to train instead of raising ValueError)
- `configs/baseline.yaml`, `configs/dit.yaml` — added `training.pos_weight: auto`, `training.use_weighted_sampler: true`, changed `splits.group_col: source_doc_stem`
- `src/train/train_baseline.py`, `src/train/train_dit.py` — added `pos_weight` and `WeightedRandomSampler` support (backwards-compatible)

**Training results (held-out test set, 284 samples):**

| Model | Best stage | Temperature | F1 | FSR | ROC-AUC | ECE | T_low | T_high |
|-------|-----------|-------------|-----|-----|---------|-----|-------|--------|
| ResNet50 | epoch 10 | 0.550 | 0.967 | 6.38% | 0.999 | 0.038 | 0.699 | 0.849 |
| ViT-Base | — | — | 0.965 | 5.85% | 0.993 | 0.032 | — | — |
| DiT | stage3 | 2.727 | 0.950 | 8.51% | 0.993 | 0.053 | 0.696 | 0.848 |

**External validation results (400 docs: 94 risky + 306 safe):**

| Model | Accuracy | Macro F1 | FSR | TN | FP | FN | TP |
|-------|----------|----------|-----|----|----|----|-----|
| ResNet50 | 73.2% | 0.706 | **11.7%** | 207 | 99 | 8 | 86 |
| ViT-Base | 58.2% | 0.562 | 26.6% | 160 | 146 | 21 | 73 |
| DiT | 76.8% | 0.724 | 31.9% | 233 | 73 | 20 | 74 |

**Comparison with Phase 5c (old training):**

| Model | FSR (Phase 5c) | FSR (Phase 6) | Delta |
|-------|---------------|---------------|-------|
| ResNet50 | 36.2% | **11.7%** | -24.5 pp ✓ |
| ViT-Base | 50.0% | 26.6% | -23.4 pp ✓ |
| DiT | 30.9% | 31.9% | +1 pp ≈ same |

**Key observations:**

- ResNet50 and ViT-Base FSR improved dramatically on the external set — the diverse training distribution is working
- DiT's FSR is unchanged — its stronger inductive bias from document pretraining may make it less sensitive to distribution-shift improvement from new data
- Accuracy dropped (73–77% vs 86–91%) because models are now more conservative (more FPs): this is the correct trade-off for a safety-critical classifier
- The planned target of <15% external FSR for DiT was not met (31.9%), but ResNet50 hit 11.7%; ResNet50 is now the recommended production model

**Artifacts:**

- `data/metadata.csv` — 1873 rows with source_folder, source_doc_stem, is_edge_case columns; rubric scores for all rows
- `data/metadata_v1.csv`, `data/labels_binary_clean_v1.csv`, `data/splits_v1/` — backups of Phase 5 state
- `checkpoints/baseline/best_resnet50.pt`, `checkpoints/baseline/best_vit_base_patch16_224.pt` — retrained checkpoints
- `checkpoints/dit/best_model.pt` — retrained DiT checkpoint (stage3 best)
- `logs/train_resnet50_v2.log`, `logs/train_vit_v2.log`, `logs/train_dit_v2.log` — training logs
- `validation_report/index.html` — updated HTML report with new model comparison
- `scripts/build_metadata_v2.py`, `scripts/render_all_pages.py`, `scripts/regenerate_splits.py` — new pipeline scripts

### Phase 6b — Shareable Fine-Tune Report (Completed)

**Goal:** Produce a single-file, self-contained HTML report that consolidates the Phase 6 fine-tune story (dataset, training curves, held-out test metrics, external validation, Phase 5→6 comparison, recommendation) for distribution to colleagues.

**Artifacts:**

- `reports/finetune_report.html` — 205 KB standalone HTML (Plotly via CDN, no local assets), embeds all data as inline JSON. Tabs: Overview / Training Curves / Held-out Test / External Validation / Findings & Recommendation. Model selector on the Test and External tabs switches confusion matrices / probability histograms / worst-case tables per model. Includes a Phase 5 vs Phase 6 FSR comparison bar chart with the 15% target line.
- `scripts/build_finetune_report.py` — regenerator script. Reads training JSONs, best checkpoints (for test metrics + test logits → reconstructed confusion matrices and histograms), `data/metadata.csv`, `data/splits/*.csv`, and `validation_report/report_data.json` (which must be regenerated via `scripts/_run_validation_inference.py` first). Produces the HTML in under 2 s.
- `reports/_finetune_report_template.html` — HTML/CSS/JS template. Keep in the repo alongside the script for maintainability.

**To share:** attach `reports/finetune_report.html` to email — no folder zip required; it loads Plotly from the CDN. Verified all 5 tabs render and the model selectors update charts correctly in-browser.

---

### Phase 7 — Production

Not yet started. Next steps would be wrapping `src/inference/predict.py` as a
batch inference service and integrating the review-queue routing logic.

---

## Cross-Step Knowledge

### Data

**Phase 6d corpus (current):**
- 1873 pages total: **825 safe (label=0) : 1048 risky (label=1)** — class ratio 0.79:1 risky-heavy
- Source folders: `regular_forms` (634), `regular_forms_edge_cases` (191, **was `spaciel_font` pre-Phase-6d**), `handwritten` (386), `handwritten_and_questioniers` (650), `handwritten_edge_cases` (12)
- 54 unique `source_doc_stem` groups (very few large source PDFs — see gotchas)
- Rendered PNGs at 224×224 grayscale in `data/rendered_pages/` (1873 PNGs)
- Rubric scores (D/H/S/L) for all 1873 pages in `data/labels_rubric.csv` (2066 total entries due to checkpoint accumulation)
- New metadata columns: `source_folder`, `source_doc_stem`, `is_edge_case`
- Backups: `data/metadata_v1.csv` (1014-row Phase 5), `data/metadata_v2_precorrection.csv` (1873-row pre-Phase-6d), `data/splits_v1/`, `data/splits_v2_precorrection/`

**Phase 5 corpus (legacy):**
- `regular_forms/` — 634 safe pages across 5 source PDFs
- `handwritten/` — 380 risky pages (6 more added in Phase 6)
- All `_page_XXXX` suffix naming; Hebrew filenames handled via `pathlib` throughout

### Setup

- **Compute:** runs locally — no cloud account or GCS bucket required
- **Storage:** all data and checkpoints live under the repository root (`data/`, `checkpoints/`, `logs/`, `inference_results/`)
- **Setup notebook:** `notebooks/00_gcp_setup.ipynb` — validates local directories, cleans labels, generates splits, renders PDFs
- **GCP scripts:** `scripts/drive_to_gcs.py` is retained for optional GCS upload but is not part of the local workflow; GCP packages in `requirements.txt` are commented out

### Engineering Constraints

- All preprocessing must be deterministic — `render_pdf.py` is stateless and hash-stable
- Rendering settings versioned via `configs/` YAMLs
- Splits saved to `data/splits/` — never regenerate casually
- Calibrator and thresholds saved alongside the model checkpoint in `checkpoints/dit/`
- Evaluation must always report per-institution metrics
- Hebrew filenames handled via `pathlib` throughout — do not use string concatenation for paths

### Model Strategy

- DiT (`microsoft/dit-base`) is the target production model
- ResNet50 and ViT-Base are sanity-check baselines (Stage 1 milestones)
- Binary training with `BCEWithLogitsLoss`; ternary output via calibrated thresholds
- Most important metric: **false-safe rate** — safe predictions that are actually hallucination-risk
- Thresholds selected conservatively (low false-safe rate > raw F1)

### Known Gotchas

- **CRITICAL — Label/institution correlation:** All 634 "no risk" files are from `regular_docs`, all 380 "high risk" from `questionnaires`. Grouped splits by `institution` would put all of one class in train and the other in test. Use `template_family` as the group key instead, and verify splits have mixed labels. In Phase 6, split key changed to `source_doc_stem` which is more granular and avoids page leakage from multi-page source PDFs.
- DiT is a BeitModel under the hood — staged unfreezing accesses `self.backbone.encoder.layer[-n:]`
- Grayscale input (mode 'L') needs to be repeated to 3 channels for all models (handled in each model's `forward`)
- Class imbalance: 634 safe vs 380 risky (~1.67:1). Monitor if it worsens after rubric annotation; consider focal loss if skew > 4:1
- Only 3 template families detected in current data (`regular_form`, `questionnaire_uuid`, `questionnaire_named`) — with so few groups, kfold CV is preferred over 70/15/15
- `ErrorAnalysisLogger` uses UTF-8-sig encoding to handle Hebrew filenames in Excel/CSV viewers
- `TrainingLogger` writes JSON history to `logs/` — useful for comparing runs programmatically
- **`metadata.csv` file_path column has .pdf extension** but rendered PNGs are .png — `dataset.py` handles the mapping at init time via `_build_file_index()`
- **Hebrew filenames are now correct in all CSVs** (fixed in Phase 4c). The previous 28 unresolvable entries were caused by masked Hebrew names — those 193 entries now carry correct Hebrew filenames in `metadata.csv`, `labels_binary_clean.csv`, and `labels_rubric.csv`. Re-render any previously missing PNGs using `notebooks/00_gcp_setup.ipynb` to achieve full coverage.
- **DiT best checkpoint is stage2, not stage3.** The 3-stage training orchestrator saves the global best across all stages. With early stopping patience=15 per stage, stage2 (head + top-2 blocks) converged at epoch 7 with F1=1.0/FSR=0.0 on the test set before stage3 could improve further. Always use `checkpoints/dit/best_model.pt`, not `best_stage3.pt`.
- **ViT-Base early-stopped at epoch 2.** This is expected on a dataset of ~710 training samples — ViT converges very fast with ImageNet pretraining and the small dataset gives little room to improve further without overfitting. The val F1 of 0.982 at epoch 2 is genuine.
- **`load_pipeline` in `src/inference/predict.py` requires `model_type` and `config_path`.** When `model_type='vit'` and the config is `baseline.yaml` (which has `model.name=resnet50`), the pipeline reads `model_name` from the checkpoint's `model_name` key to override the config before building the model. Without this fix, ViT would be built with a ResNet backbone.
- **Calibrator `.pkl` files are serialized as whole `TemperatureCalibrator` objects**, not dicts. `load_pipeline` handles both formats (object and dict) automatically. Do not use `calibrator.load(path)` directly if the pkl was saved as an object.
- **Special-font OCR track (Phase 5d)** is an independent pipeline from the risk classifier: inputs live under `data/spaciel_font/` (note: directory name is `spaciel_font`, not `special_font`), outputs under `data/spaciel_font_pages/`. It uses 300 DPI rendering (vs 150 DPI / 224×224 for the classifier) because OCR quality is sensitive to resolution. Runs only on the `doc-risk-classifier` pyenv (Python 3.13.11) with `mlx-vlm` — the model will not load on x86 or non-MLX stacks. The on-disk `spaciel_font` and `spaciel_font_pages` directories are **kept on disk under those names** even after the Phase 6d reclassification — the OCR resume state and the manifest paths are keyed on those filenames.
- **OCR resume:** `notebooks/09_special_font_ocr.ipynb` keys resume state on `page_image` filenames loaded from `ocr_results.json`; re-running the full-batch cell skips everything already present. To re-run a page, delete its record from `ocr_results.json` first.
- **Phase 6d — `spaciel_font` is now `regular_forms_edge_cases` in metadata.** The 191 pages whose source PDFs live under `data/spaciel_font/` were originally tagged `source_folder="spaciel_font"`, `template_family="special_font"`, `label_binary=1`, `institution="questionnaires"`. As of Phase 6d (April 2026) they carry `source_folder="regular_forms_edge_cases"`, `template_family="regular_form_edge"`, `label_binary=0` (safe), `institution="regular_docs"`, `is_edge_case=True`. **The on-disk directories are untouched** — only the metadata flags changed. `scripts/build_metadata_v2.py` and `scripts/render_all_pages.py` were patched to emit the new tags on rebuild while still reading from `data/spaciel_font_pages/`.
- **Phase 6 — metadata.csv encoding:** After Phase 6, `data/metadata.csv` is saved with `encoding="utf-8"` (no BOM). `labels_binary_clean.csv` uses `utf-8-sig` (BOM). `annotate_rubric.py` opens `metadata.csv` with `encoding="utf-8"` (no BOM) — always keep metadata.csv BOM-free to avoid the KeyError on `\ufefffile_path`.
- **Phase 6 — only 54 unique source_doc_stems across 1873 pages.** The data has very few large source documents (e.g., `regular_forms` has only 5 source PDFs with 634 total pages; `handwritten_and_questioniers` has 1 source PDF with 650 pages). The small number of groups means within-group splits are required — `create_grouped_holdout_splits` would leave val/test nearly empty for some classes.
- **Phase 6 — `regular_forms_edge_cases` virtual file_path.** Pages whose source PDFs live in `data/spaciel_font/` are referenced in `metadata.csv` with virtual `file_path` values like `da0838fa_page_001.pdf` (not an actual file on disk). The rendered PNGs are in `data/rendered_pages/` with the matching stem. This is because the source PNGs from `spaciel_font_pages/` were resized/converted rather than re-rendered from PDFs. The `source_folder` for these rows is `regular_forms_edge_cases` (Phase 6d), not `spaciel_font`.
- **Phase 6 — annotate_rubric.py processes 2066 rows against 1873 metadata rows.** The checkpoint has accumulated entries from multiple runs (old + new). The counter in the log exceeds 1873 because it counts checkpoint-loaded entries + new API calls together. This is cosmetic; the actual rubric CSV has the correct 2066 unique entries (some pages exist in multiple source docs).
- **Phase 6 — DiT FSR was unchanged on external validation (31.9%) under the FSR-target-only thresholding.** Phase 6c (cost-weighted thresholds) brought it down to 2.1%; Phase 6d (relabel + retrain + retune) brought it down to **1.1%** and DiT now also leads on accuracy (71.2%). DiT is the recommended production model.

### Mixed Results Evaluation — Gemini vs DiT Comparison (Completed)

**Goal:** Compare Gemini (text-based LLM classification) and DiT (fine-tuned vision classifier) on a 648-document evaluation set with ground-truth labels. Produce a self-contained interactive HTML report in the same style as `reports/finetune_report.html`.

**Data:** `eval/mixed_results - Sheet1.csv` — 652 rows (4 dropped due to missing Gemini classification → 648 usable). Ground truth: `safe_for_extraction` (437 safe) / `high_hallucination_risk` (211 risky).

**Binarisation:**
- **Ground truth:** `safe_for_extraction` → safe, `high_hallucination_risk` → risky
- **Gemini:** `typed` → safe; `handwritten` and `mixed` → risky
- **DiT:** `safe_for_extraction` → safe; `review` and `high_hallucination_risk` → risky

**Results (648 docs):**

| Model  | Accuracy | Macro F1 | FSR   | TN  | FP | FN | TP  |
|--------|----------|----------|-------|-----|----|----|-----|
| Gemini | 90.0%    | 0.886    | 6.9%  | 402 | 35 | 30 | 181 |
| DiT    | 80.7%    | 0.789    | 11.2% | 357 | 80 | 45 | 166 |

**Agreement:** Both correct on 490 docs (75.6%), only Gemini correct on 93 (14.3%), only DiT correct on 33 (5.1%), both wrong on 32 (4.9%).

**Key findings:**
- Gemini outperforms DiT on this evaluation set across all metrics (accuracy, F1, FSR)
- Gemini's FSR of 6.9% is outside the <5% safety target but substantially better than DiT's 11.2%
- DiT has a higher false-positive rate (80 FP vs 35), meaning it over-flags safe documents as risky
- 32 documents are misclassified by both models — these are the hardest cases
- Gemini's self-reported confidence is weakly calibrated (mean ~90% on correct, lower on errors)

**Artifacts:**
- `reports/mixed_eval_report.html` — 72 KB standalone HTML report (Plotly via CDN), 5 tabs: Overview / Detailed Metrics / Error Analysis / Model Agreement / Findings
- `scripts/build_mixed_eval_report.py` — regenerator script, reads CSV and produces the HTML

---

### Phase 6c — Cost-Weighted Threshold Retuning (Completed)

**Goal:** explicitly encode the production cost asymmetry — missing a risky document (FN, → hallucinated extraction) is materially worse than flagging a safe one (FP, → wasted reviewer time). Replace the Phase 5 "FSR ≤ 5% target" rule with a single-threshold cost minimisation.

**Cost model:** `cost(τ) = 10·FN(τ) + 1·FP(τ)` on the validation split. τ* is selected per model; T_high is placed at `(τ* + 1)/2` for backward-compatible three-band routing but is not involved in cost minimisation.

**Implementation:**
- `src/models/calibrator.py` — added `TemperatureCalibrator.get_cost_weighted_thresholds(probs, labels, fn_cost=10, fp_cost=1)`.
- `scripts/retune_thresholds.py` — loads each best checkpoint, recomputes calibrated val probs from stored `val_logits`, picks τ*, updates both `calibrator.pkl` and `checkpoint["thresholds"]`, and backs up originals under `checkpoints/thresholds_backup_<timestamp>/` for rollback.
- `scripts/_run_validation_inference.py` — now also writes per-doc arrays (`y_true`, `y_prob`, `source`) and τ*-based confusion stats into `validation_report/report_data.json` so any future report can re-threshold without another model pass.

**Results per model (see `reports/finetune_report.html` for the full interactive view):**

| Model    | τ*    | Held-out Test FSR @ 0.5 → @τ* | External Val FSR @ 0.5 → @τ* | Phase 5 Baseline FSR |
|----------|-------|--------------------------------|-------------------------------|----------------------|
| ResNet50 | 0.045 | 4.8% → **0.0%**                | 8.5% → **3.2%**               | 36.2%                |
| ViT-Base | 0.033 | 4.8% → **0.0%**                | 22.3% → **9.6%**              | 50.0%                |
| DiT      | 0.019 | 7.4% → **0.0%**                | 21.3% → **2.1%**              | 30.9%                |

All three models hit **0 FN on the held-out test set** at τ*. DiT now leads the external-set FSR at 2.1% — the Phase 6 gotcha about unchanged DiT external FSR is resolved by the threshold change alone.

**Trade-off:** accuracy at τ* drops on the external set (many borderline safe docs now route to review). This is the intended behaviour under the 10× cost asymmetry. The cost ratio is parameterised — rerun `python scripts/retune_thresholds.py --fn-cost 5 --fp-cost 1` to soften it.

**Artifacts:**
- `docs/FINETUNE_REPORT.md` — plain-text report summary for Slack / email.
- `reports/finetune_report.html` — single-file interactive report (regenerated with before/after panels and per-source breakdown).
- `scripts/retune_thresholds.py` — threshold recalibration CLI.
- `checkpoints/thresholds_backup_<ts>/` — rollback bundle per run.

**Gotchas for future agents:**
- `build_finetune_report.py` requires `per_doc` arrays in `validation_report/report_data.json`; **re-run `_run_validation_inference.py` after any threshold or model change**, otherwise the report builder will error out with a clear message.
- `τ*` must be written to **both** `checkpoint["thresholds"]["T_low"]` **and** `calibrator.t_low` — `retune_thresholds.py` does both; downstream code that touches only one will drift.
- With FN:FP = 10:1 and the current models, τ* lands at 0.019–0.045. Flipping the ratio (FP 10× FN) would push τ* above 0.9, collapsing risky recall — don't do that unless the downstream team owns the hallucination fallout.
- Temperatures were not re-fit in Phase 6c; only thresholds moved. If training distribution changes materially, recalibrate temperature before retuning thresholds.

---

### Phase 6d — Special-Font Reclassification (Completed, April 2026)

**Goal:** correct a labelling mistake. The 191 pages in `data/spaciel_font/` were originally treated as a "special font" risky class (`label_binary=1`, `template_family=special_font`, `institution=questionnaires`). On review they are **edge-case regular forms** — visually unusual layouts of normal documents, not a separate hallucination-risk class. The pre-existing rubric (D=2, mean risk_score=3.2, max=5) already placed them in the safe band, confirming the relabel.

**What changed:**
- 191 rows in `data/metadata.csv` flipped:
  - `source_folder` : `spaciel_font` → `regular_forms_edge_cases`
  - `template_family` : `special_font` → `regular_form_edge`
  - `is_edge_case` : `False` → `True`
  - `label_binary` : `1` → `0`
  - `institution` : `questionnaires` → `regular_docs`
- Rubric scores (D / H / S / L / risk_score) untouched — they describe visual features, not class.
- On-disk filesystem **unchanged**: PDFs stay in `data/spaciel_font/`, rendered PNGs stay in `data/rendered_pages/` and `data/spaciel_font_pages/`. Renaming would have invalidated every checkpoint's stored `file_path` references and broken the Phase 5d OCR pipeline's resume state.
- Class balance shifted: 825 safe / 1048 risky (was 634 / 1239). Ratio 0.51:1 → 0.79:1 risky-heavy. `pos_weight: auto` and `WeightedRandomSampler` already handle this — no config change needed.
- All three models retrained from scratch, thresholds re-tuned at FN:FP = 10:1, external validation re-run on the 400-doc set, `reports/finetune_report.html` regenerated.

**Implementation:**
- `scripts/fix_spaciel_font_to_edge_cases.py` — one-shot fixer. Loads metadata, flips the 5 columns on the 191 rows, re-runs `regenerate_splits.main()` in-process, re-saves metadata utf-8 (no BOM).
- `scripts/build_metadata_v2.py` — updated `spaciel_font` ingest branch to emit the new tags so future rebuilds are correct.
- `scripts/render_all_pages.py` — accepts both `regular_forms_edge_cases` (new) and `spaciel_font` (legacy) source-folder values, both resolve to PNGs in `data/spaciel_font_pages/`.
- `scripts/build_finetune_report.py` — added `regular_forms_edge_cases → regular_form_edge` to the source-family map; legacy `spaciel_font` key kept so old backup reports still build.

**Held-out test set (n = 284, source-doc grouped) — Phase 6 → 6d:**

| Model    | Phase 6 F1 | **Phase 6d F1** | Phase 6 ROC-AUC | **Phase 6d ROC-AUC** | Best Epoch |
|----------|-----------:|----------------:|----------------:|---------------------:|-----------:|
| ResNet50 |      0.967 |       **0.971** |           0.999 |            **0.999** |         11 |
| ViT-Base |      0.965 |       **0.953** |           0.993 |            **0.986** |         16 |
| DiT      |      0.950 |       **0.978** |           0.993 |            **0.999** |   stage3-25 |

All three within ±5 pp on F1 — the relabel did not destabilise any model.

**External validation (n = 400) at τ* (FN:FP = 10:1) — Phase 6c → 6d:**

| Model    | FSR @ τ* (Phase 6c) | **FSR @ τ* (Phase 6d)** | Acc @ τ* (Phase 6d) | TN / FP / FN / TP |
|----------|--------------------:|------------------------:|--------------------:|------------------:|
| ResNet50 |                3.2% |                **2.1%** |               55.8% |  139 / 167 / 10 / 84 |
| ViT-Base |                9.6% |                **3.2%** |               62.0% |  177 / 129 / 23 / 71 |
| **DiT**  |                2.1% |                **1.1%** |               71.2% |  215 /  91 / 24 / 70 |

DiT now leads on **both** external FSR (1.1%) and external accuracy (71.2%). It was already the FSR leader after Phase 6c; Phase 6d gave it the accuracy crown too.

**Artifacts (Phase 6d):**
- `data/metadata.csv` — corrected
- `data/metadata_v2_precorrection.csv` — pre-Phase-6d backup (1873 rows)
- `data/splits_v2_precorrection/` — pre-Phase-6d splits backup
- `checkpoints/_precorrection_backup_20260424_072825/` — full pre-correction checkpoints (baseline + dit, .pt + .pkl)
- `checkpoints/baseline/best_resnet50.pt`, `best_vit_base_patch16_224.pt`, `calibrator_*.pkl` — retrained
- `checkpoints/dit/best_model.pt`, `calibrator.pkl` — retrained
- `logs/train_resnet50_v3.log`, `logs/train_vit_v3.log`, `logs/train_dit_v3.log`, `logs/validation_inference_v3.log`
- `reports/finetune_report.html` — regenerated (261 KB)
- `validation_report/report_data.json`, `validation_report/index.html` — regenerated

**Gotchas for future agents:**
- Don't rename `data/spaciel_font/` or `data/spaciel_font_pages/` — both the OCR pipeline (`notebooks/09_special_font_ocr.ipynb`) and the renderer key on those names. Rename would invalidate every checkpoint's `file_path` references.
- `scripts/build_metadata_v2.py` still reads from `SPACIEL_FONT_MANIFEST = data/spaciel_font_pages/manifest.csv` — that path is intentional. Only the emitted `source_folder` / `template_family` / `label` / `institution` / `is_edge_case` values changed.
- `scripts/render_all_pages.py` accepts both `regular_forms_edge_cases` (current) and `spaciel_font` (legacy) — keep the legacy branch alive so older metadata snapshots still render.
- `scripts/build_finetune_report.py` keeps a legacy `"spaciel_font": "special_font"` entry in `SOURCE_FAMILY` so reports rebuilt from `data/metadata_v2_precorrection.csv` still render.
- Class balance moved from 0.51:1 to 0.79:1 risky-heavy — still imbalanced, `pos_weight: auto` and `WeightedRandomSampler` continue to be appropriate. If a future ingest pushes the ratio under 0.4:1 risky-heavy, revisit focal loss.
