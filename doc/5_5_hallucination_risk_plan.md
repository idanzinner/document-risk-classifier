# 5/5 Development Plan — Single-Page Hebrew PDF Hallucination-Risk Classifier

## Goal

Build a **production-ready classifier** for single-page Hebrew PDFs that predicts:

- `safe_for_extraction`
- `review`
- `high_hallucination_risk`

This is **not** a handwriting detector.

It is a **document extractability / hallucination-risk classifier**.

A document is safe when it contains enough clear, reliable, document-specific information to support downstream extraction without guessing.

A document is risky when it is mostly blank, partial, handwriting-dominant, weakly filled, or visually unclear enough that the downstream model is likely to invent information.

---

## Problem Framing

### What the model should learn
The model should learn whether a page has:

1. **Enough document-specific content**
2. **Enough reliable structure**
3. **Low dependence on unclear handwriting**
4. **Enough visual clarity for safe extraction**

### What the model should *not* optimize for
Do **not** frame the task as:

- handwritten vs printed
- OCR quality estimation
- Hebrew text understanding
- prompt optimization
- LLM classification

This is primarily a **document image classification** problem.

---

## Final Labeling Policy

### Production outputs
- `safe_for_extraction`
- `review`
- `high_hallucination_risk`

### Existing training labels
You currently have **binary labels**. Keep them.

Use the scoring rubric below to:
- refine borderline cases
- build a diagnostic subset
- tune thresholds
- define the `review` band

---

## Quantitative Risk Rubric

Use 4 annotation components.

### 1. Information Density (`D`)
How much **document-specific content** is present, excluding boilerplate/template text?

Score:
- `0` = almost no real content; mostly blank/template
- `1` = sparse content; a few filled fields/fragments only
- `2` = moderate useful content; enough to understand major parts
- `3` = high content density; substantial filled content/report/letter

**Document-specific content** means content describing this patient/event/case, not generic questions or headings.

---

### 2. Handwriting Dependence (`H`)
How much does understanding the document depend on handwriting?

Score:
- `0` = none / irrelevant
- `1` = minor only; signature, checkmarks, tiny notes
- `2` = meaningful but secondary
- `3` = primary or essential source of information

Rule:
This is **not** “how much handwriting exists”.
This is “if handwriting is ignored, how much useful content remains?”

---

### 3. Structure Completeness (`S`)
Only apply to structured forms/questionnaires/checklists.

Score:
- `0` = not a form / not relevant
- `1` = mostly complete / well filled
- `2` = partially filled, important blanks remain
- `3` = mostly blank / nearly empty

---

### 4. Legibility Risk (`L`)
How hard is it to reliably read the informative content?

Score:
- `0` = clear
- `1` = slightly degraded
- `2` = noticeably hard to read
- `3` = unreliable due to scan quality or illegible handwriting

---

## Risk Score

Compute:

```text
risk_score = (3 - D) + H + S + L
```

Range: `0` to `12`

### Default interpretation
- `0–3` → `safe_for_extraction`
- `4–6` → `review`
- `7–12` → `high_hallucination_risk`

---

## Override Rules

### Hard Risk Override
Force `high_hallucination_risk` if any of these apply:
- mostly blank form with only marks/checks/signature/date
- useful information is fragmentary and cannot support extraction
- handwriting is dominant and core content is not reliably legible
- patient-specific content is largely missing even though template text is present

### Hard Safe Override
Force `safe_for_extraction` if:
- substantial printed report/summary/letter exists
- enough patient-specific content is clearly visible
- handwriting is incidental and non-critical

---

## Borderline Definition

Borderline cases are not random difficult samples.
They are cases where one or more signals conflict.

Examples:
- structured form with only a few filled answers
- printed document with one important handwritten section
- form with many questions and very little patient-specific content
- partial questionnaire with checkmarks but little context
- mostly printed page with scan artifacts and sparse handwriting

### Borderline band
Use:
- `risk_score 4–6`
- plus any sample where annotators disagree
- plus any binary-labeled sample that contradicts the rubric

These become the **borderline diagnostic set**.

---

## Recommended Data Strategy

### Dataset layout
Each input is:
- one PDF
- one page
- one label

### Splits
Use grouped splits by the strongest available source variable:
- institution
- template family
- document source
- upload/source batch if available

Do **not** randomly split similar forms across train and validation.

### Recommended split
- 70% train
- 15% validation
- 15% test

If dataset is small:
- 5-fold grouped cross-validation for model selection
- final held-out grouped test set for the last evaluation only

---

## Development Architecture

### Input pipeline
1. Load PDF
2. Render page to image
3. Convert to grayscale
4. Resize with aspect-ratio preservation and padding
5. Store rendered cache

### Rendering settings
- Renderer: `PyMuPDF`
- DPI: `150–200`
- Color mode: grayscale
- Preserve aspect ratio
- No aggressive crop

---

## Model Strategy

### Recommended model order

#### Baseline 1 — EfficientNet / ResNet
Purpose:
- sanity check
- fast baseline
- verifies task learnability

#### Baseline 2 — Generic ViT
Purpose:
- compare generic vision transformer vs document-native model

#### Target model — DiT
Purpose:
- strongest fit for document image classification
- document-native visual backbone

### Do not use for v1
- DSPy / GEPA
- T5 / Gemma / LLM prompt classifiers
- OCR-first routing
- multimodal extraction models as classifiers

---

## Training Plan

### Stage 0 — Baseline labeling audit
Before model training:
1. sample 100 borderline documents
2. score with `D/H/S/L`
3. compare score vs existing binary labels
4. flag contradictions
5. fix annotation guide before scaling

This stage is mandatory.

---

### Stage 1 — Baseline models
Train:
- EfficientNet or ResNet
- ViT

Measure:
- grouped validation F1
- per-institution performance
- calibration quality
- false-safe rate

Purpose:
- establish a floor
- catch leakage or label inconsistency early

---

### Stage 2 — DiT training
Training schedule:
1. freeze backbone, train classifier head
2. unfreeze top transformer blocks, low LR
3. full fine-tune only if validation materially improves

Recommended loss:
- `BCEWithLogitsLoss` for binary training
- optionally focal loss if positive/negative balance becomes skewed later

---

## Output Policy

### Internal model output
Train a binary score:
- low risk
- high risk

### Production output mapping
Map score to:
- `safe_for_extraction`
- `review`
- `high_hallucination_risk`

Use calibrated thresholds.

Example:
- score < `T_low` → safe
- score between `T_low` and `T_high` → review
- score > `T_high` → high risk

Thresholds should be selected from the validation set based on business cost.

---

## Calibration Plan

Calibration is not optional.

### Required
- train model
- collect validation logits
- apply temperature scaling
- select decision thresholds from calibrated probabilities

### Why
Raw model confidence is usually unreliable.
You need thresholding for:
- safe auto-processing
- review routing
- high-risk blocking

---

## Evaluation Metrics

### Primary metrics
- document-level F1
- precision for `safe_for_extraction`
- recall for `high_hallucination_risk`
- false-safe rate
- review rate

### Required secondary metrics
- ROC-AUC
- PR-AUC
- Expected Calibration Error (ECE)
- per-institution F1 / recall / false-safe rate

### Most important production metric
**False safe rate**:
documents classified as safe that are actually hallucination-risk.

This is likely the costliest failure.

---

## Error Analysis Plan

Create error slices:

1. printed report / summary / letter
2. structured form mostly complete
3. structured form partially complete
4. empty or nearly empty questionnaire
5. questionnaire with minor handwritten marks
6. handwriting-dominant page
7. poor scan / legibility problems

For every false safe and false risky case, log:
- prediction
- confidence
- institution/template
- risk rubric score if available
- scan quality notes
- handwriting presence notes

---

## Augmentation Policy

Use only scan-realistic augmentations:

- slight rotation
- blur
- contrast/brightness shift
- JPEG compression artifacts
- mild perspective warp
- salt-and-pepper / scan noise
- border shadows

Do **not** use:
- aggressive cropping
- arbitrary artistic augmentations
- transformations that remove small handwritten evidence

---

## Borderline Diagnostic Set

Create a fixed diagnostic subset of `100–150` samples.

Include:
- `risk_score 4–6`
- annotator disagreements
- contradictory binary labels
- sparse handwritten forms
- mostly blank questionnaires
- printed pages with critical handwritten content
- degraded scans

This set should **not** be used casually.
Use it for:
- threshold tuning
- model comparison
- qualitative debugging
- regression tests

---

## Suggested Repository Structure

```text
project/
├── data/
│   ├── raw_pdfs/
│   ├── rendered_pages/
│   ├── metadata.csv
│   ├── labels_binary.csv
│   ├── labels_rubric.csv
│   └── splits/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_rendering_checks.ipynb
│   ├── 03_label_consistency.ipynb
│   ├── 04_baseline_training.ipynb
│   ├── 05_dit_training.ipynb
│   └── 06_calibration_eval.ipynb
├── src/
│   ├── data/
│   │   ├── render_pdf.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── models/
│   │   ├── resnet_baseline.py
│   │   ├── vit_baseline.py
│   │   ├── dit_classifier.py
│   │   └── calibrator.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_dit.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── service_schema.py
│   └── utils/
│       ├── metrics.py
│       ├── logging.py
│       └── visualization.py
├── configs/
│   ├── baseline.yaml
│   ├── dit.yaml
│   └── inference.yaml
└── README.md
```

---

## Development Milestones

### Milestone 1 — Label audit
- finalize rubric
- score 100 borderline samples
- identify contradictions
- finalize annotation guide

### Milestone 2 — Data pipeline
- PDF renderer
- image cache
- metadata table
- grouped split generation

### Milestone 3 — Baselines
- ResNet/EfficientNet
- ViT
- grouped validation
- calibration baseline

### Milestone 4 — DiT
- fine-tune DiT
- compare against baselines
- evaluate on diagnostic set

### Milestone 5 — Thresholds
- calibrate probabilities
- choose `safe/review/risky` thresholds
- produce confusion analysis

### Milestone 6 — Production service
- export model
- batch inference endpoint
- structured logging
- confidence routing
- review queue integration

---

## Annotation Guide for Borderline Cases

### Safe examples
- printed medical summary with complete content
- structured form with enough filled patient-specific fields
- printed report with a signature or tiny handwritten note
- letter with substantial clear printed narrative

### Risky examples
- empty questionnaire
- questionnaire with only a name, date, or a few checkmarks
- page where useful information depends mainly on handwriting
- handwriting-only note
- mixed form where printed structure exists but patient-specific content is too sparse

### Review examples
- partially filled form with some readable answers but many blanks
- printed page with one potentially important handwritten section
- form with weak scan quality and low information density
- borderline cases where annotators hesitate

---

## Recommended Business Rule

If the product goal is to prevent hallucinations, prioritize:

1. low false-safe rate
2. acceptable review rate
3. only then maximize automation coverage

That means thresholds should be conservative first.

---

## Cursor Handoff Notes

### Priority order for implementation
1. build renderer + metadata pipeline
2. build grouped split logic
3. implement baseline CNN / ViT
4. implement rubric storage and borderline audit tooling
5. implement DiT training
6. implement calibration + thresholding
7. implement inference service

### Engineering constraints
- all preprocessing must be deterministic
- rendering settings must be versioned
- splits must be saved to disk
- thresholds must be versioned alongside the model
- evaluation must report per-institution metrics

### Acceptance criteria
A model version is production-candidate only if:
- grouped test performance is stable
- false-safe rate is below target
- calibration is acceptable
- borderline diagnostic set performance is reviewed manually
- per-institution failure spread is understood

---

## Final Recommendation

Build this as:

**single-page PDF rendering → document-image classifier (DiT) → calibrated thresholds → safe/review/risky output**

Use the risk rubric to clean and stabilize the decision boundary.
Use grouped institutional splits to avoid fake generalization.
Optimize for low false-safe rate, not raw F1 alone.

This is the highest-probability 5/5 path for your dataset, constraints, and production goal.
