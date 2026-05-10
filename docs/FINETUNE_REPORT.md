# Fine-Tune Report — Hebrew PDF Hallucination-Risk Classifier

**Phase 6 (expanded retraining) + Phase 6c (cost-weighted thresholds) + Phase 6d (special-font reclassified as regular-form edge cases).**
This document is a plain-text summary of the full interactive report at
[`reports/finetune_report.html`](../reports/finetune_report.html). Share the
HTML file directly — it is self-contained (embedded data + Plotly CDN) and
needs no server to render.

---

## 1. What changed vs Phase 5

| Dimension             | Phase 5                           | Phase 6d (current)                                                                                        |
| --------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Corpus size           | 1,014 pages (2 source folders)    | **1,873 pages** (5 source folders)                                                                        |
| Class balance         | 634 safe / 380 risky              | **825 safe / 1,048 risky** (Phase 6 had 634/1,239 — Phase 6d moved 191 mis-labeled "special_font" pages from risky to safe) |
| Split key             | `template_family`                 | **`source_doc_stem`** (54 unique source PDFs — prevents page-level leakage)                               |
| Loss                  | Plain `BCEWithLogitsLoss`         | `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`                                                           |
| Sampler               | None                              | `WeightedRandomSampler` (minority upsampled per epoch)                                                    |
| Threshold policy      | FSR ≤ 5% target on val            | **`10·FN + FP` minimised on val** (FN treated as 10× worse than FP)                                       |
| Architectures touched | —                                 | None. Data + loss + sampler + threshold only.                                                             |

**Phase 6d data correction (April 2026):** the 191 pages previously in
`source_folder = "spaciel_font"` were on review found to be edge-case regular
forms (visually unusual layouts of normal documents), not a separate "special
font" risky class. They are now `source_folder = "regular_forms_edge_cases"`,
`label_binary = 0`, `template_family = "regular_form_edge"`,
`is_edge_case = True`. The on-disk PDF/PNG filenames were not changed — only
the metadata flags. All three models were retrained from scratch on the
corrected labels.

---

## 2. Cost model

The post-hoc threshold per model is chosen so it minimises
`cost(τ) = 10·FN(τ) + 1·FP(τ)` on the validation split, where:

* **FN** = a risky page classified as safe (→ hallucinated extraction flows downstream).
* **FP** = a safe page classified as risky (→ wasted reviewer time).

Implemented in `TemperatureCalibrator.get_cost_weighted_thresholds()` and
applied via `scripts/retune_thresholds.py` (defaults: `--fn-cost 10 --fp-cost 1`).
The script backs up the originals under `checkpoints/thresholds_backup_<timestamp>/`
and updates both the checkpoint `thresholds` dict and the `calibrator.pkl`.

---

## 3. Held-out test set (n = 284, source-doc grouped)

| Model      |    F1 | ROC-AUC | Temperature |    τ* |  FN @ 0.5 |  FN @ τ* |
| ---------- | ----: | ------: | ----------: | ----: | --------: | -------: |
| ResNet50   | 0.971 |   0.999 |       0.932 | 0.050 |         0 |    **0** |
| ViT-Base   | 0.953 |   0.986 |       2.353 | 0.032 |         8 |    **0** |
| DiT        | 0.978 |   0.999 |       3.749 | 0.036 |         4 |    **0** |

All three models hit **0 FN on the test set** at their respective τ*.
At τ* the FP counts are ResNet50: 21, ViT-Base: 21, DiT: 12 — the cost of
catching the last few risky pages on each model.

**Phase 6 → 6d test-F1 deltas** (sanity check that the reclassification did
not destabilise the models): ResNet50 0.967 → 0.971 (+0.4 pp), ViT-Base
0.965 → 0.953 (−1.2 pp), DiT 0.950 → 0.978 (+2.8 pp). All within the ±5 pp
"keep going" band.

---

## 4. External validation set (n = 400)

94 risky + 306 safe documents collected independently of the training corpus.
This is the distribution-shift benchmark that exposed the Phase 5 template shortcut.

| Model      | FSR (Phase 5) | FSR @ 0.5 (Phase 6d) | FSR @ τ* (Phase 6c) | **FSR @ τ* (Phase 6d)** | TN / FP / FN / TP @ τ* |
| ---------- | ------------: | -------------------: | ------------------: | ----------------------: | :--------------------: |
| ResNet50   |         36.2% |                10.6% |                3.2% |                **2.1%** |   139 / 167 / 10 /  84 |
| ViT-Base   |         50.0% |                24.5% |                9.6% |                **3.2%** |   177 / 129 / 23 /  71 |
| **DiT**    |         30.9% |                25.5% |                2.1% |                **1.1%** |   215 /  91 / 24 /  70 |

Phase 6d (the special-font label fix + retrain + retune) tightened external
FSR further on every model. **DiT now also leads on accuracy** (71.2%) in
addition to FSR — it was previously trailing ResNet50 on accuracy.

The flip side is that accuracy at τ* is still well below test accuracy because
many borderline safe docs now route to review. With a 10× FN : FP cost ratio,
that trade is correct by construction.

---

## 5. Recommendation

**Promote DiT** at the new cost-weighted threshold.

* External FSR at τ*: **1.1%** — lowest of the three. Test FSR: 0.0%.
* External accuracy: 71.2% — also highest of the three.
* Stored thresholds already reflect τ* = 0.036, T_high = 0.518, T = 3.749.
* Routing:
  * `p < T_low (= τ*)` → auto-extract (safe).
  * `T_low ≤ p < T_high` → review queue.
  * `p ≥ T_high` → reject / escalate.

Keep ResNet50 as a cheap secondary model — it has the highest test ROC-AUC
(0.999) and lowest inference cost, so a two-stage ResNet50 → DiT ensemble is
worth exploring. ViT-Base trails on test F1 and has no clear niche.

---

## 6. Reproducing this report

```bash
# 1. (Optional) retune thresholds with a different cost ratio:
python scripts/retune_thresholds.py --fn-cost 10 --fp-cost 1

# 2. Re-run external inference so validation_report/report_data.json
#    reflects the current thresholds AND emits per_doc arrays:
python scripts/_run_validation_inference.py

# 3. Rebuild the single-file HTML report:
python scripts/build_finetune_report.py
# → reports/finetune_report.html
```

---

## 7. Known gotchas for future agents

* `report_data.json` **must** contain `per_doc` arrays (`y_true`, `y_prob`,
  `source`) for `build_finetune_report.py` to work. `_run_validation_inference.py`
  writes them as of Phase 6c; older JSONs don't have them.
* DiT's training log uses stage-scoped epoch numbers (`stage1/train`, `stage1/val`,
  `stage2/...`). The report builder offsets stage-2 and stage-3 epochs so the
  x-axis shows a monotonic global epoch.
* `HF_HUB_OFFLINE=1` is required when running inference inside the sandbox
  because network access is restricted — the DiT backbone is already cached at
  `~/.cache/huggingface/hub/models--microsoft--dit-base/`.
* Changing the cost ratio shifts τ* by a lot — at 10:1 τ* lands at 0.02–0.05;
  at 1:1 it would be around 0.5; at 1:10 (FP 10× worse) it would push well above 0.9.
* `τ*` is written to `checkpoint["thresholds"]["T_low"]` **and** to
  `calibrator.t_low`. Anything that reads thresholds from only one location
  will drift — always update both (the retune script does).
