---
adr: 0001
title: Add parallel simple two-folder ingestion path (safe/ + risky/)
status: accepted
date: 2026-05-11
deciders: [user, agent]
tags: [data-pipeline, ingestion]
supersedes: []
---

# ADR-0001: Add parallel simple two-folder ingestion path (safe/ + risky/)

## Context

The existing pipeline ingests five labelled source folders
(`regular_forms`, `regular_forms_edge_cases`, `handwritten`,
`handwritten_and_questioniers`, `handwritten_edge_cases`) into a single
`data/metadata.csv` with rich provenance columns
(`source_folder`, `template_family`, `source_doc_stem`,
`is_edge_case`, `institution`). Phase 5–6 trained checkpoints, the
external validation report, and `reports/finetune_report.html` all
depend on that schema and its `source_doc_stem`-grouped splits.

The user requested that the pipeline "start from only two folders:
safe, risky (both under data)". Replacing the existing ingestion would
invalidate every committed artefact; ignoring the request would block
the new use-case.

## Decision

Add a **parallel** "simple" ingestion path that consumes
`data/safe/` and `data/risky/` (raw PDFs, possibly nested) and produces
its own artefacts under disjoint paths:

- `data/metadata_simple.csv` (minimal schema + rubric placeholder columns)
- `data/rendered_pages_simple/` (224×224 grayscale PNGs, flat, unique names)
- `data/splits_simple/{train,val,test}.csv` (random stratified 70/15/15)

The existing 5-folder pipeline (`build_metadata_v2.py`,
`render_all_pages.py`, `regenerate_splits.py`, `data/metadata.csv`,
`data/rendered_pages/`, `data/splits/`) is **untouched**. Training
scripts in `src/train/` consume whichever pipeline is referenced by
their config — `configs/baseline_simple.yaml` and
`configs/dit_simple.yaml` point at the simple artefacts.

## Rejected alternatives

| Option | Why rejected |
|--------|--------------|
| Replace the 5-folder pipeline outright | Invalidates Phase 5/6/6c/6d checkpoints, `validation_report/`, and `reports/finetune_report.html`; loses `source_doc_stem` document-level grouping that prevents page leakage; loses ground-truth provenance needed for per-source FSR breakdowns. |
| Migrate existing data into `data/safe/` + `data/risky/` | Same artefact-invalidation problem as outright replacement; loses the ability to reproduce historical reports from `data/metadata_v1.csv` and `data/metadata_v2_precorrection.csv`. |
| Modify `src/data/dataset.py` to read from per-class folders directly without metadata.csv | Hidden coupling between dataset and on-disk layout; harder to inspect/split/audit; breaks the existing schema contract documented in `docs/INTERFACES.md`. |
| Reuse `data/metadata.csv` with a `pipeline_source` flag | Mixes incompatible split strategies (source-doc-grouped vs random stratified) in one file; risk of accidentally training the new pipeline with rows from the old schema. |

## Consequences

- **Positive:** existing checkpoints and reports remain valid; new pipeline is fully isolated and inspectable; future agents can run either pipeline without ambiguity by choosing a config; ADR makes the choice explicit so nobody re-derives "let's just replace it".
- **Negative:** two metadata files now exist (`metadata.csv` and `metadata_simple.csv`); future schema changes must be applied to both if they should affect both pipelines; the unique-PNG-filename strategy (path slug + `__page_NNN`) is a contract the renderer and dataset must agree on — documented in the script headers.

## Evidence

- User responses captured via the interactive question form in the chat
  session that produced this ADR: scope = "add new ingestion path",
  contents = "PDFs, possibly nested", metadata richness = "minimal + keep
  rubric", split strategy = "random stratified per page".
- Existing `src/data/dataset.py` (lines 106–127) flattens rendered PNG
  paths to basename only — confirming that the simple pipeline must
  generate globally-unique flat filenames.
- `docs/PROJECT_STATUS.md` § Cross-Step Knowledge documents that the
  current 5-folder schema is depended on by every committed artefact.

---

> **Immutability:** ADRs are append-only. To override, write a new ADR
> with `supersedes: [ADR-0001]` and update this file's frontmatter to
> `status: superseded-by-ADR-NNNN`. Never edit the body.
