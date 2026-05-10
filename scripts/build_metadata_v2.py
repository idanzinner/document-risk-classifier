"""
build_metadata_v2.py — Rebuild data/metadata.csv to include all 6 source folders.

New folders being ingested:
  data/handwritten/               — 6 new files (label=1), 380 already in metadata
  data/handwritten_edge_cases/    — 12 new single-page files (label=1)
  data/handwritten_and_questioniers/ — 652 new files (label=1)
  data/spaciel_font/              — 191 pages from 6 multi-page PDFs.
                                    NOTE (Phase 6d): these are now classified
                                    as regular_forms_edge_cases (label=0).
                                    The on-disk directory keeps its legacy
                                    name; the manifest still drives ingest.

Adds new metadata columns (non-breaking):
  source_folder    — name of the source data folder
  source_doc_stem  — original document stem (pages from same source PDF share this)
  is_edge_case     — bool, True for *_edge_cases folders

Also updates data/labels_binary_clean.csv with new rows so that
annotate_rubric.py can pick them up for rubric scoring.

Backs up:
  data/metadata.csv              → data/metadata_v1.csv
  data/labels_binary_clean.csv   → data/labels_binary_clean_v1.csv
  data/splits/                   → data/splits_v1/

Usage:
  python scripts/build_metadata_v2.py
"""

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Source folder definitions
# ---------------------------------------------------------------------------

SOURCE_FOLDERS = {
    "regular_forms": {
        "dir": DATA_DIR / "regular_forms",
        "label": 0,
        "institution": "regular_docs",
        "template_family": "regular_form",
        "is_edge_case": False,
    },
    "handwritten": {
        "dir": DATA_DIR / "handwritten",
        "label": 1,
        "institution": "questionnaires",
        "template_family": None,  # inferred per-file from stem
        "is_edge_case": False,
    },
    "handwritten_edge_cases": {
        "dir": DATA_DIR / "handwritten_edge_cases",
        "label": 1,
        "institution": "questionnaires",
        "template_family": "handwritten_edge",
        "is_edge_case": True,
    },
    "handwritten_and_questioniers": {
        "dir": DATA_DIR / "handwritten_and_questioniers",
        "label": 1,
        "institution": "questionnaires",
        "template_family": "mixed_hwq",
        "is_edge_case": False,
    },
}

# spaciel_font is handled via its manifest (multi-page PDFs already split)
SPACIEL_FONT_PAGES_DIR = DATA_DIR / "spaciel_font_pages"
SPACIEL_FONT_MANIFEST = SPACIEL_FONT_PAGES_DIR / "manifest.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_SUFFIX_RE = re.compile(r"_page_\d+$", re.IGNORECASE)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def get_source_doc_stem(file_stem: str) -> str:
    """Strip _page_NNNN suffix to get the source document stem."""
    return PAGE_SUFFIX_RE.sub("", file_stem)


def infer_template_family_handwritten(stem: str) -> str:
    """Infer template_family for files in data/handwritten/."""
    doc_stem = get_source_doc_stem(stem)
    if UUID_RE.match(doc_stem):
        return "questionnaire_uuid"
    return "questionnaire_named"


def extract_page_num(stem: str) -> int:
    """Extract page number from a _page_NNNN or _page_NNN suffix."""
    m = PAGE_SUFFIX_RE.search(stem)
    if not m:
        return 1
    digits = re.search(r"\d+$", m.group())
    return int(digits.group()) if digits else 1


METADATA_COLS = [
    "file_path",
    "page_num",
    "institution",
    "template_family",
    "label_binary",
    "D",
    "H",
    "S",
    "L",
    "risk_score",
    "split",
    "drive_link",
    "drive_id",
    "source_folder",
    "source_doc_stem",
    "is_edge_case",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Backup existing files ──────────────────────────────────────────────
    meta_path = DATA_DIR / "metadata.csv"
    meta_v1_path = DATA_DIR / "metadata_v1.csv"
    lbc_path = DATA_DIR / "labels_binary_clean.csv"
    lbc_v1_path = DATA_DIR / "labels_binary_clean_v1.csv"
    splits_dir = DATA_DIR / "splits"
    splits_v1_dir = DATA_DIR / "splits_v1"

    if not meta_path.exists():
        print("ERROR: data/metadata.csv not found. Run setup notebook first.", file=sys.stderr)
        sys.exit(1)

    shutil.copy(meta_path, meta_v1_path)
    print(f"Backed up metadata.csv → metadata_v1.csv")

    if lbc_path.exists():
        shutil.copy(lbc_path, lbc_v1_path)
        print(f"Backed up labels_binary_clean.csv → labels_binary_clean_v1.csv")

    if splits_dir.exists():
        if splits_v1_dir.exists():
            shutil.rmtree(splits_v1_dir)
        shutil.copytree(splits_dir, splits_v1_dir)
        print(f"Backed up splits/ → splits_v1/")

    # ── Load old metadata ──────────────────────────────────────────────────
    old_meta = pd.read_csv(meta_v1_path)
    print(f"\nOld metadata: {len(old_meta)} rows")
    print(f"  label_binary: {old_meta['label_binary'].value_counts().to_dict()}")

    # Build stem → row mapping (stem = Path(file_path).stem)
    old_meta["_stem"] = old_meta["file_path"].apply(lambda fp: Path(fp).stem)
    old_stem_set = set(old_meta["_stem"])

    # Assign source_folder / source_doc_stem / is_edge_case to old rows
    # Infer from institution and template_family
    def _source_folder_for_old_row(row):
        if row["institution"] == "regular_docs":
            return "regular_forms"
        tf = row.get("template_family", "")
        if tf == "questionnaire_uuid":
            return "handwritten"
        if tf == "questionnaire_named":
            return "handwritten"
        return "handwritten"

    old_meta["source_folder"] = old_meta.apply(_source_folder_for_old_row, axis=1)
    old_meta["source_doc_stem"] = old_meta["_stem"].apply(get_source_doc_stem)
    old_meta["is_edge_case"] = False
    # Reset split to None — will be regenerated
    old_meta["split"] = None

    # ── Enumerate new files from each PDF source folder ────────────────────
    all_rows: list[dict] = []

    # Carry over old rows first
    for _, row in old_meta.iterrows():
        r = {col: row.get(col) for col in METADATA_COLS}
        all_rows.append(r)

    new_count = 0

    for folder_name, props in SOURCE_FOLDERS.items():
        folder_dir = props["dir"]
        if not folder_dir.exists():
            print(f"WARNING: {folder_dir} does not exist — skipping")
            continue

        pdfs = sorted(folder_dir.glob("*.pdf"))
        already_known = sum(1 for p in pdfs if p.stem in old_stem_set)
        truly_new = [p for p in pdfs if p.stem not in old_stem_set]
        print(f"\n{folder_name}: {len(pdfs)} PDFs total, {already_known} already in metadata, {len(truly_new)} new")

        for pdf_path in truly_new:
            stem = pdf_path.stem
            source_doc_stem = get_source_doc_stem(stem)

            tf = props["template_family"]
            if tf is None:
                tf = infer_template_family_handwritten(stem)

            page_num = extract_page_num(stem)

            row = {
                "file_path": pdf_path.name,
                "page_num": page_num,
                "institution": props["institution"],
                "template_family": tf,
                "label_binary": props["label"],
                "D": -1,
                "H": -1,
                "S": -1,
                "L": -1,
                "risk_score": -1,
                "split": None,
                "drive_link": "",
                "drive_id": "",
                "source_folder": folder_name,
                "source_doc_stem": source_doc_stem,
                "is_edge_case": props["is_edge_case"],
            }
            all_rows.append(row)
            new_count += 1

    # ── Process regular_forms_edge_cases pages from spaciel_font manifest ──
    # Phase 6d reclassification: the 191 pages whose source PDFs live under
    # data/spaciel_font/ (and whose rendered PNGs are listed in the
    # spaciel_font_pages/manifest.csv) are edge-case regular forms (safe),
    # not a separate "special font" risky class. The on-disk directory names
    # are retained for backwards compatibility with existing checkpoints and
    # the Phase 5d OCR pipeline.
    if SPACIEL_FONT_MANIFEST.exists():
        manifest = pd.read_csv(SPACIEL_FONT_MANIFEST)
        # Only include pages that rendered successfully
        manifest = manifest[manifest["status"].isin(["rendered", "skipped"])].copy()
        print(f"\nregular_forms_edge_cases (from spaciel_font manifest): "
              f"{len(manifest)} valid page entries")

        for _, mrow in manifest.iterrows():
            page_image = mrow["page_image"]  # e.g. "da0838fa_page_001.png"
            stem = Path(page_image).stem      # e.g. "da0838fa_page_001"
            source_doc_stem = get_source_doc_stem(stem)
            # Virtual file_path — suffix .pdf so dataset.py converts to .png correctly
            file_path = stem + ".pdf"
            page_num = int(mrow["page_num"])

            row = {
                "file_path": file_path,
                "page_num": page_num,
                "institution": "regular_docs",
                "template_family": "regular_form_edge",
                "label_binary": 0,
                "D": -1,
                "H": -1,
                "S": -1,
                "L": -1,
                "risk_score": -1,
                "split": None,
                "drive_link": "",
                "drive_id": "",
                "source_folder": "regular_forms_edge_cases",
                "source_doc_stem": source_doc_stem,
                "is_edge_case": True,
            }
            all_rows.append(row)
            new_count += 1
    else:
        print(f"WARNING: spaciel_font manifest not found at {SPACIEL_FONT_MANIFEST}")

    # ── Build and validate final DataFrame ────────────────────────────────
    new_meta = pd.DataFrame(all_rows)

    # Remove helper columns if any leaked in
    for col in ["_stem"]:
        if col in new_meta.columns:
            new_meta.drop(columns=[col], inplace=True)

    # Ensure correct column order; fill missing with None
    for col in METADATA_COLS:
        if col not in new_meta.columns:
            new_meta[col] = None
    new_meta = new_meta[METADATA_COLS]

    # Deduplicate on file_path (keep first occurrence = old metadata wins)
    dupes = new_meta.duplicated(subset=["file_path"], keep="first").sum()
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate file_path values found — keeping first occurrence")
        new_meta = new_meta.drop_duplicates(subset=["file_path"], keep="first")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"New metadata summary: {len(new_meta)} total rows ({new_count} new)")
    print(f"\nlabel_binary distribution:")
    print(new_meta["label_binary"].value_counts().to_string())
    print(f"\nsource_folder distribution:")
    print(new_meta["source_folder"].value_counts().to_string())
    print(f"\ntemplate_family distribution:")
    print(new_meta["template_family"].value_counts().to_string())

    safe = (new_meta["label_binary"] == 0).sum()
    risky = (new_meta["label_binary"] == 1).sum()
    print(f"\nClass balance: {safe} safe : {risky} risky  (ratio {safe/risky:.2f}:1)")

    # ── Save new metadata.csv ──────────────────────────────────────────────
    # Use utf-8 (no BOM) so annotate_rubric.py can read it without encoding issues
    new_meta.to_csv(meta_path, index=False, encoding="utf-8")
    print(f"\nSaved → {meta_path}  ({len(new_meta)} rows)")

    # ── Update labels_binary_clean.csv ────────────────────────────────────
    # utf-8-sig for Excel/CSV viewers to display Hebrew correctly
    lbc_new = new_meta[["file_path", "page_num", "label_binary", "drive_link", "drive_id"]].copy()
    lbc_new.to_csv(lbc_path, index=False, encoding="utf-8-sig")
    print(f"Saved → {lbc_path}  ({len(lbc_new)} rows)")

    print(f"\nDone. Next steps:")
    print(f"  1. python scripts/render_all_pages.py      # render new pages")
    print(f"  2. python scripts/annotate_rubric.py --concurrency 10   # score new pages")
    print(f"  3. python scripts/regenerate_splits.py     # regenerate train/val/test splits")


if __name__ == "__main__":
    main()
