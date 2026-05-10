"""
fix_spaciel_font_to_edge_cases.py — One-shot data correction (Phase 6d).

Background:
    The 191 pages currently tagged ``source_folder == "spaciel_font"`` in
    ``data/metadata.csv`` were originally treated as a separate "special font"
    risky class (label_binary=1). On review they are actually edge-case
    regular forms (label_binary=0) — visually unusual layouts of normal
    documents, not handwritten/questionnaire content.

What this script does (all in-memory, then writes once):

    For every row where ``source_folder == "spaciel_font"`` flip:

        source_folder    : "spaciel_font"   -> "regular_forms_edge_cases"
        template_family  : "special_font"   -> "regular_form_edge"
        is_edge_case     : False            -> True
        label_binary     : 1                -> 0
        institution      : "questionnaires" -> "regular_docs"

    The 5 rubric columns (D / H / S / L / risk_score) are left untouched —
    Claude's vision-based rubric describes document features and is independent
    of the binary label. Sanity check on the existing rubric: mean risk_score
    for these 191 pages is 3.2 (max 5), which is already in the "safe" band.

What this script does NOT do:

    - Move any files on disk. ``data/spaciel_font/`` PDFs and the rendered
      PNGs in ``data/rendered_pages/`` keep their existing filenames so
      every checkpoint's stored ``file_path`` references stay valid.
    - Touch the Phase 5d OCR pipeline (``data/spaciel_font_pages/`` +
      ``notebooks/09_special_font_ocr.ipynb``). That is an independent OCR
      track and renaming would break its resume state.
    - Re-annotate rubric scores.

After flipping the columns, the script re-runs ``regenerate_splits.main``
in-process so ``data/splits/{train,val,test}.csv`` pick up the new labels,
and finally re-writes ``data/metadata.csv`` with ``encoding="utf-8"`` (no
BOM) because ``annotate_rubric.py`` opens it with strict UTF-8 and would
KeyError on ``\ufefffile_path`` otherwise.

Backups created upstream of this script (see plan):
    data/metadata_v2_precorrection.csv
    data/splits_v2_precorrection/{train,val,test}.csv
    checkpoints/_precorrection_backup_<ts>/

Usage:
    python scripts/fix_spaciel_font_to_edge_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
META_PATH = DATA_DIR / "metadata.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import regenerate_splits

OLD_SOURCE = "spaciel_font"
NEW_SOURCE = "regular_forms_edge_cases"
OLD_TEMPLATE = "special_font"
NEW_TEMPLATE = "regular_form_edge"
OLD_INSTITUTION = "questionnaires"
NEW_INSTITUTION = "regular_docs"


def _flip_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["source_folder"] == OLD_SOURCE
    n = int(mask.sum())
    if n == 0:
        print(f"No rows with source_folder == {OLD_SOURCE!r} — nothing to do.")
        return df, 0

    out = df.copy()
    out.loc[mask, "source_folder"] = NEW_SOURCE
    out.loc[mask, "template_family"] = NEW_TEMPLATE
    out.loc[mask, "is_edge_case"] = True
    out.loc[mask, "label_binary"] = 0
    out.loc[mask, "institution"] = NEW_INSTITUTION
    return out, n


def _print_label_balance(df: pd.DataFrame, label: str) -> None:
    safe = int((df["label_binary"] == 0).sum())
    risky = int((df["label_binary"] == 1).sum())
    ratio = safe / risky if risky else float("inf")
    print(f"  {label:14s}  safe={safe:5d}  risky={risky:5d}  ratio={ratio:.3f}:1")


def main() -> None:
    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {META_PATH}")
    df = pd.read_csv(META_PATH)
    print(f"  rows: {len(df)}")
    print("  before flip:")
    _print_label_balance(df, "overall")

    new_df, n_flipped = _flip_metadata(df)
    print(f"\nFlipped {n_flipped} rows ({OLD_SOURCE} -> {NEW_SOURCE})")
    print("  after flip:")
    _print_label_balance(new_df, "overall")

    print("\nWriting metadata.csv (utf-8, no BOM)")
    new_df.to_csv(META_PATH, index=False, encoding="utf-8")

    print("\n--- Re-running regenerate_splits.main() ---")
    regenerate_splits.main()

    # regenerate_splits writes utf-8-sig (BOM); re-write without BOM so that
    # annotate_rubric.py (which expects strict utf-8) keeps working.
    print("\nRe-saving metadata.csv without BOM")
    final = pd.read_csv(META_PATH, encoding="utf-8-sig")
    final.to_csv(META_PATH, index=False, encoding="utf-8")

    print("\nDone.")


if __name__ == "__main__":
    main()
