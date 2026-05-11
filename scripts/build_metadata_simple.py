"""
build_metadata_simple.py — Build data/metadata_simple.csv from the simple
two-folder layout (data/safe/ + data/risky/).

Pipeline entry point for the "simple" ingestion path (ADR-0001).

Walks both folders recursively for *.pdf files, opens each PDF with
PyMuPDF to count its pages, and emits ONE row per page into
data/metadata_simple.csv.

Filename-uniqueness contract
----------------------------
`HallucinationRiskDataset` flattens rendered PNGs to basename. To avoid
collisions when nested subdirectories contain identically-named PDFs,
each row's `file_path` is a slug built from the PDF's relative path:

    data/safe/sub/foo.pdf, page 1
        → slug "safe__sub__foo__page_001"
        → file_path "safe__sub__foo__page_001.pdf"
        → PNG "safe__sub__foo__page_001.png"

The renderer (`render_pages_simple.py`) writes exactly this filename to
`data/rendered_pages_simple/`. The metadata's `source_pdf` column keeps
the original relative path for traceability.

Metadata schema (minimal + rubric placeholders, per ADR-0001):
  file_path        slug-based virtual path with .pdf extension
  page_num         1-indexed page number within the source PDF
  label_binary     0 (safe) or 1 (risky)
  source_pdf       original relative path from data/, e.g. "safe/sub/foo.pdf"
  num_pages        total page count of the source PDF (denormalised)
  split            empty until make_splits_simple.py runs
  D, H, S, L       rubric scores (init -1 = unannotated)
  risk_score       int (init -1)

Usage:
  python scripts/build_metadata_simple.py
  python scripts/build_metadata_simple.py --safe-dir data/safe --risky-dir data/risky
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger("build_metadata_simple")

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

DEFAULT_SAFE_DIR = DATA_DIR / "safe"
DEFAULT_RISKY_DIR = DATA_DIR / "risky"
DEFAULT_OUTPUT = DATA_DIR / "metadata_simple.csv"

METADATA_COLS = [
    "file_path",
    "page_num",
    "label_binary",
    "source_pdf",
    "num_pages",
    "split",
    "D",
    "H",
    "S",
    "L",
    "risk_score",
]

# Safety check: refuse to consume the legacy 5-folder pipeline by accident.
_DISALLOWED_SOURCE_NAMES = {
    "regular_forms",
    "regular_forms_edge_cases",
    "handwritten",
    "handwritten_edge_cases",
    "handwritten_and_questioniers",
    "spaciel_font",
}


def make_slug(rel_path: Path) -> str:
    """Build a globally-unique slug from a PDF's path relative to data/.

    Joins path components with "__" and strips the .pdf suffix. Trailing
    "__page_NNN" is appended by the caller.
    """
    parts = list(rel_path.with_suffix("").parts)
    if not parts:
        raise ValueError(f"Cannot slug empty relative path: {rel_path}")
    return "__".join(parts)


def count_pages(pdf_path: Path) -> int:
    """Return the page count of a PDF using PyMuPDF. Raises on failure."""
    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def scan_folder(
    folder_dir: Path,
    label: int,
    data_root: Path,
) -> tuple[list[dict], int, int]:
    """Walk one source folder recursively, emit one row per page.

    Returns: (rows, n_pdfs_seen, n_pdfs_failed)
    """
    rows: list[dict] = []
    failed = 0

    if not folder_dir.is_dir():
        logger.warning("Source folder not found: %s — skipping", folder_dir)
        return rows, 0, 0

    pdfs = sorted(folder_dir.rglob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found under %s", folder_dir)
        return rows, 0, 0

    seen_slugs: set[str] = set()

    for pdf_path in pdfs:
        rel_to_data = pdf_path.relative_to(data_root)
        try:
            n_pages = count_pages(pdf_path)
        except Exception as exc:
            logger.error("Failed to open %s: %s", pdf_path, exc)
            failed += 1
            continue

        if n_pages == 0:
            logger.warning("PDF has 0 pages, skipping: %s", pdf_path)
            failed += 1
            continue

        slug_base = make_slug(rel_to_data)
        if slug_base in seen_slugs:
            logger.error(
                "Duplicate slug '%s' from %s — slug collisions are a bug, "
                "rename or restructure the source PDFs.",
                slug_base, pdf_path,
            )
            failed += 1
            continue
        seen_slugs.add(slug_base)

        for page_idx in range(n_pages):
            page_num = page_idx + 1
            slug = f"{slug_base}__page_{page_num:03d}"
            rows.append({
                "file_path": f"{slug}.pdf",
                "page_num": page_num,
                "label_binary": label,
                "source_pdf": str(rel_to_data).replace("\\", "/"),
                "num_pages": n_pages,
                "split": "",
                "D": -1,
                "H": -1,
                "S": -1,
                "L": -1,
                "risk_score": -1,
            })

    return rows, len(pdfs), failed


def main(
    safe_dir: Path = DEFAULT_SAFE_DIR,
    risky_dir: Path = DEFAULT_RISKY_DIR,
    output_csv: Path = DEFAULT_OUTPUT,
    data_root: Path = DATA_DIR,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Sanity: refuse if someone passes a legacy folder by mistake
    for d in (safe_dir, risky_dir):
        if d.name in _DISALLOWED_SOURCE_NAMES:
            logger.error(
                "Refusing to ingest legacy folder '%s' via the simple pipeline. "
                "Use scripts/build_metadata_v2.py for the 5-folder pipeline.",
                d.name,
            )
            sys.exit(2)

    if not safe_dir.exists() and not risky_dir.exists():
        logger.error(
            "Neither %s nor %s exists. Create the folders and add PDFs first.",
            safe_dir, risky_dir,
        )
        sys.exit(1)

    logger.info("Ingesting from:")
    logger.info("  safe : %s", safe_dir)
    logger.info("  risky: %s", risky_dir)
    logger.info("Data root for slugs: %s", data_root)

    safe_rows, n_safe_pdfs, n_safe_fail = scan_folder(safe_dir, 0, data_root)
    risky_rows, n_risky_pdfs, n_risky_fail = scan_folder(risky_dir, 1, data_root)

    rows = safe_rows + risky_rows
    if not rows:
        logger.error("No pages enumerated. Check folder contents and try again.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=METADATA_COLS)

    # Final uniqueness check (defensive — slug collisions would break dataset.py)
    dupes = df["file_path"].duplicated(keep=False)
    if dupes.any():
        offenders = df.loc[dupes, ["file_path", "source_pdf"]].head(10)
        logger.error("Duplicate file_path values detected:\n%s", offenders.to_string())
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    safe_pages = int((df["label_binary"] == 0).sum())
    risky_pages = int((df["label_binary"] == 1).sum())
    ratio = (safe_pages / risky_pages) if risky_pages else float("inf")

    print("=" * 60)
    print(f"Saved → {output_csv}  ({len(df)} rows)")
    print()
    print(f"Source PDFs : safe={n_safe_pdfs}  risky={n_risky_pdfs}")
    print(f"Failed PDFs : safe={n_safe_fail}  risky={n_risky_fail}")
    print(f"Pages       : safe={safe_pages}  risky={risky_pages}")
    print(f"Class ratio : {ratio:.2f} safe : 1 risky")
    print()
    print("Next steps:")
    print("  1. python scripts/render_pages_simple.py")
    print("  2. python scripts/make_splits_simple.py")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build metadata_simple.csv from data/safe + data/risky",
    )
    p.add_argument("--safe-dir", type=Path, default=DEFAULT_SAFE_DIR)
    p.add_argument("--risky-dir", type=Path, default=DEFAULT_RISKY_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--data-root", type=Path, default=DATA_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        safe_dir=args.safe_dir,
        risky_dir=args.risky_dir,
        output_csv=args.output,
        data_root=args.data_root,
    )
