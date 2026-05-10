#!/usr/bin/env python3
"""
split_pdfs_to_pages.py — Render all PDFs in data/spaciel_font/ into
per-page PNG images at 300 DPI and write a manifest CSV.

Python environment: doc-risk-classifier
    ~/.pyenv/versions/doc-risk-classifier/bin/python scripts/split_pdfs_to_pages.py

Output:
    data/spaciel_font_pages/<stem>_page_001.png  ...
    data/spaciel_font_pages/manifest.csv
"""

import csv
import sys
from pathlib import Path

import fitz  # PyMuPDF — already in doc-risk-classifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = ROOT / "data" / "spaciel_font"
OUTPUT_DIR = ROOT / "data" / "spaciel_font_pages"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

DPI = 300         # high resolution for OCR quality
SKIP_EXISTING = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stem(path: Path) -> str:
    """
    Builds a filesystem-safe stem from a PDF path, prefixing with the
    immediate parent directory name when the PDF lives inside a subdirectory
    of INPUT_DIR (e.g. יוכבד_מזרחי/74-80.pdf → יוכבד_מזרחי_74-80).
    """
    rel = path.relative_to(INPUT_DIR)
    parts = list(rel.parts)       # e.g. ['יוכבד_מזרחי', '74-80.pdf']
    if len(parts) == 1:
        return path.stem          # top-level PDF — use stem as-is
    # nested: join all parts except the final extension
    parent_parts = parts[:-1]
    stem = path.stem
    return "_".join(parent_parts + [stem])


def render_pdf_to_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Renders every page of *pdf_path* to an individual PNG file.

    Args:
        pdf_path:     Path to the source PDF.
        output_dir:   Directory where PNGs are written.
        dpi:          Rendering resolution.
        skip_existing: Skip pages whose output PNG already exists.

    Returns:
        List of dicts: {page_image, source_pdf, page_num, status}.
        status is one of 'rendered', 'skipped', 'error'.
    """
    stem = _safe_stem(pdf_path)
    records: list[dict] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  [ERROR] Cannot open {pdf_path}: {exc}", file=sys.stderr)
        return records

    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            out_name = f"{stem}_page_{page_num:03d}.png"
            out_path = output_dir / out_name

            if skip_existing and out_path.exists():
                records.append(
                    {
                        "page_image": str(out_path.relative_to(output_dir)),
                        "source_pdf": str(pdf_path.relative_to(INPUT_DIR)),
                        "page_num": page_num,
                        "status": "skipped",
                    }
                )
                continue

            try:
                page = doc[page_idx]
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                # Render in RGB (colour) — Gemma 4 is a colour model
                pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
                pixmap.save(str(out_path))
                records.append(
                    {
                        "page_image": str(out_path.relative_to(output_dir)),
                        "source_pdf": str(pdf_path.relative_to(INPUT_DIR)),
                        "page_num": page_num,
                        "status": "rendered",
                    }
                )
            except Exception as exc:
                print(
                    f"  [ERROR] Page {page_num} of {pdf_path.name}: {exc}",
                    file=sys.stderr,
                )
                records.append(
                    {
                        "page_image": str(out_path.relative_to(output_dir)),
                        "source_pdf": str(pdf_path.relative_to(INPUT_DIR)),
                        "page_num": page_num,
                        "status": "error",
                    }
                )
    finally:
        doc.close()

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== Splitting PDFs to per-page images ===")
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  DPI    : {DPI}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all PDFs recursively
    pdf_files = sorted(INPUT_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found under {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s):")
    for p in pdf_files:
        print(f"  {p.relative_to(INPUT_DIR)}")
    print()

    all_records: list[dict] = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.relative_to(INPUT_DIR)}")
        records = render_pdf_to_pages(pdf_path, OUTPUT_DIR, dpi=DPI, skip_existing=SKIP_EXISTING)
        n_rendered = sum(1 for r in records if r["status"] == "rendered")
        n_skipped  = sum(1 for r in records if r["status"] == "skipped")
        n_errors   = sum(1 for r in records if r["status"] == "error")
        print(
            f"  {len(records)} pages — "
            f"{n_rendered} rendered, {n_skipped} skipped, {n_errors} errors"
        )
        all_records.extend(records)

    # Write manifest (all pages, regardless of status)
    manifest_fields = ["page_image", "source_pdf", "page_num", "status"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_fields)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)

    # Summary
    total   = len(all_records)
    rendered = sum(1 for r in all_records if r["status"] == "rendered")
    skipped  = sum(1 for r in all_records if r["status"] == "skipped")
    errors   = sum(1 for r in all_records if r["status"] == "error")

    print()
    print(f"=== Done ===")
    print(f"  Total pages : {total}")
    print(f"  Rendered    : {rendered}")
    print(f"  Skipped     : {skipped}")
    print(f"  Errors      : {errors}")
    print(f"  Manifest    : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
