"""
render_all_pages.py — Render all unrendered pages from data/metadata.csv
to data/rendered_pages/ at 224×224 grayscale.

Rendering strategy by source_folder:
  regular_forms, handwritten, handwritten_edge_cases,
  handwritten_and_questioniers:
      Source PDF is at data/{source_folder}/{file_path}.
      Rendered via PyMuPDF at 150 DPI, aspect-ratio-preserved, padded to 224×224 grayscale.

  regular_forms_edge_cases (Phase 6d) and the legacy spaciel_font alias:
      Source high-res PNGs already exist in data/spaciel_font_pages/.
      They are resized+converted to 224×224 grayscale (no re-rendering from PDF needed).
      The on-disk source directory is retained as data/spaciel_font_pages/
      to preserve the Phase 5d OCR pipeline's manifest/resume state.

Already-rendered PNGs are skipped (skip-existing = True by default).

Usage:
  python scripts/render_all_pages.py [--force]

  --force : Re-render even if a PNG already exists in rendered_pages/
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
RENDERED_DIR = DATA_DIR / "rendered_pages"
SPACIEL_FONT_PAGES_DIR = DATA_DIR / "spaciel_font_pages"

DPI = 150
TARGET_SIZE = (224, 224)

SOURCE_FOLDER_DIRS = {
    "regular_forms": DATA_DIR / "regular_forms",
    "handwritten": DATA_DIR / "handwritten",
    "handwritten_edge_cases": DATA_DIR / "handwritten_edge_cases",
    "handwritten_and_questioniers": DATA_DIR / "handwritten_and_questioniers",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_pdf_page(pdf_path: Path, dpi: int, target_size: tuple[int, int]) -> Image.Image:
    """Render page 0 of a single-page PDF to a 224×224 grayscale PIL Image."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    finally:
        doc.close()

    img = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    return _resize_pad(img, target_size, fill=255)


def _convert_png_to_classifier(source_png: Path, target_size: tuple[int, int]) -> Image.Image:
    """Convert an existing high-res PNG to 224×224 grayscale for the classifier."""
    img = Image.open(str(source_png)).convert("L")
    return _resize_pad(img, target_size, fill=255)


def _resize_pad(img: Image.Image, target_size: tuple[int, int], fill: int = 255) -> Image.Image:
    """Aspect-ratio-preserving resize then center-pad to target_size."""
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    canvas = Image.new("L", (target_w, target_h), color=fill)
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(force: bool = False) -> None:
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run build_metadata_v2.py first.", file=sys.stderr)
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta)} rows")

    # Filter to rows that need rendering (or all rows if --force)
    total = 0
    rendered = 0
    skipped = 0
    errors = 0

    for _, row in meta.iterrows():
        file_path = str(row["file_path"])
        source_folder = str(row.get("source_folder", ""))
        png_name = Path(file_path).with_suffix(".png").name
        out_path = RENDERED_DIR / png_name

        if out_path.exists() and not force:
            skipped += 1
            total += 1
            continue

        # Determine source and render strategy
        if source_folder in SOURCE_FOLDER_DIRS:
            pdf_path = SOURCE_FOLDER_DIRS[source_folder] / file_path
            if not pdf_path.exists():
                print(f"  [MISS] PDF not found: {pdf_path}", file=sys.stderr)
                errors += 1
                total += 1
                continue
            try:
                img = _render_pdf_page(pdf_path, DPI, TARGET_SIZE)
                img.save(str(out_path), format="PNG")
                rendered += 1
            except Exception as exc:
                print(f"  [ERR] {pdf_path.name}: {exc}", file=sys.stderr)
                errors += 1

        elif source_folder in ("regular_forms_edge_cases", "spaciel_font"):
            # Source is the high-res PNG in spaciel_font_pages/. The directory
            # name is preserved from Phase 5d; the metadata source_folder is
            # the new Phase 6d label "regular_forms_edge_cases" (we still
            # accept the legacy "spaciel_font" tag for backwards compatibility
            # with old metadata snapshots).
            stem = Path(file_path).stem  # e.g. "da0838fa_page_001"
            source_png = SPACIEL_FONT_PAGES_DIR / f"{stem}.png"
            if not source_png.exists():
                print(f"  [MISS] regular_forms_edge_cases PNG not found: {source_png}", file=sys.stderr)
                errors += 1
                total += 1
                continue
            try:
                img = _convert_png_to_classifier(source_png, TARGET_SIZE)
                img.save(str(out_path), format="PNG")
                rendered += 1
            except Exception as exc:
                print(f"  [ERR] {source_png.name}: {exc}", file=sys.stderr)
                errors += 1

        else:
            print(f"  [WARN] Unknown source_folder '{source_folder}' for {file_path}", file=sys.stderr)
            errors += 1

        total += 1

        if total % 100 == 0:
            print(f"  Progress: {total}/{len(meta)} | rendered={rendered} skipped={skipped} errors={errors}")

    print(f"\n{'='*60}")
    print(f"Render complete: {total} total")
    print(f"  Rendered : {rendered}")
    print(f"  Skipped  : {skipped}  (already existed)")
    print(f"  Errors   : {errors}")
    print(f"  Output   : {RENDERED_DIR}")

    # Spot-check: confirm a few new renders exist
    png_count = sum(1 for _ in RENDERED_DIR.glob("*.png"))
    print(f"\nTotal PNGs in rendered_pages/: {png_count}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render all metadata pages to 224×224 grayscale PNGs")
    p.add_argument("--force", action="store_true", help="Re-render even if PNG already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
