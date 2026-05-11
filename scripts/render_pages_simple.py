"""
render_pages_simple.py — Render every page in metadata_simple.csv to
224×224 grayscale PNG under data/rendered_pages_simple/.

Second stage of the simple two-folder pipeline (ADR-0001). Pre-requisite:
`build_metadata_simple.py` must have produced data/metadata_simple.csv.

Rendering parameters mirror render_all_pages.py (the legacy pipeline) so
both ingestion paths produce visually-identical training inputs:
    DPI         150
    TARGET_SIZE 224 × 224
    Mode        L (grayscale)
    Resize      aspect-ratio preserving + center-pad with white (255)

Skip-existing is on by default, so re-runs after adding new PDFs are cheap.

Usage:
  python scripts/render_pages_simple.py
  python scripts/render_pages_simple.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

logger = logging.getLogger("render_pages_simple")

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

DEFAULT_METADATA = DATA_DIR / "metadata_simple.csv"
DEFAULT_RENDERED_DIR = DATA_DIR / "rendered_pages_simple"

DPI = 150
TARGET_SIZE = (224, 224)


def _render_page(pdf_path: Path, page_idx: int, dpi: int, target: tuple[int, int]) -> Image.Image:
    """Render one PDF page (0-indexed) to a 224×224 grayscale PIL image."""
    doc = fitz.open(str(pdf_path))
    try:
        if page_idx < 0 or page_idx >= doc.page_count:
            raise IndexError(
                f"page_idx {page_idx} out of range for {pdf_path} "
                f"(has {doc.page_count} pages)"
            )
        page = doc[page_idx]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    finally:
        doc.close()

    img = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    return _resize_pad(img, target, fill=255)


def _resize_pad(img: Image.Image, target_size: tuple[int, int], fill: int = 255) -> Image.Image:
    """Aspect-preserving resize then center-pad to target_size with `fill`."""
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    canvas = Image.new("L", (target_w, target_h), color=fill)
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def main(
    metadata_csv: Path = DEFAULT_METADATA,
    rendered_dir: Path = DEFAULT_RENDERED_DIR,
    data_root: Path = DATA_DIR,
    force: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not metadata_csv.exists():
        logger.error(
            "Metadata not found: %s. Run scripts/build_metadata_simple.py first.",
            metadata_csv,
        )
        sys.exit(1)

    rendered_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    logger.info("Loaded %d rows from %s", len(meta), metadata_csv)

    required = {"file_path", "page_num", "source_pdf"}
    missing = required - set(meta.columns)
    if missing:
        logger.error("Metadata missing columns: %s", sorted(missing))
        sys.exit(1)

    rendered = 0
    skipped = 0
    errors = 0
    missing_pdfs: list[str] = []

    for i, row in enumerate(meta.itertuples(index=False), start=1):
        file_path = str(row.file_path)
        source_pdf_rel = str(row.source_pdf)
        page_num = int(row.page_num)

        png_name = Path(file_path).with_suffix(".png").name
        out_path = rendered_dir / png_name

        if out_path.exists() and not force:
            skipped += 1
            continue

        pdf_path = (data_root / source_pdf_rel).resolve()
        if not pdf_path.exists():
            errors += 1
            if len(missing_pdfs) < 10:
                missing_pdfs.append(source_pdf_rel)
            continue

        try:
            img = _render_page(pdf_path, page_num - 1, DPI, TARGET_SIZE)
            img.save(str(out_path), format="PNG")
            rendered += 1
        except Exception as exc:
            logger.error("Render failed for %s page %d: %s", pdf_path, page_num, exc)
            errors += 1

        if i % 100 == 0:
            logger.info(
                "Progress: %d/%d  rendered=%d skipped=%d errors=%d",
                i, len(meta), rendered, skipped, errors,
            )

    print("=" * 60)
    print(f"Rendered : {rendered}")
    print(f"Skipped  : {skipped}  (already existed)")
    print(f"Errors   : {errors}")
    if missing_pdfs:
        print("First missing source PDFs:")
        for p in missing_pdfs:
            print(f"  - {p}")
    print(f"Output   : {rendered_dir}")

    png_count = sum(1 for _ in rendered_dir.glob("*.png"))
    print(f"Total PNGs on disk: {png_count}")
    print()
    print("Next step:")
    print("  python scripts/make_splits_simple.py")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render every page in metadata_simple.csv to 224×224 grayscale PNG",
    )
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--rendered-dir", type=Path, default=DEFAULT_RENDERED_DIR)
    p.add_argument("--data-root", type=Path, default=DATA_DIR)
    p.add_argument("--force", action="store_true", help="Re-render even if PNG already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        metadata_csv=args.metadata,
        rendered_dir=args.rendered_dir,
        data_root=args.data_root,
        force=args.force,
    )
