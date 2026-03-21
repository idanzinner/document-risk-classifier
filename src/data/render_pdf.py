"""
render_pdf.py — PDF-to-image rendering utilities.

Renders single-page PDFs to PIL Images at a configurable DPI,
with optional grayscale conversion and aspect-ratio-preserving resize
padded to a square target size.  All rendering is deterministic.
"""

import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def render_page(
    pdf_path: str,
    dpi: int = 150,
    grayscale: bool = True,
    target_size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Renders a single-page PDF to a PIL Image.

    - Grayscale conversion if grayscale=True
    - Preserves aspect ratio, pads to target_size with white
    - Deterministic (same input always gives same output)

    Args:
        pdf_path: Path to the PDF file (single-page expected).
        dpi: Rendering resolution in dots per inch.
        grayscale: If True, returns mode 'L'; otherwise 'RGB'.
        target_size: (width, height) of the output image in pixels.

    Returns:
        PIL.Image in mode 'L' (grayscale) or 'RGB'.
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        colorspace = fitz.csGRAY if grayscale else fitz.csRGB
        pixmap = page.get_pixmap(matrix=matrix, colorspace=colorspace, alpha=False)
    finally:
        doc.close()

    # Convert pixmap to PIL Image
    if grayscale:
        img = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    else:
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    # Aspect-ratio-preserving resize + centered white padding
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create white canvas and paste resized image centered
    if grayscale:
        canvas = Image.new("L", (target_w, target_h), color=255)
    else:
        canvas = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))

    return canvas


def render_all(
    pdf_dir: str,
    output_dir: str,
    dpi: int = 150,
    grayscale: bool = True,
    target_size: tuple[int, int] = (224, 224),
    skip_existing: bool = True,
) -> list[dict]:
    """
    Renders all PDFs in pdf_dir (recursive search), saves PNG files to output_dir.

    The output directory structure mirrors the subdirectory layout of pdf_dir.
    Each PDF is rendered to a PNG with the same stem and '.png' extension.

    Args:
        pdf_dir: Directory containing PDF files (searched recursively).
        output_dir: Directory where rendered PNGs are saved.
        dpi: Rendering resolution.
        grayscale: If True, renders in grayscale.
        target_size: Output image dimensions (width, height).
        skip_existing: If True, skips files already present in output_dir.

    Returns:
        List of dicts with keys: pdf_path, rendered_path, status
        status is one of: 'rendered', 'skipped', 'error'
    """
    pdf_root = Path(pdf_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_root.rglob("*.pdf"))
    results: list[dict] = []

    for pdf_path in pdf_files:
        # Mirror subdirectory structure in output
        rel_path = pdf_path.relative_to(pdf_root)
        rendered_path = out_root / rel_path.with_suffix(".png")

        if skip_existing and rendered_path.exists():
            results.append(
                {
                    "pdf_path": str(pdf_path),
                    "rendered_path": str(rendered_path),
                    "status": "skipped",
                }
            )
            continue

        try:
            rendered_path.parent.mkdir(parents=True, exist_ok=True)
            img = render_page(str(pdf_path), dpi=dpi, grayscale=grayscale, target_size=target_size)
            img.save(str(rendered_path), format="PNG")
            results.append(
                {
                    "pdf_path": str(pdf_path),
                    "rendered_path": str(rendered_path),
                    "status": "rendered",
                }
            )
            logger.debug("Rendered: %s -> %s", pdf_path, rendered_path)
        except Exception as exc:
            logger.warning("Failed to render %s: %s", pdf_path, exc)
            results.append(
                {
                    "pdf_path": str(pdf_path),
                    "rendered_path": str(rendered_path),
                    "status": "error",
                }
            )

    n_rendered = sum(1 for r in results if r["status"] == "rendered")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_errors = sum(1 for r in results if r["status"] == "error")
    logger.info(
        "render_all complete: %d rendered, %d skipped, %d errors (total %d PDFs)",
        n_rendered, n_skipped, n_errors, len(results),
    )

    return results
