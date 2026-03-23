"""
drive_to_gcs.py — Download PDFs from Google Drive and upload them to a GCS bucket.

Reads data/labels_binary_clean.csv (produced by scripts/clean_labels.py), extracts the
Google Drive file ID for each row, downloads each PDF, and uploads it to:

    gs://<BUCKET>/raw_pdfs/<file_path>

After all PDFs are uploaded the script also uploads the metadata CSVs:

    gs://<BUCKET>/data/metadata.csv
    gs://<BUCKET>/data/labels_binary_clean.csv

Download strategy (tried in order):
    1. Google Drive API with Application Default Credentials (ADC) — works automatically
       on Vertex AI Workbench when the Drive folder is shared with the VM service account.
       Find your service account email with:
           curl -s "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" \
                -H "Metadata-Flavor: Google"
       Then share the Drive folder with that email (Viewer access).
    2. gdown anonymous download — fallback for publicly shared files ("Anyone with the link").

Prerequisites:
    pip install google-api-python-client google-auth gdown google-cloud-storage gcsfs tqdm

Authentication:
    On Vertex AI Workbench / Colab Enterprise: share the Drive folder with the VM service
    account email. ADC credentials are picked up automatically.
    Locally: run  gcloud auth application-default login

Usage:
    python scripts/drive_to_gcs.py --bucket my-bucket-name
    python scripts/drive_to_gcs.py --bucket my-bucket-name --input data/labels_binary_clean.csv
    python scripts/drive_to_gcs.py --bucket my-bucket-name --workers 4 --dry-run

Options:
    --bucket          GCS bucket name (required, no gs:// prefix)
    --input           Path to cleaned labels CSV (default: data/labels_binary_clean.csv)
    --data-dir        Local data directory containing the CSV files (default: data/)
    --prefix          GCS prefix / folder for raw PDFs (default: raw_pdfs)
    --workers         Parallel download/upload workers (default: 4)
    --dry-run         Print what would be done without downloading or uploading
    --no-skip-existing  Re-upload files already in GCS (default: skip existing)
    --tmp-dir         Local temp directory for downloads (default: system temp)
    --no-drive-api    Skip Drive API and use only gdown (useful if ADC not configured)
"""

import argparse
import csv
import io
import logging
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level Drive API service (built once, reused across threads)
_drive_service = None
_drive_service_error: str | None = None


def _import_or_fail(pkg: str, install_hint: str) -> object:
    try:
        import importlib
        return importlib.import_module(pkg)
    except ImportError:
        logger.error("Missing package '%s'. Install with: %s", pkg, install_hint)
        sys.exit(1)


def extract_file_id(drive_link: str) -> str:
    """Extract the GDrive file ID from a viewer URL."""
    match = re.search(r"/file/d/([^/]+)/", drive_link)
    return match.group(1) if match else ""


def gcs_blob_exists(bucket, blob_name: str) -> bool:
    blob = bucket.blob(blob_name)
    return blob.exists()


def _build_drive_service():
    """Build a Google Drive API service using Application Default Credentials.

    On Vertex AI Workbench the VM service account credentials are used automatically.
    The Drive folder must be shared with the VM service account email.
    """
    global _drive_service, _drive_service_error
    if _drive_service is not None or _drive_service_error is not None:
        return _drive_service

    try:
        import google.auth
        from googleapiclient.discovery import build

        creds, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.info("Drive API ready (project=%s)", project or "unknown")
    except Exception as exc:
        _drive_service_error = str(exc)
        logger.warning(
            "Drive API unavailable (%s) — will fall back to gdown for all files.", exc
        )
    return _drive_service


def _download_via_drive_api(file_id: str, dest_path: Path) -> bool:
    """Download using the Drive API with ADC credentials. Returns True on success."""
    service = _drive_service
    if service is None:
        return False
    try:
        from googleapiclient.http import MediaIoBaseDownload

        request = service.files().get_media(fileId=file_id)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if not dest_path.exists() or dest_path.stat().st_size == 0:
            dest_path.unlink(missing_ok=True)
            return False
        return True
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.debug("Drive API download failed for %s: %s", file_id, exc)
        return False


def _download_via_gdown(file_id: str, dest_path: Path) -> bool:
    """Download via gdown (works for publicly shared files). Returns True on success."""
    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed; skipping gdown fallback")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(dest_path), quiet=True, fuzzy=False)
        if not dest_path.exists() or dest_path.stat().st_size == 0:
            dest_path.unlink(missing_ok=True)
            return False
        return True
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.debug("gdown download failed for %s: %s", file_id, exc)
        return False


def download_from_drive(
    file_id: str,
    dest_path: Path,
    dry_run: bool = False,
    use_drive_api: bool = True,
) -> bool:
    """Download a single file from Google Drive.

    Tries Drive API first (authenticated, reliable), then falls back to gdown
    (anonymous, works for publicly shared files).
    Returns True on success.
    """
    if dry_run:
        logger.info("[DRY RUN] Would download Drive ID %s -> %s", file_id, dest_path)
        return True

    if use_drive_api and _drive_service is not None:
        if _download_via_drive_api(file_id, dest_path):
            return True
        logger.debug("Drive API failed for %s, trying gdown fallback", file_id)

    return _download_via_gdown(file_id, dest_path)


def upload_to_gcs(local_path: Path, bucket, blob_name: str, dry_run: bool = False) -> bool:
    """Upload a local file to GCS. Returns True on success."""
    if dry_run:
        logger.info("[DRY RUN] Would upload %s -> gs://%s/%s", local_path, bucket.name, blob_name)
        return True
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return True
    except Exception as exc:
        logger.warning("Upload failed for %s: %s", blob_name, exc)
        return False


def process_row(
    row: dict,
    bucket,
    pdf_prefix: str,
    tmp_dir: Path,
    skip_existing: bool,
    dry_run: bool,
    use_drive_api: bool = True,
) -> tuple[str, str]:
    """Download one PDF from Drive and upload to GCS. Returns (filename, status)."""
    filename = row["file_path"]
    file_id = row.get("drive_id") or extract_file_id(row.get("drive_link", ""))

    if not file_id:
        return filename, "error:no_drive_id"

    blob_name = f"{pdf_prefix}/{filename}"

    if skip_existing and not dry_run:
        try:
            if gcs_blob_exists(bucket, blob_name):
                return filename, "skipped"
        except Exception:
            pass

    tmp_path = tmp_dir / filename
    if not download_from_drive(file_id, tmp_path, dry_run=dry_run, use_drive_api=use_drive_api):
        return filename, "error:download_failed"

    ok = upload_to_gcs(tmp_path, bucket, blob_name, dry_run=dry_run)

    # Clean up temp file immediately to save disk space
    if not dry_run and tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    return filename, "uploaded" if ok else "error:upload_failed"


def upload_csv_files(bucket, data_dir: Path, dry_run: bool) -> None:
    """Upload metadata CSVs to gs://<BUCKET>/data/."""
    csv_files = ["metadata.csv", "labels_binary_clean.csv", "labels_binary.csv"]
    for name in csv_files:
        local = data_dir / name
        if local.exists():
            upload_to_gcs(local, bucket, f"data/{name}", dry_run=dry_run)
            logger.info("Uploaded %s -> gs://%s/data/%s", name, bucket.name if not dry_run else "<BUCKET>", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive -> GCS migration for PDF dataset")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs:// prefix)")
    parser.add_argument(
        "--input", default="data/labels_binary_clean.csv",
        help="Cleaned labels CSV with drive_id column (default: data/labels_binary_clean.csv)",
    )
    parser.add_argument("--data-dir", default="data/", help="Local data directory (default: data/)")
    parser.add_argument("--prefix", default="raw_pdfs", help="GCS prefix for PDFs (default: raw_pdfs)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without downloading or uploading",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-upload files already in GCS (default: skip existing)",
    )
    parser.add_argument(
        "--tmp-dir", default=None,
        help="Temporary directory for downloaded PDFs (default: system temp)",
    )
    parser.add_argument(
        "--no-drive-api", action="store_true",
        help="Skip Drive API (ADC) and use only gdown. Use if ADC is not configured.",
    )
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing
    use_drive_api = not args.no_drive_api

    # Initialise Drive API service (uses ADC — automatic on Vertex AI Workbench)
    if use_drive_api and not args.dry_run:
        _build_drive_service()
        if _drive_service is None:
            logger.warning(
                "Drive API not available. Falling back to gdown for all downloads.\n"
                "  To enable the Drive API: share your Drive folder with the VM service account.\n"
                "  Find it with: curl -s http://metadata.google.internal/computeMetadata/v1/"
                "instance/service-accounts/default/email -H 'Metadata-Flavor: Google'\n"
                "  Or run with --no-drive-api to suppress this warning."
            )

    # Read cleaned CSV
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(
            "Input CSV not found: %s\n  Run: python scripts/clean_labels.py first", input_path
        )
        sys.exit(1)

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        logger.error("Input CSV is empty")
        sys.exit(1)

    logger.info("Loaded %d rows from %s", len(rows), input_path)

    if args.dry_run:
        logger.info("[DRY RUN mode] No files will be downloaded or uploaded")
        # Still need a bucket placeholder for dry-run logging
        class _FakeBucket:
            name = args.bucket
        bucket = _FakeBucket()
    else:
        storage = _import_or_fail(
            "google.cloud.storage", "pip install google-cloud-storage>=2.10.0"
        )
        client = storage.Client()
        try:
            bucket = client.bucket(args.bucket)
            # Verify bucket is accessible
            bucket.reload()
            logger.info("Connected to gs://%s", args.bucket)
        except Exception as exc:
            logger.error("Cannot access bucket gs://%s: %s", args.bucket, exc)
            logger.error(
                "Ensure the bucket exists and you have Storage Object Admin permissions.\n"
                "To create it: gsutil mb -l us-central1 gs://%s", args.bucket
            )
            sys.exit(1)

    # Set up temp directory
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        own_tmp = False
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hallucination_pdfs_"))
        own_tmp = True

    logger.info("Using temp dir: %s", tmp_dir)
    logger.info("Uploading %d PDFs to gs://%s/%s/", len(rows), args.bucket, args.prefix)

    results = {"uploaded": 0, "skipped": 0, "error": 0}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_row, row, bucket, args.prefix, tmp_dir,
                    skip_existing, args.dry_run, use_drive_api,
                ): row["file_path"]
                for row in rows
            }
            for i, future in enumerate(as_completed(futures), 1):
                filename, status = future.result()
                if status.startswith("error"):
                    results["error"] += 1
                    logger.warning("[%d/%d] ERROR %s — %s", i, len(rows), filename, status)
                elif status == "skipped":
                    results["skipped"] += 1
                    if i % 100 == 0:
                        logger.info("[%d/%d] ... (skipping existing)", i, len(rows))
                else:
                    results["uploaded"] += 1
                    if i % 50 == 0 or i == len(rows):
                        logger.info("[%d/%d] uploaded %s", i, len(rows), filename)
    finally:
        if own_tmp and not args.dry_run:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "PDF migration complete: uploaded=%d  skipped=%d  errors=%d",
        results["uploaded"], results["skipped"], results["error"],
    )

    if results["error"] > 0:
        logger.warning(
            "%d files failed. Re-run with --no-skip-existing=False to retry only failures, "
            "or check Drive sharing permissions (files must be 'Anyone with the link can view').",
            results["error"],
        )

    # Upload CSV metadata files
    logger.info("Uploading CSV metadata files to gs://%s/data/", args.bucket)
    upload_csv_files(bucket, Path(args.data_dir), args.dry_run)

    # Print expected GCS structure
    logger.info(
        "\nExpected GCS structure after migration:\n"
        "  gs://%s/\n"
        "    raw_pdfs/          # %d PDFs\n"
        "    rendered_pages/    # populated by render_pdf.py\n"
        "    data/\n"
        "      metadata.csv\n"
        "      labels_binary_clean.csv\n"
        "      labels_binary.csv\n"
        "      splits/\n"
        "    checkpoints/\n"
        "      baseline/\n"
        "      dit/\n"
        "    logs/\n"
        "      baseline/\n"
        "      dit/",
        args.bucket,
        len(rows),
    )


if __name__ == "__main__":
    main()
