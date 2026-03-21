"""
clean_labels.py — Align labels_binary.csv with the INTERFACES.md schema and
build the master metadata.csv required by the training pipeline.

Input:  data/labels_binary.csv  (columns: file name, link, label_binary, local path)
Output:
  data/labels_binary_clean.csv  — conforms to INTERFACES.md labels_binary schema
  data/metadata.csv             — full master index with all schema columns

Usage:
    python scripts/clean_labels.py [--input data/labels_binary.csv] [--out-dir data/]

Institution logic:
  - "no risk" files are all in 'מסמכים רגילים' (regular documents) → institution = "regular_docs"
  - "high risk" files are all in 'pdf_s - שאלונים וכתב יד' (questionnaires) → institution = "questionnaires"

WARNING: Label is perfectly correlated with institution/folder. Flag this before
training — grouped splits will be degenerate if institution == label source.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def extract_file_id(drive_link: str) -> str:
    """Extract the file ID from a Google Drive viewer URL."""
    match = re.search(r"/file/d/([^/]+)/", drive_link)
    if match:
        return match.group(1)
    return ""


def extract_page_num(filename: str) -> int:
    """Extract 1-based page number from filename like 'doc_page_0003.pdf'."""
    match = re.search(r"_page_(\d+)\.pdf$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def derive_institution(label_text: str) -> str:
    """Map raw label text to institution group.

    WARNING: institution is perfectly correlated with label in the current
    dataset. All 'no risk' files originate from 'regular_docs' and all
    'high risk' files from 'questionnaires'. Verify this does not introduce
    data leakage before using institution as a grouping key for splits.
    """
    if label_text.strip().lower() == "no risk":
        return "regular_docs"
    return "questionnaires"


def derive_template_family(filename: str, institution: str) -> str:
    """Best-effort template family from filename prefix."""
    stem = Path(filename).stem  # e.g. '1cb73767-...-page_0001'
    # UUIDs → generic document
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
    )
    if uuid_pattern.match(stem):
        if institution == "regular_docs":
            return "regular_form"
        return "questionnaire_uuid"
    # Hebrew filenames with numeric prefix like '104-105...' → named questionnaire
    if institution == "questionnaires":
        return "questionnaire_named"
    return "regular_form"


def convert_label(label_text: str) -> int:
    """Convert 'no risk' -> 0, 'high risk' -> 1."""
    mapping = {"no risk": 0, "high risk": 1}
    key = label_text.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unrecognised label value: '{label_text}'")
    return mapping[key]


def clean_labels(input_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        print("ERROR: input CSV is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(raw_rows)} rows from {input_path}")

    for raw in raw_rows:
        filename = raw.get("file name", "").strip()
        drive_link = raw.get("link", "").strip()
        label_text = raw.get("label_binary", "").strip()

        if not filename or not label_text:
            print(f"  SKIP (missing filename or label): {raw}", file=sys.stderr)
            continue

        label_int = convert_label(label_text)
        page_num = extract_page_num(filename)
        institution = derive_institution(label_text)
        template_family = derive_template_family(filename, institution)
        drive_id = extract_file_id(drive_link)

        rows.append(
            {
                "file_path": filename,
                "page_num": page_num,
                "institution": institution,
                "template_family": template_family,
                "label_binary": label_int,
                "drive_link": drive_link,
                "drive_id": drive_id,
            }
        )

    print(f"Cleaned {len(rows)} rows")
    label_dist = {0: sum(1 for r in rows if r["label_binary"] == 0),
                  1: sum(1 for r in rows if r["label_binary"] == 1)}
    print(f"Label distribution: safe={label_dist[0]}, risky={label_dist[1]}")
    inst_dist = {}
    for r in rows:
        inst_dist[r["institution"]] = inst_dist.get(r["institution"], 0) + 1
    print(f"Institution distribution: {inst_dist}")
    print(
        "\nWARNING: institution is perfectly correlated with label. "
        "Grouped splits by institution will be train=safe/test=risky or vice-versa. "
        "Consider using document-family or file-prefix as the grouping key instead.\n"
    )

    # Write labels_binary_clean.csv (matches INTERFACES.md schema)
    labels_clean_path = out_dir / "labels_binary_clean.csv"
    labels_fieldnames = ["file_path", "page_num", "label_binary", "drive_link", "drive_id"]
    with open(labels_clean_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=labels_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {labels_clean_path}")

    # Write metadata.csv (full master index — INTERFACES.md metadata schema)
    metadata_path = out_dir / "metadata.csv"
    metadata_fieldnames = [
        "file_path", "page_num", "institution", "template_family",
        "label_binary", "D", "H", "S", "L", "risk_score", "split",
        "drive_link", "drive_id",
    ]
    metadata_rows = []
    for r in rows:
        metadata_rows.append(
            {
                "file_path": r["file_path"],
                "page_num": r["page_num"],
                "institution": r["institution"],
                "template_family": r["template_family"],
                "label_binary": r["label_binary"],
                "D": -1,       # stub — populate during rubric annotation
                "H": -1,
                "S": -1,
                "L": -1,
                "risk_score": -1,
                "split": "",   # populated by src/data/splits.py
                "drive_link": r["drive_link"],
                "drive_id": r["drive_id"],
            }
        )
    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Wrote {metadata_path}")
    print("\nNext steps:")
    print("  1. Run scripts/drive_to_gcs.py to upload PDFs to GCS")
    print("  2. Populate D/H/S/L rubric scores in metadata.csv before training")
    print("  3. Run src/data/splits.py to assign the 'split' column")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean labels_binary.csv for GCP pipeline")
    parser.add_argument(
        "--input",
        default="data/labels_binary.csv",
        help="Path to raw labels CSV (default: data/labels_binary.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/",
        help="Output directory for cleaned CSVs (default: data/)",
    )
    args = parser.parse_args()

    clean_labels(Path(args.input), Path(args.out_dir))


if __name__ == "__main__":
    main()
