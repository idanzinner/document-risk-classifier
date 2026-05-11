"""
annotate_rubric_simple.py — Auto-annotate rubric scores (D, H, S, L) for the
simple two-folder pipeline (ADR-0001).

Reads `data/metadata_simple.csv`, scores each page via Claude vision against
the same D/H/S/L rubric used by `scripts/annotate_rubric.py`, and writes:

  data/labels_rubric_simple.csv         — full rubric annotation file
  data/rubric_simple_checkpoint.jsonl   — incremental checkpoint (resume support)
  data/metadata_simple.csv              — updated in place with D/H/S/L/risk_score

Rubric (identical to legacy `annotate_rubric.py`):
  D — Information Density    0-3
  H — Handwriting Dependence 0-3
  S — Structural Clarity     0-3
  L — Legibility             0-3
  risk_score = (3 - D) + H + S + L           range 0-12
  label_ternary:  ≤ 3 safe_for_extraction | ≤ 6 review | > 6 high_hallucination_risk

PNG resolution
--------------
The simple pipeline uses slug filenames (e.g. `safe__sub__foo__page_001.png`)
that are guaranteed unique, so resolution is a single direct stem match. There
is no Hebrew-name fallback and no fabricated scores — pages missing PNGs are
logged and skipped.

Environment
-----------
Set ANTHROPIC_API_KEY in the environment before running. The `anthropic`
package is required (currently not listed in requirements.txt but used by
the legacy script as well).

Usage
-----
  python scripts/annotate_rubric_simple.py
  python scripts/annotate_rubric_simple.py --concurrency 10
  python scripts/annotate_rubric_simple.py --dry-run        # no API calls; synthetic scores
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("annotate_rubric_simple")

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

DEFAULT_METADATA = DATA_DIR / "metadata_simple.csv"
DEFAULT_RENDERED_DIR = DATA_DIR / "rendered_pages_simple"
DEFAULT_RUBRIC_CSV = DATA_DIR / "labels_rubric_simple.csv"
DEFAULT_CHECKPOINT = DATA_DIR / "rubric_simple_checkpoint.jsonl"

THRESHOLD_SAFE = 3
THRESHOLD_REVIEW = 6

MODEL = "claude-sonnet-4-5"

RUBRIC_PROMPT = """\
You are annotating a scanned Hebrew document page for LLM extraction risk.

Score the page on four dimensions (each 0–3, integer):

D — Information Density
  0: Nearly blank page, whitespace or a single line of text
  1: Sparse content, some printed text but mostly empty
  2: Moderate content, structured text, some tables or fields
  3: Dense content, full paragraphs, tables, or lots of printed text

H — Handwriting Dependence
  0: No handwriting whatsoever — fully printed/typed
  1: Minor handwriting — a signature or one or two filled-in fields
  2: Moderate handwriting — several handwritten fields or annotations
  3: Heavy handwriting — most content is handwritten or large free-text areas

S — Structural Clarity
  0: Clean, clear structure — well-defined fields, readable layout
  1: Mostly clear structure with minor ambiguities
  2: Somewhat unclear structure — overlapping elements, poor alignment
  3: Very unclear structure — complex mixed layout, hard to parse visually

L — Legibility
  0: Perfectly legible — crisp scan, clear text
  1: Mostly legible — minor fading or noise
  2: Partially legible — noticeable degradation, some text hard to read
  3: Poor legibility — heavily degraded scan, significant noise or blur

Respond with ONLY a valid JSON object, no other text:
{"D": <int>, "H": <int>, "S": <int>, "L": <int>}
"""


def ternary_label(risk_score: int) -> str:
    if risk_score <= THRESHOLD_SAFE:
        return "safe_for_extraction"
    if risk_score <= THRESHOLD_REVIEW:
        return "review"
    return "high_hallucination_risk"


def load_pages(metadata_csv: Path) -> list[dict]:
    """Load file_path / page_num / label_binary rows from metadata_simple.csv."""
    pages: list[dict] = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pages.append({
                "file_path": row["file_path"],
                "page_num": int(row["page_num"]),
                "label_binary": int(row["label_binary"]),
            })
    return pages


def load_checkpoint(checkpoint: Path) -> dict[str, dict]:
    """Return existing checkpoint records keyed by file_path."""
    done: dict[str, dict] = {}
    if not checkpoint.exists():
        return done
    with open(checkpoint, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done[record["file_path"]] = record
    return done


def append_checkpoint(checkpoint: Path, record: dict) -> None:
    with open(checkpoint, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_png(file_path: str, rendered_dir: Path) -> Path | None:
    """Direct stem match: <file_path>.pdf → <rendered_dir>/<stem>.png."""
    stem = Path(file_path).stem
    candidate = rendered_dir / f"{stem}.png"
    return candidate if candidate.exists() else None


def _parse_scores(text: str) -> dict:
    """Parse a Claude response into clamped {D, H, S, L} ints."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    raw = json.loads(text)
    return {k: max(0, min(3, int(raw[k]))) for k in ("D", "H", "S", "L")}


async def score_page(
    client,
    sem: asyncio.Semaphore,
    page: dict,
    rendered_dir: Path,
    dry_run: bool,
) -> dict | None:
    """Score a single page; return record dict, or None if PNG missing/failed."""
    png = resolve_png(page["file_path"], rendered_dir)
    if png is None:
        logger.warning("PNG missing for %s — skipping (no fabricated scores)", page["file_path"])
        return None

    if dry_run:
        scores = {"D": 2, "H": 0, "S": 0, "L": 0}
        risk = (3 - scores["D"]) + scores["H"] + scores["S"] + scores["L"]
        return {
            "file_path": page["file_path"],
            "page_num": page["page_num"],
            **scores,
            "risk_score": risk,
            "label_ternary": ternary_label(risk),
            "_source": "dry_run",
        }

    import anthropic  # imported lazily so --dry-run works without the package

    image_data = base64.standard_b64encode(png.read_bytes()).decode("utf-8")

    for attempt in range(3):
        try:
            async with sem:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=64,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": RUBRIC_PROMPT},
                        ],
                    }],
                )
            scores = _parse_scores(response.content[0].text)
            risk = (3 - scores["D"]) + scores["H"] + scores["S"] + scores["L"]
            return {
                "file_path": page["file_path"],
                "page_num": page["page_num"],
                **scores,
                "risk_score": risk,
                "label_ternary": ternary_label(risk),
                "_source": "claude",
            }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "parse error for %s (attempt %d): %s", page["file_path"], attempt + 1, exc,
            )
            await asyncio.sleep(1)
        except anthropic.RateLimitError:
            wait = 5 * (attempt + 1)
            logger.info("rate limit — waiting %ds", wait)
            await asyncio.sleep(wait)
        except Exception as exc:  # noqa: BLE001 — surface and abandon this page
            logger.error("API error for %s: %s", page["file_path"], exc)
            break
    return None


async def annotate_all(
    pages: list[dict],
    rendered_dir: Path,
    checkpoint: Path,
    api_key: str,
    concurrency: int,
    dry_run: bool,
) -> list[dict]:
    done = load_checkpoint(checkpoint)
    remaining = [p for p in pages if p["file_path"] not in done]

    n_scoreable = sum(1 for p in remaining if resolve_png(p["file_path"], rendered_dir) is not None)
    n_missing = len(remaining) - n_scoreable

    print(f"Total pages   : {len(pages)}", flush=True)
    print(f"Already done  : {len(done)}", flush=True)
    print(f"To annotate   : {len(remaining)}", flush=True)
    print(f"  scoreable   : {n_scoreable}", flush=True)
    print(f"  no PNG (skip): {n_missing}", flush=True)
    if dry_run:
        print("[DRY RUN] no real API calls will be made", flush=True)

    if dry_run:
        client = None
    else:
        try:
            import anthropic
        except ImportError:
            logger.error(
                "anthropic package not installed. Run `pip install anthropic` "
                "or pass --dry-run.",
            )
            sys.exit(2)
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set in environment.")
            sys.exit(2)
        client = anthropic.AsyncAnthropic(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = list(done.values())
    completed = len(done)
    total = len(pages)
    start = time.time()

    async def process(page: dict) -> None:
        nonlocal completed
        record = await score_page(client, sem, page, rendered_dir, dry_run=dry_run)
        if record is not None:
            append_checkpoint(checkpoint, record)
            results.append(record)
        completed += 1
        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        print(
            f"  [{completed}/{total}] {Path(page['file_path']).stem[:60]}"
            f" | {rate:.1f}/s | ETA {eta:.0f}s",
            flush=True,
        )

    await asyncio.gather(*[process(p) for p in remaining])
    return results


def write_rubric_csv(results: list[dict], rubric_csv: Path) -> None:
    fieldnames = ["file_path", "page_num", "D", "H", "S", "L", "risk_score", "label_ternary"]
    rubric_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(rubric_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["file_path"]):
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nWrote {len(results)} rows → {rubric_csv}", flush=True)


def update_metadata(results: list[dict], metadata_csv: Path) -> None:
    """Patch D/H/S/L/risk_score in metadata_simple.csv for scored rows."""
    lookup = {r["file_path"]: r for r in results}
    rows: list[dict] = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            fp = row["file_path"]
            if fp in lookup:
                r = lookup[fp]
                for col in ("D", "H", "S", "L", "risk_score"):
                    row[col] = r[col]
            rows.append(row)

    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {metadata_csv} with rubric scores", flush=True)


def print_validation(results: list[dict], pages: list[dict]) -> None:
    binary_by_fp = {p["file_path"]: p["label_binary"] for p in pages}
    source_dist = Counter(r.get("_source", "claude") for r in results)
    dists = {k: Counter(r[k] for r in results) for k in ("D", "H", "S", "L", "risk_score")}
    ternary_dist = Counter(r["label_ternary"] for r in results)
    cross: Counter = Counter()
    for r in results:
        cross[(r["label_ternary"], binary_by_fp.get(r["file_path"], -1))] += 1

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Sources: {dict(source_dist)}")
    for k in ("D", "H", "S", "L", "risk_score"):
        print(f"{k}: {dict(sorted(dists[k].items()))}")
    print("\nlabel_ternary:")
    for label in ("safe_for_extraction", "review", "high_hallucination_risk"):
        print(f"  {label}: {ternary_dist.get(label, 0)}")
    print("\nCross-tab (label_ternary vs label_binary):")
    print(f"  {'label_ternary':<35} {'binary=0':>10} {'binary=1':>10}")
    for label in ("safe_for_extraction", "review", "high_hallucination_risk"):
        print(f"  {label:<35} {cross.get((label, 0), 0):>10} {cross.get((label, 1), 0):>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-annotate rubric scores for the simple pipeline (ADR-0001)",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="Path to metadata_simple.csv (read + updated in place)")
    parser.add_argument("--rendered-dir", type=Path, default=DEFAULT_RENDERED_DIR,
                        help="Directory holding the rendered 224x224 PNGs")
    parser.add_argument("--rubric-csv", type=Path, default=DEFAULT_RUBRIC_CSV,
                        help="Output CSV with full rubric per page")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Incremental checkpoint JSONL (resume support)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent Claude API requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; emit synthetic scores for plumbing tests")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.metadata.exists():
        logger.error("Metadata not found: %s — run build_metadata_simple.py first.", args.metadata)
        sys.exit(1)
    if not args.rendered_dir.is_dir():
        logger.error(
            "Rendered dir not found: %s — run render_pages_simple.py first.",
            args.rendered_dir,
        )
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    pages = load_pages(args.metadata)
    results = asyncio.run(
        annotate_all(
            pages=pages,
            rendered_dir=args.rendered_dir,
            checkpoint=args.checkpoint,
            api_key=api_key,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
    )

    if not results:
        logger.error("No results — nothing to write.")
        sys.exit(1)

    write_rubric_csv(results, args.rubric_csv)
    update_metadata(results, args.metadata)
    print_validation(results, pages)

    if args.checkpoint.exists() and not args.dry_run:
        print(f"\nCheckpoint retained at {args.checkpoint}")


if __name__ == "__main__":
    main()
