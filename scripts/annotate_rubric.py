"""
Annotate rubric scores (D, H, S, L) for all rendered page images using Claude vision.

Rubric:
  D — Information Density    0-3  (higher = more content on the page)
  H — Handwriting Dependence 0-3  (higher = more handwriting)
  S — Structural Clarity     0-3  (higher = less clear / more complex structure)
  L — Legibility             0-3  (higher = harder to read)

risk_score = (3 - D) + H + S + L   (range 0-12)
label_ternary:
  risk_score <= 3  -> safe_for_extraction
  risk_score <= 6  -> review
  risk_score > 6   -> high_hallucination_risk

PNG lookup strategy:
  1. Direct stem match: file_path.replace('.pdf', '.png')
  2. Prefix+datetime+page match: for Hebrew-named files where the CSV has underscores
     instead of Hebrew chars, match by (leading-digits-prefix, YYYYMMDD_HHMMSS, page)
  3. Fallback: if no PNG found and label_binary=1 (risky), assign conservative scores
     D=1, H=2, S=1, L=2 -> risk_score=7 -> high_hallucination_risk

Outputs:
  data/labels_rubric.csv         — full rubric annotation file
  data/rubric_checkpoint.jsonl   — incremental checkpoint (resume support)
  Prints validation summary and updates data/metadata.csv at the end.

Usage:
  cd /path/to/for_tal
  python scripts/annotate_rubric.py [--concurrency 10] [--dry-run]
"""

import argparse
import asyncio
import base64
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RENDERED_DIR = DATA_DIR / "rendered_pages"
BINARY_CSV = DATA_DIR / "labels_binary_clean.csv"
RUBRIC_CSV = DATA_DIR / "labels_rubric.csv"
METADATA_CSV = DATA_DIR / "metadata.csv"
CHECKPOINT = DATA_DIR / "rubric_checkpoint.jsonl"

# ─── Thresholds (must match notebook 03) ─────────────────────────────────────
THRESHOLD_SAFE = 3
THRESHOLD_REVIEW = 6

# ─── Claude config ─────────────────────────────────────────────────────────────
# Set ANTHROPIC_API_KEY in your environment (e.g. via .env or shell export).
# Never commit API keys to source control.
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

# ─── Regex helpers for fallback PNG lookup ────────────────────────────────────
DATE_RE = re.compile(r"(\d{8}_\d{6,7})_page_(\d+)")
PREFIX_RE = re.compile(r"^([\d\-]+)")

# ─── Fallback scores for unresolvable risky pages ─────────────────────────────
# These are all label_binary=1 questionnaire pages with fully-Hebrew names
# Assign: dense-ish content, some handwriting, moderate clarity/legibility issues
FALLBACK_SCORES = {"D": 1, "H": 2, "S": 1, "L": 2}  # risk_score = 2+2+1+2 = 7

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


def build_png_lookup() -> dict[tuple, Path]:
    """Build a (prefix, datetime, page) -> Path lookup for fallback resolution."""
    lookup: dict[tuple, list[Path]] = defaultdict(list)
    for p in RENDERED_DIR.glob("*.png"):
        m_dt = DATE_RE.search(p.stem)
        m_pfx = PREFIX_RE.match(p.stem)
        dt = m_dt.group(1) if m_dt else None
        page = m_dt.group(2) if m_dt else None
        pfx = m_pfx.group(1) if m_pfx else ""
        if dt:
            lookup[(pfx, dt, page)].append(p)
    # Keep only unambiguous entries
    return {k: v[0] for k, v in lookup.items() if len(v) == 1}


def resolve_png(file_path: str, png_lookup: dict) -> Path | None:
    """Return the PNG path for a given file_path CSV entry, or None."""
    stem = Path(file_path).stem
    # 1. Direct match
    direct = RENDERED_DIR / f"{stem}.png"
    if direct.exists():
        return direct
    # 2. Prefix+datetime+page match
    m_dt = DATE_RE.search(stem)
    m_pfx = PREFIX_RE.match(stem)
    if m_dt:
        dt = m_dt.group(1)
        page = m_dt.group(2)
        pfx = m_pfx.group(1) if m_pfx else ""
        candidate = png_lookup.get((pfx, dt, page))
        if candidate:
            return candidate
    return None


def load_pages() -> list[dict]:
    """Load file_path + page_num + label_binary from labels_binary_clean.csv."""
    pages = []
    with open(BINARY_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pages.append({
                "file_path": row["file_path"],
                "page_num": int(row["page_num"]),
                "label_binary": int(row["label_binary"]),
            })
    return pages


def load_checkpoint() -> dict[str, dict]:
    """Return dict keyed by file_path for already-scored pages."""
    done: dict[str, dict] = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done[record["file_path"]] = record
    return done


def append_checkpoint(record: dict) -> None:
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def score_page(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    page: dict,
    png_lookup: dict,
    dry_run: bool = False,
) -> dict | None:
    """Call Claude vision and return a scored record, or None on failure."""
    png = resolve_png(page["file_path"], png_lookup)

    # Fallback for unresolvable pages (all label_binary=1)
    if png is None:
        D, H, S, L = (FALLBACK_SCORES[k] for k in ("D", "H", "S", "L"))
        risk = (3 - D) + H + S + L
        return {
            "file_path": page["file_path"],
            "page_num": page["page_num"],
            "D": D, "H": H, "S": S, "L": L,
            "risk_score": risk,
            "label_ternary": ternary_label(risk),
            "_source": "fallback",
        }

    if dry_run:
        return {
            "file_path": page["file_path"],
            "page_num": page["page_num"],
            "D": 2, "H": 0, "S": 0, "L": 0,
            "risk_score": 1,
            "label_ternary": "safe_for_extraction",
            "_source": "dry_run",
        }

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
            text = response.content[0].text.strip()
            # Strip markdown code fence if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            scores = json.loads(text)
            D = max(0, min(3, int(scores["D"])))
            H = max(0, min(3, int(scores["H"])))
            S = max(0, min(3, int(scores["S"])))
            L = max(0, min(3, int(scores["L"])))
            risk = (3 - D) + H + S + L
            return {
                "file_path": page["file_path"],
                "page_num": page["page_num"],
                "D": D, "H": H, "S": S, "L": L,
                "risk_score": risk,
                "label_ternary": ternary_label(risk),
                "_source": "claude",
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] parse error for {page['file_path']!r} (attempt {attempt+1}): {e}", flush=True)
            await asyncio.sleep(1)
        except anthropic.RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"  [RATE LIMIT] waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] {page['file_path']!r}: {e}", flush=True)
            break
    return None


async def annotate_all(concurrency: int, dry_run: bool) -> list[dict]:
    pages = load_pages()
    done = load_checkpoint()
    remaining = [p for p in pages if p["file_path"] not in done]
    png_lookup = build_png_lookup()

    fallback_total = sum(1 for p in pages if resolve_png(p["file_path"], png_lookup) is None)

    print(f"Total pages: {len(pages)}", flush=True)
    print(f"Already done (checkpoint): {len(done)}", flush=True)
    print(f"To annotate: {len(remaining)}", flush=True)
    print(f"  of which fallback (no PNG): {sum(1 for p in remaining if resolve_png(p['file_path'], png_lookup) is None)}", flush=True)
    print(f"  of which Claude vision: {sum(1 for p in remaining if resolve_png(p['file_path'], png_lookup) is not None)}", flush=True)
    if dry_run:
        print("[DRY RUN] — no real API calls will be made", flush=True)

    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    sem = asyncio.Semaphore(concurrency)

    results = list(done.values())
    completed = len(done)
    total = len(pages)
    start = time.time()

    async def process(page: dict) -> None:
        nonlocal completed
        record = await score_page(client, sem, page, png_lookup, dry_run=dry_run)
        if record:
            append_checkpoint(record)
            results.append(record)
        completed += 1
        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        print(
            f"  [{completed}/{total}] {Path(page['file_path']).stem[:50]}"
            f" | {rate:.1f}/s | ETA {eta:.0f}s",
            flush=True,
        )

    await asyncio.gather(*[process(p) for p in remaining])
    return results


def write_rubric_csv(results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r["file_path"])
    fieldnames = ["file_path", "page_num", "D", "H", "S", "L", "risk_score", "label_ternary"]
    with open(RUBRIC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nWrote {len(results_sorted)} rows to {RUBRIC_CSV}", flush=True)


def update_metadata(results: list[dict]) -> None:
    lookup = {r["file_path"]: r for r in results}
    rows = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            fp = row["file_path"]
            if fp in lookup:
                r = lookup[fp]
                row["D"] = r["D"]
                row["H"] = r["H"]
                row["S"] = r["S"]
                row["L"] = r["L"]
                row["risk_score"] = r["risk_score"]
            rows.append(row)

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {METADATA_CSV} with rubric scores", flush=True)


def print_validation(results: list[dict], pages: list[dict]) -> None:
    from collections import Counter
    binary_by_fp = {p["file_path"]: p["label_binary"] for p in pages}
    source_dist = Counter(r.get("_source", "claude") for r in results)

    d_dist = Counter(r["D"] for r in results)
    h_dist = Counter(r["H"] for r in results)
    s_dist = Counter(r["S"] for r in results)
    l_dist = Counter(r["L"] for r in results)
    ternary_dist = Counter(r["label_ternary"] for r in results)
    risk_dist = Counter(r["risk_score"] for r in results)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Sources: {dict(source_dist)}")
    print(f"D: { {k: d_dist[k] for k in sorted(d_dist)} }")
    print(f"H: { {k: h_dist[k] for k in sorted(h_dist)} }")
    print(f"S: { {k: s_dist[k] for k in sorted(s_dist)} }")
    print(f"L: { {k: l_dist[k] for k in sorted(l_dist)} }")
    print(f"risk_score: { {k: risk_dist[k] for k in sorted(risk_dist)} }")
    print(f"\nlabel_ternary:")
    for label in ["safe_for_extraction", "review", "high_hallucination_risk"]:
        print(f"  {label}: {ternary_dist.get(label, 0)}")

    print("\nCross-tab (label_ternary vs label_binary):")
    cross: dict = {}
    from collections import Counter as C
    cross = C()
    for r in results:
        b = binary_by_fp.get(r["file_path"], -1)
        cross[(r["label_ternary"], b)] += 1

    print(f"  {'label_ternary':<35} {'binary=0':>10} {'binary=1':>10}")
    for label in ["safe_for_extraction", "review", "high_hallucination_risk"]:
        print(f"  {label:<35} {cross.get((label, 0), 0):>10} {cross.get((label, 1), 0):>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate rubric scores with Claude vision")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent API requests")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use synthetic scores")
    args = parser.parse_args()

    pages = load_pages()
    results = asyncio.run(annotate_all(args.concurrency, args.dry_run))

    if not results:
        print("No results — nothing to write.", flush=True)
        sys.exit(1)

    write_rubric_csv(results)
    update_metadata(results)
    print_validation(results, pages)

    if CHECKPOINT.exists() and not args.dry_run:
        print(f"\nCheckpoint retained at {CHECKPOINT}")


if __name__ == "__main__":
    main()
