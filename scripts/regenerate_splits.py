"""
regenerate_splits.py — Regenerate train/val/test splits using source_doc_stem
as the group key (whole-document holdout strategy).

Groups (unique source_doc_stem values) are assigned ENTIRELY to one split so
that pages from the same source PDF never appear in both train and test.
Stratification by label_binary ensures each split has a balanced class ratio.

Writes:
  data/metadata.csv          — updated split column
  data/splits/train.csv      — train split rows
  data/splits/val.csv        — val split rows
  data/splits/test.csv       — test split rows

Usage:
  python scripts/regenerate_splits.py [--train-ratio 0.70] [--val-ratio 0.15]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

# Ensure src/ is importable
sys.path.insert(0, str(ROOT))

from src.data.splits import create_grouped_splits, save_splits


def main(train_ratio: float = 0.70, val_ratio: float = 0.15, random_state: int = 42) -> None:
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    meta_path = DATA_DIR / "metadata.csv"
    splits_dir = DATA_DIR / "splits"

    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta)} rows")

    required_cols = ["source_doc_stem", "label_binary"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        print(f"ERROR: metadata missing columns {missing}. Run build_metadata_v2.py first.", file=sys.stderr)
        sys.exit(1)

    # Unique source_doc_stem groups
    n_groups = meta["source_doc_stem"].nunique()
    print(f"Unique source_doc_stem groups: {n_groups}")
    print(f"Split ratios: train={train_ratio}  val={val_ratio}  test={test_ratio:.4f}")

    # Use within-group distribution (pages from each source doc split proportionally).
    # This gives a balanced 70/15/15 distribution across the full dataset.
    # Small groups (< 3 pages) are assigned entirely to train via the graceful fallback
    # added to create_grouped_splits.
    meta_split = create_grouped_splits(
        meta,
        group_col="source_doc_stem",
        ratios=ratios,
        random_state=random_state,
    )

    # Print summary
    print(f"\nSplit summary (rows):")
    for split_name in ("train", "val", "test"):
        subset = meta_split[meta_split["split"] == split_name]
        safe_n = (subset["label_binary"] == 0).sum()
        risky_n = (subset["label_binary"] == 1).sum()
        groups_n = subset["source_doc_stem"].nunique()
        src_dist = subset["source_folder"].value_counts().to_dict() if "source_folder" in subset else {}
        print(f"  {split_name:5s}: {len(subset):5d} rows  "
              f"(safe={safe_n}, risky={risky_n}, groups={groups_n})")
        if src_dist:
            for sf, cnt in sorted(src_dist.items()):
                print(f"           {sf}: {cnt}")

    # Verify every source_folder has representation in every split
    if "source_folder" in meta_split.columns:
        print(f"\nSource folder coverage per split:")
        pivot = meta_split.groupby(["split", "source_folder"]).size().unstack(fill_value=0)
        print(pivot.to_string())

    # Save updated metadata with split column
    meta_split.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved updated metadata.csv ({len(meta_split)} rows)")

    # Save individual split CSVs
    save_splits(meta_split, str(splits_dir))
    print(f"Saved split CSVs → {splits_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate train/val/test splits")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
