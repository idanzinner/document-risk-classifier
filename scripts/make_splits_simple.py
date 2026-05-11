"""
make_splits_simple.py — Assign random 70/15/15 train/val/test splits to
data/metadata_simple.csv, stratified by label_binary.

Third stage of the simple two-folder pipeline (ADR-0001). Pre-requisite:
`build_metadata_simple.py` must have produced data/metadata_simple.csv.

Strategy
--------
Random per-page split, stratified by `label_binary`. Each page is treated
as an independent sample; there is no source-document grouping in the
simple pipeline. (The legacy pipeline groups by `source_doc_stem` to
prevent page leakage from multi-page source PDFs — this is a deliberate
simplification per the user's choice in ADR-0001.)

Outputs
-------
- Updates `split` column in `data/metadata_simple.csv` in place.
- Writes one CSV per split to `data/splits_simple/{train,val,test}.csv`.

Usage:
  python scripts/make_splits_simple.py
  python scripts/make_splits_simple.py --train-ratio 0.8 --val-ratio 0.1
  python scripts/make_splits_simple.py --seed 0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("make_splits_simple")

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"

DEFAULT_METADATA = DATA_DIR / "metadata_simple.csv"
DEFAULT_SPLITS_DIR = DATA_DIR / "splits_simple"


def stratified_three_way_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Assign 'split' column ('train'/'val'/'test') stratified by label_col.

    Pure-numpy implementation so we don't lean on sklearn for a 3-way split.
    Per-class indices are shuffled with a seeded generator, then sliced by
    floor-then-distribute-remainder so the totals always equal class size.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError(
            f"Ratios must be non-negative; got train={train_ratio}, "
            f"val={val_ratio}, test={test_ratio:.6f}"
        )
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    out = df.copy()
    out["split"] = ""
    rng = np.random.default_rng(seed)

    for cls, group in out.groupby(label_col, sort=False):
        idx = group.index.to_numpy()
        idx = rng.permutation(idx)
        n = len(idx)

        # Floor allocation, distribute remainder to highest fractional part
        fractions = np.array([train_ratio, val_ratio, test_ratio])
        raw = fractions * n
        counts = np.floor(raw).astype(int)
        remainder = int(n - counts.sum())
        if remainder > 0:
            order = np.argsort(raw - counts)[::-1][:remainder]
            counts[order] += 1

        n_train, n_val, n_test = counts
        out.loc[idx[:n_train], "split"] = "train"
        out.loc[idx[n_train: n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val:], "split"] = "test"

        logger.info(
            "Class %s: n=%d → train=%d val=%d test=%d",
            cls, n, n_train, n_val, n_test,
        )

    if (out["split"] == "").any():
        n_missing = int((out["split"] == "").sum())
        raise RuntimeError(f"{n_missing} rows did not receive a split assignment")

    return out


def main(
    metadata_csv: Path = DEFAULT_METADATA,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
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

    meta = pd.read_csv(metadata_csv)
    logger.info("Loaded %d rows from %s", len(meta), metadata_csv)

    required = {"file_path", "label_binary"}
    missing = required - set(meta.columns)
    if missing:
        logger.error("Metadata missing required columns: %s", sorted(missing))
        sys.exit(1)

    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    logger.info(
        "Stratified random split:  train=%.2f  val=%.2f  test=%.4f  seed=%d",
        train_ratio, val_ratio, test_ratio, seed,
    )

    meta_split = stratified_three_way_split(
        meta,
        label_col="label_binary",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print()
    print("Split summary:")
    print(f"{'split':<8} {'rows':>6}  {'safe':>5}  {'risky':>6}")
    for split_name in ("train", "val", "test"):
        subset = meta_split[meta_split["split"] == split_name]
        safe = int((subset["label_binary"] == 0).sum())
        risky = int((subset["label_binary"] == 1).sum())
        print(f"{split_name:<8} {len(subset):>6}  {safe:>5}  {risky:>6}")

    meta_split.to_csv(metadata_csv, index=False, encoding="utf-8")
    logger.info("Wrote updated %s", metadata_csv)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        subset = meta_split[meta_split["split"] == split_name].reset_index(drop=True)
        out = splits_dir / f"{split_name}.csv"
        subset.to_csv(out, index=False, encoding="utf-8")
        logger.info("Wrote %d rows → %s", len(subset), out)

    print()
    print(f"Done.  Configs that consume these artefacts:")
    print(f"  configs/baseline_simple.yaml")
    print(f"  configs/dit_simple.yaml")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified random 70/15/15 split for metadata_simple.csv",
    )
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        metadata_csv=args.metadata,
        splits_dir=args.splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
