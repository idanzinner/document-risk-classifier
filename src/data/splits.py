"""
splits.py — Grouped train/val/test split utilities.

Two strategies are provided:

create_grouped_splits (original):
    Distributes rows from EACH group proportionally across all splits.
    Every group appears in every split.  Suitable when there are few,
    large groups (e.g. template_family with 3 values).

create_grouped_holdout_splits (new):
    Assigns ENTIRE groups to exactly one split (no within-group split).
    Suitable when group_col is a document-level identifier (source_doc_stem)
    and groups represent whole source documents that must not leak across splits.
    Uses stratified allocation to balance the label distribution per split.

All split assignments are saved to disk so downstream steps use exactly
the same partitions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def create_grouped_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    ratios: dict[str, float],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assigns a 'split' column to metadata_df by splitting samples within each group.

    Every unique value of group_col appears in all splits. Within each group, samples
    are shuffled and divided according to ratios, so the overall sample distribution
    matches the target fractions while all groups are represented everywhere.

    Args:
        metadata_df: DataFrame with at least group_col column.
        group_col: Column name used to define groups (e.g. 'template_family').
        ratios: Dict mapping split name to fraction, e.g.
                {'train': 0.70, 'val': 0.15, 'test': 0.15}.
                Values must sum to 1.0 (within floating-point tolerance).
        random_state: Seed for reproducibility.

    Returns:
        Copy of metadata_df with a new 'split' column (values match ratios keys).

    Raises:
        ValueError: If ratios don't sum to 1, or a group has too few samples to
                    populate every split with at least 1 row.
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    split_names = list(ratios.keys())
    fractions = np.array([ratios[s] for s in split_names])

    rng = np.random.default_rng(random_state)
    df = metadata_df.copy()
    df["split"] = None

    for group_val, group_df in df.groupby(group_col, sort=False):
        idx = group_df.index.to_numpy()
        idx = rng.permutation(idx)
        n = len(idx)

        # Floor-based allocation; distribute remainder to splits with largest fractional parts
        raw_counts = fractions * n
        counts = np.floor(raw_counts).astype(int)
        remainder = n - counts.sum()
        fractional_parts = raw_counts - counts
        extra_indices = np.argsort(fractional_parts)[::-1][:remainder]
        counts[extra_indices] += 1

        if counts.min() < 1:
            # Groups too small to populate every split get assigned to the dominant split (train)
            dominant = split_names[int(np.argmax(fractions))]
            logger.warning(
                "Group '%s' has only %d samples — assigning all to '%s'",
                group_val, n, dominant,
            )
            df.loc[idx, "split"] = dominant
            continue

        cursor = 0
        for split_name, count in zip(split_names, counts):
            df.loc[idx[cursor: cursor + count], "split"] = split_name
            cursor += count

    for split_name in split_names:
        n = (df["split"] == split_name).sum()
        logger.info("create_grouped_splits: split='%s' -> %d rows", split_name, n)

    return df


def create_grouped_holdout_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    ratios: dict[str, float],
    stratify_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assigns every unique group (by group_col) ENTIRELY to one split.

    Unlike create_grouped_splits (which distributes rows from each group
    proportionally across all splits), this function assigns each WHOLE
    group to exactly one split.  This is the correct strategy when
    group_col is a document-level identifier (e.g. source_doc_stem) and
    all pages from the same source PDF must not appear in more than one split.

    Groups are allocated to splits proportionally by group count (not page count),
    stratified by stratify_col so that each split has a balanced label distribution.
    Groups too few to populate every split (e.g. a stratum with only 2 groups for
    3 splits) are assigned to the split with the highest ratio.

    Args:
        metadata_df:  DataFrame with at least group_col column.
        group_col:    Column whose unique values define groups (e.g. 'source_doc_stem').
        ratios:       Dict mapping split name → fraction; values must sum to 1.0.
        stratify_col: Optional column for stratification.  Majority label within
                      each group is used as that group's stratum label.
        random_state: Seed for reproducibility.

    Returns:
        Copy of metadata_df with 'split' column added (or replaced).
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    split_names = list(ratios.keys())
    fractions = np.array([ratios[s] for s in split_names])
    dominant_split = split_names[int(np.argmax(fractions))]

    rng = np.random.default_rng(random_state)
    df = metadata_df.copy()
    df["split"] = None

    # Build group → stratum mapping (majority label within each group)
    if stratify_col is not None and stratify_col in df.columns:
        group_stratum = (
            df.groupby(group_col)[stratify_col]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={stratify_col: "_stratum"})
        )
    else:
        groups_unique = df[[group_col]].drop_duplicates()
        group_stratum = groups_unique.assign(_stratum="all")

    for stratum_val, sg in group_stratum.groupby("_stratum"):
        groups = rng.permutation(sg[group_col].values)
        n = len(groups)
        if n == 0:
            continue

        # Proportional floor allocation across splits
        raw_counts = fractions * n
        counts = np.floor(raw_counts).astype(int)
        remainder = int(n - counts.sum())
        if remainder > 0:
            extra_indices = np.argsort(raw_counts - counts)[::-1][:remainder]
            counts[extra_indices] += 1

        # When a stratum is too small to put at least 1 group per split,
        # assign all groups to the dominant split (usually train)
        if counts.min() < 0 or n < len(split_names):
            counts[:] = 0
            counts[np.argmax(fractions)] = n

        cursor = 0
        for split_name, count in zip(split_names, counts):
            for group in groups[cursor: cursor + count]:
                df.loc[df[group_col] == group, "split"] = split_name
            cursor += count

    # Safety net: any unassigned rows go to the dominant split
    unassigned = df["split"].isna().sum()
    if unassigned > 0:
        logger.warning(
            "create_grouped_holdout_splits: %d unassigned rows → assigned to '%s'",
            unassigned, dominant_split,
        )
        df.loc[df["split"].isna(), "split"] = dominant_split

    for split_name in split_names:
        n = (df["split"] == split_name).sum()
        logger.info(
            "create_grouped_holdout_splits: split='%s' -> %d rows",
            split_name, n,
        )

    return df


def create_kfold_splits(
    metadata_df: pd.DataFrame,
    group_col: str,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Grouped k-fold cross-validation splits.

    Uses sklearn's GroupKFold to ensure that no group appears in both
    train and validation folds.  The row order is shuffled before folding
    for reproducibility.

    Args:
        metadata_df: DataFrame with at least group_col column.
        group_col: Column used to define groups.
        n_folds: Number of folds.
        random_state: Seed for reproducibility.

    Returns:
        List of (train_df, val_df) tuples, length == n_folds.
    """
    df = metadata_df.copy().reset_index(drop=True)

    # Shuffle rows reproducibly before folding
    rng = np.random.default_rng(random_state)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    groups = df[group_col].values
    gkf = GroupKFold(n_splits=n_folds)

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        logger.info(
            "create_kfold_splits: fold %d/%d -> train=%d rows, val=%d rows",
            fold_idx + 1, n_folds, len(train_df), len(val_df),
        )
        folds.append((train_df, val_df))

    return folds


def save_splits(metadata_df: pd.DataFrame, output_dir: str) -> None:
    """
    Saves train.csv, val.csv, test.csv to output_dir.

    Expects metadata_df to have a 'split' column already assigned.

    Args:
        metadata_df: DataFrame with 'split' column.
        output_dir: Directory path (created if it does not exist).

    Raises:
        KeyError: If 'split' column is missing from metadata_df.
    """
    if "split" not in metadata_df.columns:
        raise KeyError("metadata_df must have a 'split' column. Run create_grouped_splits first.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split_name in metadata_df["split"].unique():
        split_df = metadata_df[metadata_df["split"] == split_name].reset_index(drop=True)
        csv_path = out_path / f"{split_name}.csv"
        split_df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(split_df), csv_path)


def load_splits(splits_dir: str) -> dict[str, pd.DataFrame]:
    """
    Loads pre-saved split files from splits_dir.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Dict {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}.

    Raises:
        FileNotFoundError: If any of the three expected CSV files is missing.
    """
    path = Path(splits_dir)
    result: dict[str, pd.DataFrame] = {}

    for split_name in ("train", "val", "test"):
        csv_path = path / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {csv_path}. "
                "Run save_splits() before load_splits()."
            )
        result[split_name] = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(result[split_name]), csv_path)

    return result
