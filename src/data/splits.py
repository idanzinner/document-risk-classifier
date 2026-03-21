"""
splits.py — Grouped train/val/test split utilities.

All splitting strategies keep institution groups whole to prevent
data leakage across splits.  Split assignments are saved to disk
so that downstream steps use exactly the same partitions.
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
    Assigns a 'split' column to metadata_df keeping groups whole.

    Groups are shuffled once with random_state, then greedily assigned to splits
    in proportion to ratios until each split's target group count is reached.

    Args:
        metadata_df: DataFrame with at least group_col column.
        group_col: Column name used to define groups (e.g. 'institution').
        ratios: Dict mapping split name to fraction, e.g.
                {'train': 0.70, 'val': 0.15, 'test': 0.15}.
                Values must sum to 1.0 (within floating-point tolerance).
        random_state: Seed for reproducibility.

    Returns:
        Copy of metadata_df with a new 'split' column (values match ratios keys).

    Raises:
        ValueError: If there are fewer unique groups than splits, or ratios don't sum to 1.
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    groups = np.array(sorted(metadata_df[group_col].unique()))
    n_groups = len(groups)
    n_splits = len(ratios)

    if n_groups < n_splits:
        raise ValueError(
            f"Not enough unique groups ({n_groups}) for {n_splits} splits. "
            "Each split requires at least 1 group."
        )

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(groups)

    # Compute target group counts per split
    split_names = list(ratios.keys())
    fractions = np.array([ratios[s] for s in split_names])

    # Greedy allocation: floor counts, then distribute remainder to largest fractions
    raw_counts = fractions * n_groups
    counts = np.floor(raw_counts).astype(int)
    remainder = n_groups - counts.sum()
    # Give extra groups to splits whose fractional part is largest
    fractional_parts = raw_counts - counts
    extra_indices = np.argsort(fractional_parts)[::-1][:remainder]
    counts[extra_indices] += 1

    # Ensure every split gets at least 1 group
    for i, c in enumerate(counts):
        if c < 1:
            raise ValueError(
                f"Split '{split_names[i]}' received 0 groups with {n_groups} total groups "
                f"and ratio {fractions[i]:.3f}. Increase the number of groups or adjust ratios."
            )

    # Build group -> split mapping
    group_to_split: dict[str, str] = {}
    cursor = 0
    for split_name, count in zip(split_names, counts):
        for g in shuffled[cursor: cursor + count]:
            group_to_split[g] = split_name
        cursor += count

    df = metadata_df.copy()
    df["split"] = df[group_col].map(group_to_split)

    for split_name in split_names:
        n = (df["split"] == split_name).sum()
        logger.info("create_grouped_splits: split='%s' -> %d rows", split_name, n)

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
