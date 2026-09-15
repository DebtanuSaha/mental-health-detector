"""
dataset_split.py

Train/validation/test splitting with explicit leakage-awareness.

Neither Komati nor Dreaddit exposes a user/author ID in the public CSV
(Dreaddit has post_id, which identifies a *post*, not an *author* —
using it for grouping would not actually prevent same-user leakage
across splits). So today this always falls back to stratified random
splitting by label. Group-based splitting is implemented and tested
here anyway, so it's a one-line config change (`group_based: true` +
`group_column`) if a future dataset provides real author IDs — this
avoids silently reintroducing leakage later by having to bolt it on
under time pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> SplitResult:
    """Standard stratified-by-label random split. Used when no reliable
    grouping column is available (see module docstring)."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[label_col],
        random_state=random_seed,
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_col],
        random_state=random_seed,
    )
    return SplitResult(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> SplitResult:
    """
    Group-based split: all rows sharing the same `group_col` value land
    in the same split, preventing leakage when multiple posts/segments
    come from the same author. Not stratified by label (GroupShuffleSplit
    doesn't support both simultaneously) — check class balance per split
    after calling this, and consider re-seeding if a split comes out
    badly skewed.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if group_col not in df.columns:
        raise ValueError(
            f"group_split requires column '{group_col}', not present in "
            f"dataframe (columns: {list(df.columns)}). Use stratified_split "
            f"instead if no grouping column is available."
        )

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    relative_test_size = test_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=random_seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    return SplitResult(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    label_col: str = "label",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Take a class-balance-preserving subsample of size `n` from `df`.

    For a fast initial fine-tuning pass (e.g. to validate the pipeline
    or estimate training time) before committing to a full run on the
    entire real dataset. Returns `df` unchanged if `n >= len(df)`.
    """
    if n >= len(df):
        return df
    subsampled, _ = train_test_split(
        df, train_size=n, stratify=df[label_col], random_state=random_seed
    )
    return subsampled.reset_index(drop=True)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")
