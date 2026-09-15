"""
dataset_statistics.py

Computes the descriptive statistics the brief's Phase 1 asks for:
record counts, class distribution, missing values, duplicate records,
text-length distribution, and a small stratified sample for manual
inspection.

Everything here operates on the normalized (text, label, source) schema
produced by dataset_loader.py, so it's dataset-agnostic — the same
functions run against Komati or Dreaddit.

Privacy note: `sample_inspection()` is the ONLY function in this module
that returns raw text, and it's meant for a developer to eyeball locally
during Phase 1 — never wire its output into a persistent log or an API
response. Every other function returns aggregate numbers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DatasetStats:
    dataset_name: str
    n_rows: int
    n_missing_text: int
    n_missing_label: int
    n_duplicate_texts: int
    class_distribution: dict  # {label_value: count}
    class_distribution_pct: dict  # {label_value: percentage}
    text_length_chars: dict  # {mean, median, min, max, std} in characters
    text_length_words: dict  # {mean, median, min, max, std} in words

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_rows": self.n_rows,
            "n_missing_text": self.n_missing_text,
            "n_missing_label": self.n_missing_label,
            "n_duplicate_texts": self.n_duplicate_texts,
            "class_distribution": self.class_distribution,
            "class_distribution_pct": self.class_distribution_pct,
            "text_length_chars": self.text_length_chars,
            "text_length_words": self.text_length_words,
        }


def _length_summary(lengths: pd.Series) -> dict:
    if len(lengths) == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None,
                "p95": None, "p99": None}
    return {
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(lengths.median()), 2),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "std": round(float(lengths.std()), 2),
        "p95": round(float(lengths.quantile(0.95)), 2),
        "p99": round(float(lengths.quantile(0.99)), 2),
    }


def compute_statistics(
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetStats:
    """Compute the full Phase 1 statistics report for one loaded dataset."""
    n_rows = len(df)

    n_missing_text = int(df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum())
    n_missing_label = int(df[label_col].isna().sum())

    non_null_text = df[text_col].dropna().astype(str)
    n_duplicate_texts = int(non_null_text.duplicated(keep="first").sum())

    class_counts = df[label_col].value_counts(dropna=False).to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}
    class_pct = {k: round(100 * v / n_rows, 2) for k, v in class_counts.items()} if n_rows else {}

    char_lengths = non_null_text.str.len()
    word_lengths = non_null_text.str.split().apply(len)

    return DatasetStats(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_missing_text=n_missing_text,
        n_missing_label=n_missing_label,
        n_duplicate_texts=n_duplicate_texts,
        class_distribution=class_counts,
        class_distribution_pct=class_pct,
        text_length_chars=_length_summary(char_lengths),
        text_length_words=_length_summary(word_lengths),
    )


def sample_inspection(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    n_per_class: int = 5,
    max_chars_preview: int = 240,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Return a small, stratified, truncated preview of raw text per class —
    for local manual inspection only (see module docstring privacy note).
    """
    samples = []
    for label_value, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        picked = group.sample(n=n, random_state=random_seed)
        for _, row in picked.iterrows():
            text = str(row[text_col])
            preview = text[:max_chars_preview] + ("…" if len(text) > max_chars_preview else "")
            samples.append({"label": label_value, "text_preview": preview, "char_len": len(text)})
    return pd.DataFrame(samples)


def print_report(stats: DatasetStats) -> None:
    """Pretty-print a DatasetStats to stdout. No raw text is ever included."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {stats.dataset_name}")
    print(f"{'=' * 60}")
    print(f"Total rows:              {stats.n_rows}")
    print(f"Missing/empty text:      {stats.n_missing_text}")
    print(f"Missing label:           {stats.n_missing_label}")
    print(f"Duplicate text rows:     {stats.n_duplicate_texts}")
    print("\nClass distribution:")
    for label, count in stats.class_distribution.items():
        pct = stats.class_distribution_pct.get(label, 0)
        print(f"  {label:>15}: {count:>7} ({pct}%)")
    print("\nText length (characters):")
    for k, v in stats.text_length_chars.items():
        print(f"  {k:>8}: {v}")
    print("\nText length (words):")
    for k, v in stats.text_length_words.items():
        print(f"  {k:>8}: {v}")
