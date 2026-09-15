"""
Tests for Phase 2: text_cleaner, dataset_split, baseline model, evaluate.

All run against synthetic in-memory data or the tests/fixtures/ CSVs —
no real dataset download required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.preprocessing.text_cleaner import PreprocessConfig, clean_text, clean_series  # noqa: E402
from ml.splitting.dataset_split import group_split, stratified_split, stratified_subsample  # noqa: E402
from ml.models.baseline import build_baseline_pipeline, save_pipeline, load_pipeline  # noqa: E402
from ml.training.evaluate import evaluate_binary  # noqa: E402


# --- text_cleaner ---

def test_clean_text_default_preserves_case_and_punctuation():
    cfg = PreprocessConfig()
    out = clean_text("I feel SO tired!!! :) check http://example.com now", cfg)
    assert "SO" in out  # lowercase off by default
    assert "!!!" in out  # punctuation preserved
    assert "<URL>" in out  # URL replaced by default
    assert "http://example.com" not in out


def test_clean_text_lowercase_when_enabled():
    cfg = PreprocessConfig(lowercase=True)
    out = clean_text("HELLO World", cfg)
    assert out == "hello world"


def test_clean_text_whitespace_normalization():
    cfg = PreprocessConfig()
    out = clean_text("too    many   \n\n  spaces", cfg)
    assert out == "too many spaces"


def test_clean_text_handles_none():
    cfg = PreprocessConfig()
    assert clean_text(None, cfg) == ""


def test_clean_series():
    cfg = PreprocessConfig()
    s = pd.Series(["Hello   world", "Visit https://x.com now"])
    out = clean_series(s, cfg)
    assert out[0] == "Hello world"
    assert "<URL>" in out[1]


# --- dataset_split ---

def _balanced_df(n_per_class=20):
    texts = [f"post {i}" for i in range(n_per_class * 2)]
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({"text": texts, "label": labels})


def test_stratified_split_ratios_approximately_correct():
    df = _balanced_df(50)
    result = stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=1)
    total = len(df)
    assert abs(len(result.train) / total - 0.7) < 0.05
    assert abs(len(result.val) / total - 0.15) < 0.05
    assert abs(len(result.test) / total - 0.15) < 0.05
    # no overlap
    all_idx = set(result.train.index) | set(result.val.index) | set(result.test.index)
    assert len(result.train) + len(result.val) + len(result.test) == total


def test_stratified_split_preserves_class_balance():
    df = _balanced_df(50)
    result = stratified_split(df, random_seed=1)
    for split_df in (result.train, result.val, result.test):
        counts = split_df["label"].value_counts(normalize=True)
        assert abs(counts.get(0, 0) - 0.5) < 0.1


def test_stratified_split_bad_ratios_raises():
    df = _balanced_df(10)
    with pytest.raises(ValueError):
        stratified_split(df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_group_split_keeps_groups_together():
    df = pd.DataFrame(
        {
            "text": [f"post {i}" for i in range(20)],
            "label": [0, 1] * 10,
            "user_id": [i % 5 for i in range(20)],  # 5 groups, 4 rows each
        }
    )
    result = group_split(df, group_col="user_id", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=1)
    train_groups = set(result.train["user_id"])
    val_groups = set(result.val["user_id"])
    test_groups = set(result.test["user_id"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_group_split_missing_column_raises():
    df = _balanced_df(10)
    with pytest.raises(ValueError):
        group_split(df, group_col="nonexistent_col")


# --- stratified_subsample ---

def test_stratified_subsample_reduces_size_and_preserves_balance():
    df = _balanced_df(100)  # 100 class-0, 100 class-1
    sub = stratified_subsample(df, n=40, label_col="label", random_seed=1)
    assert len(sub) == 40
    counts = sub["label"].value_counts()
    assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1  # balance preserved


def test_stratified_subsample_n_larger_than_df_returns_unchanged():
    df = _balanced_df(10)
    sub = stratified_subsample(df, n=1000, label_col="label", random_seed=1)
    assert len(sub) == len(df)


def test_stratified_subsample_reproducible_with_same_seed():
    df = _balanced_df(50)
    sub1 = stratified_subsample(df, n=20, label_col="label", random_seed=7)
    sub2 = stratified_subsample(df, n=20, label_col="label", random_seed=7)
    assert sub1["text"].tolist() == sub2["text"].tolist()


# --- baseline model ---

def test_build_and_fit_baseline_pipeline():
    df = pd.DataFrame(
        {
            "text": [
                "I feel hopeless and want to give up",
                "such a great day at the park today",
                "nothing matters anymore, I feel so empty",
                "just finished a fun movie night with friends",
            ]
            * 5,
            "label": [1, 0, 1, 0] * 5,
        }
    )
    pipeline = build_baseline_pipeline(
        tfidf_cfg={"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0},
        logreg_cfg={"C": 1.0, "max_iter": 200, "class_weight": "balanced", "solver": "liblinear"},
    )
    pipeline.fit(df["text"], df["label"])
    preds = pipeline.predict(df["text"])
    assert len(preds) == len(df)
    assert set(preds) <= {0, 1}


def test_save_and_load_pipeline_roundtrip(tmp_path):
    df = pd.DataFrame(
        {"text": ["cat dog bird", "sun moon star", "cat dog bird", "sun moon star"], "label": [0, 1, 0, 1]}
    )
    pipeline = build_baseline_pipeline(
        tfidf_cfg={"max_features": 50, "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0},
        logreg_cfg={"C": 1.0, "max_iter": 100, "class_weight": "balanced", "solver": "liblinear"},
    )
    pipeline.fit(df["text"], df["label"])
    path = save_pipeline(pipeline, tmp_path)
    assert path.exists()

    loaded = load_pipeline(path)
    np.testing.assert_array_equal(pipeline.predict(df["text"]), loaded.predict(df["text"]))


def test_load_pipeline_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "does_not_exist.joblib")


# --- evaluate ---

def test_evaluate_binary_perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    result = evaluate_binary(y_true, y_pred, y_proba, positive_label=1)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.roc_auc == 1.0
    assert result.false_negative_rate_positive_class == 0.0


def test_evaluate_binary_reports_false_negatives():
    # Both true positives predicted as negative -> FNR should be 1.0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    result = evaluate_binary(y_true, y_pred, positive_label=1)
    assert result.false_negative_rate_positive_class == 1.0
    assert result.per_class["1"]["recall"] == 0.0


def test_evaluate_binary_without_proba_skips_auc():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = evaluate_binary(y_true, y_pred, y_proba=None)
    assert result.roc_auc is None
    assert result.pr_auc is None
