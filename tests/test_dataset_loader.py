"""
Tests for ml/data/dataset_loader.py and ml/data/dataset_statistics.py,
run against the small synthetic fixtures under tests/fixtures/.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.data.dataset_loader import (  # noqa: E402
    DatasetLoadError,
    load_config,
    load_dreaddit,
    load_komati,
)
from ml.data.dataset_statistics import compute_statistics, sample_inspection  # noqa: E402


@pytest.fixture
def fixture_config():
    config = load_config(REPO_ROOT / "configs" / "dataset_config.yaml")
    config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
    config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
    config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
    return config


def test_load_komati_normalizes_labels(fixture_config):
    loaded = load_komati(fixture_config, base_dir=REPO_ROOT)
    assert set(loaded.df.columns) == {"text", "label", "source"}
    assert set(loaded.df["label"].unique()) <= {0, 1}
    assert (loaded.df["source"] == "komati").all()
    assert len(loaded.df) == 12


def test_load_komati_missing_file_raises(fixture_config, tmp_path):
    fixture_config["komati"]["raw_path"] = "does/not/exist.csv"
    with pytest.raises(DatasetLoadError):
        load_komati(fixture_config, base_dir=REPO_ROOT)


def test_load_komati_unknown_label_value_raises(fixture_config, tmp_path):
    bad_csv = tmp_path / "bad_komati.csv"
    bad_csv.write_text('text,class\n"hello there",mystery_label\n')
    fixture_config["komati"]["raw_path"] = str(bad_csv)
    with pytest.raises(DatasetLoadError):
        load_komati(fixture_config, base_dir=".")


def test_load_dreaddit_train_split(fixture_config):
    loaded = load_dreaddit(fixture_config, base_dir=REPO_ROOT, split="train")
    assert set(loaded.df.columns) == {"text", "label", "source"}
    assert len(loaded.df) == 8
    assert (loaded.df["source"] == "dreaddit").all()


def test_load_dreaddit_keep_liwc_features(fixture_config):
    loaded = load_dreaddit(
        fixture_config, base_dir=REPO_ROOT, split="train", keep_liwc_features=True
    )
    assert "lex_liwc_WC" in loaded.df.columns
    assert "lex_liwc_Tone" in loaded.df.columns


def test_compute_statistics_class_distribution(fixture_config):
    loaded = load_komati(fixture_config, base_dir=REPO_ROOT)
    stats = compute_statistics(loaded.df, dataset_name="komati")
    assert stats.n_rows == 12
    assert sum(stats.class_distribution.values()) == 12
    assert stats.n_missing_text == 0
    assert stats.n_missing_label == 0


def test_compute_statistics_detects_duplicates_and_missing():
    df = pd.DataFrame(
        {
            "text": ["hello world", "hello world", "", None, "another post"],
            "label": [0, 0, 1, 1, 0],
        }
    )
    stats = compute_statistics(df, dataset_name="synthetic")
    assert stats.n_rows == 5
    assert stats.n_missing_text == 2  # empty string + None
    assert stats.n_duplicate_texts == 1  # second "hello world"


def test_sample_inspection_is_stratified_and_truncated():
    df = pd.DataFrame(
        {
            "text": [f"post number {i} " + "x" * 300 for i in range(10)],
            "label": [0, 1] * 5,
        }
    )
    preview = sample_inspection(df, n_per_class=2, max_chars_preview=50, random_seed=1)
    assert len(preview) == 4  # 2 per class x 2 classes
    assert (preview["char_len"] > 50).all()  # original length preserved in char_len
    assert preview["text_preview"].str.len().max() <= 51  # truncated + ellipsis char


def test_sample_inspection_handles_class_with_fewer_rows_than_requested():
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 0, 1]})
    preview = sample_inspection(df, n_per_class=5, max_chars_preview=100, random_seed=1)
    # label 0 only has 2 rows, label 1 only has 1 — should not error, just cap
    assert len(preview) == 3
