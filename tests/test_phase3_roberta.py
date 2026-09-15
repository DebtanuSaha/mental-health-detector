"""
Tests for Phase 3: TextClassificationDataset, token_length_analysis, and
a full end-to-end fine-tuning smoke test against the tiny local RoBERTa
checkpoint under tests/fixtures/tiny_roberta/ (random weights,
architecturally real — see scripts/build_tiny_test_model.py).

These tests need torch/transformers/datasets installed and the tiny
checkpoint present; they're slower than Phase 1/2's tests (a few seconds
each) since they run real forward/backward passes, just on a tiny model.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

pytest.importorskip("torch")
pytest.importorskip("transformers")

TINY_MODEL_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_roberta"

from ml.data.text_classification_dataset import TextClassificationDataset  # noqa: E402
from ml.data.token_length_analysis import compute_token_length_stats  # noqa: E402
from ml.models.roberta_classifier import load_tokenizer_and_model, save_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_tokenizer():
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built — run scripts/build_tiny_test_model.py")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TINY_MODEL_PATH))


# --- TextClassificationDataset ---

def test_text_classification_dataset_len_and_getitem(tiny_tokenizer):
    encodings = dict(tiny_tokenizer(["hello world", "goodbye world"], padding=True, truncation=True))
    labels = [0, 1]
    ds = TextClassificationDataset(encodings, labels)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert int(item["labels"]) == 0


# --- token_length_analysis ---

def test_compute_token_length_stats_basic(tiny_tokenizer):
    texts = pd.Series(["short text", "a somewhat longer piece of text here", "hi"])
    stats = compute_token_length_stats(texts, tiny_tokenizer, sample_size=None)
    assert stats["n_tokenized"] == 3
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert 0 <= stats["pct_truncated_at_512"] <= 100


def test_compute_token_length_stats_sampling_is_reproducible(tiny_tokenizer):
    texts = pd.Series([f"post number {i} with some words" for i in range(50)])
    stats1 = compute_token_length_stats(texts, tiny_tokenizer, sample_size=10, random_seed=7)
    stats2 = compute_token_length_stats(texts, tiny_tokenizer, sample_size=10, random_seed=7)
    assert stats1 == stats2


def test_truncation_percentages_are_monotonic_non_increasing(tiny_tokenizer):
    texts = pd.Series(["word " * n for n in range(1, 30)])
    stats = compute_token_length_stats(texts, tiny_tokenizer, sample_size=None)
    caps = [128, 256, 384, 512]
    pct = [stats[f"pct_truncated_at_{c}"] for c in caps]
    assert all(pct[i] >= pct[i + 1] for i in range(len(pct) - 1))


# --- roberta_classifier wrapper ---

def test_load_tokenizer_and_model_from_local_path():
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built")
    tokenizer, model = load_tokenizer_and_model(str(TINY_MODEL_PATH), num_labels=2)
    assert tokenizer is not None
    assert model.config.num_labels == 2


def test_save_and_reload_model(tmp_path):
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built")
    tokenizer, model = load_tokenizer_and_model(str(TINY_MODEL_PATH), num_labels=2)
    out_dir = save_model(model, tokenizer, tmp_path / "saved_model")
    assert (out_dir / "config.json").exists()

    from transformers import AutoModelForSequenceClassification
    reloaded = AutoModelForSequenceClassification.from_pretrained(str(out_dir))
    assert reloaded.config.num_labels == 2


# --- end-to-end fine-tuning smoke test ---

def test_end_to_end_finetuning_runs_on_tiny_checkpoint(tmp_path, monkeypatch):
    """
    Full pipeline: load fixtures, dedupe/preprocess/split (Phase 2 code,
    reused), tokenize, fine-tune for 1 tiny epoch, evaluate, save.
    Confirms the real training code path works end-to-end — not just its
    individual pieces — without needing network access or the real
    ~450MB mental-roberta-base download.
    """
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built")

    import yaml
    from ml.training.train_roberta import run_finetuning

    with open(REPO_ROOT / "configs" / "dataset_config.yaml", "r", encoding="utf-8") as f:
        ds_config = yaml.safe_load(f)
    ds_config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
    ds_config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
    ds_config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
    ds_path = tmp_path / "dataset_config.yaml"
    with open(ds_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ds_config, f)

    with open(REPO_ROOT / "configs" / "training.yaml", "r", encoding="utf-8") as f:
        t_config = yaml.safe_load(f)
    t_config["roberta"].update(
        max_seq_length=64, batch_size=2, eval_batch_size=2, num_epochs=1,
        eval_steps=2, logging_steps=1, early_stopping_patience=5,
        output_dir=str(tmp_path / "models_roberta"),
    )
    t_path = tmp_path / "training.yaml"
    with open(t_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(t_config, f)

    metrics = run_finetuning(
        training_config_path=t_path,
        dataset_config_path=ds_path,
        base_dir=REPO_ROOT,
        model_name_or_path_override=str(TINY_MODEL_PATH),
        dataset_override="komati",
    )

    assert metrics["dataset"] == "komati"
    assert metrics["n_train"] > 0
    assert "test_metrics" in metrics
    assert "macro_f1" in metrics["test_metrics"]
    saved_dir = Path(t_config["roberta"]["output_dir"])
    if not saved_dir.is_absolute():
        saved_dir = REPO_ROOT / saved_dir
    assert (saved_dir / "komati" / "config.json").exists()
