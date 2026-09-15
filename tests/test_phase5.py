"""
Tests for Phase 5: Integrated Gradients (against the tiny local RoBERTa
checkpoint — real Captum mechanics, real HF architecture, no network
needed), the baseline linear-coefficient explainer, and the unified
explain() dispatcher's predictor-type routing.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

pytest.importorskip("captum")
pytest.importorskip("torch")

TINY_MODEL_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_roberta"

from ml.explainability.baseline_explainer import explain_with_linear_coefficients  # noqa: E402
from ml.explainability.explainer import explain  # noqa: E402
from ml.explainability.integrated_gradients import _build_baseline_input_ids, explain_with_integrated_gradients  # noqa: E402
from ml.inference.model_predictor import BaselinePredictor, RobertaPredictor  # noqa: E402
from ml.models.baseline import build_baseline_pipeline, save_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def tiny_tokenizer_and_model():
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built — run scripts/build_tiny_test_model.py")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TINY_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(TINY_MODEL_PATH))
    return tok, model


@pytest.fixture
def baseline_pipeline():
    df = pd.DataFrame({
        "text": ["I feel hopeless and want to give up", "great day at the park with friends"] * 10,
        "label": [1, 0] * 10,
    })
    pipeline = build_baseline_pipeline(
        tfidf_cfg={"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0},
        logreg_cfg={"C": 1.0, "max_iter": 200, "class_weight": "balanced", "solver": "liblinear"},
    )
    pipeline.fit(df["text"], df["label"])
    return pipeline


# --- integrated_gradients ---

def test_build_baseline_input_ids_keeps_structural_tokens(tiny_tokenizer_and_model):
    tokenizer, _ = tiny_tokenizer_and_model
    encoded = tokenizer("hello world", return_tensors="pt")
    input_ids = encoded["input_ids"]
    ref = _build_baseline_input_ids(input_ids, tokenizer)

    assert ref.shape == input_ids.shape
    # First token (bos) and last non-pad token (eos) should be unchanged
    assert ref[0, 0].item() == input_ids[0, 0].item()
    last_idx = input_ids.shape[1] - 1
    assert ref[0, last_idx].item() == input_ids[0, last_idx].item()


def test_explain_with_integrated_gradients_returns_one_score_per_token(tiny_tokenizer_and_model):
    tokenizer, model = tiny_tokenizer_and_model
    result = explain_with_integrated_gradients("I feel hopeless today", tokenizer, model, target_class=1, n_steps=10)

    encoded = tokenizer("I feel hopeless today", return_tensors="pt")
    expected_len = encoded["input_ids"].shape[1]

    assert len(result.token_attributions) == expected_len
    assert 0.0 <= result.predicted_proba <= 1.0
    assert result.target_class == 1
    assert len(result.caveats) > 0


def test_explain_with_integrated_gradients_different_target_classes_differ(tiny_tokenizer_and_model):
    tokenizer, model = tiny_tokenizer_and_model
    result_0 = explain_with_integrated_gradients("some text here", tokenizer, model, target_class=0, n_steps=10)
    result_1 = explain_with_integrated_gradients("some text here", tokenizer, model, target_class=1, n_steps=10)
    # Predicted probabilities for class 0 and class 1 should be complementary (softmax over 2 classes)
    assert abs((result_0.predicted_proba + result_1.predicted_proba) - 1.0) < 1e-3


# --- baseline_explainer ---

def test_explain_with_linear_coefficients_identifies_positive_words(baseline_pipeline):
    result = explain_with_linear_coefficients("I feel hopeless today", baseline_pipeline)
    positive_features = {f.feature for f in result.top_positive_features}
    assert "hopeless" in positive_features or "feel" in positive_features
    assert 0.0 <= result.predicted_proba <= 1.0
    assert len(result.caveats) > 0


def test_explain_with_linear_coefficients_empty_text_does_not_crash(baseline_pipeline):
    result = explain_with_linear_coefficients("", baseline_pipeline)
    assert result.top_positive_features == []
    assert result.top_negative_features == []


# --- unified explainer dispatch ---

def test_explain_dispatches_to_integrated_gradients_for_roberta_predictor(tmp_path):
    if not TINY_MODEL_PATH.exists():
        pytest.skip("tests/fixtures/tiny_roberta not built")
    predictor = RobertaPredictor(TINY_MODEL_PATH)
    result = explain("I feel hopeless today", predictor)
    assert result.method == "integrated_gradients"
    d = result.to_dict()
    assert "token_attributions" in d
    assert "caveats" in d


def test_explain_dispatches_to_linear_coefficients_for_baseline_predictor(tmp_path, baseline_pipeline):
    path = save_pipeline(baseline_pipeline, tmp_path, filename="test_pipeline.joblib")
    predictor = BaselinePredictor(path)
    result = explain("I feel hopeless today", predictor)
    assert result.method == "linear_coefficients"
    d = result.to_dict()
    assert "top_positive_features" in d
    assert "caveats" in d


def test_explain_raises_for_unsupported_predictor_type():
    class UnknownPredictor:
        def predict_proba(self, text):
            return 0.5

    with pytest.raises(TypeError):
        explain("text", UnknownPredictor())
