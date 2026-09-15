"""
Tests for Phase 4 risk scoring — uses fake predictors/sentiment (fixed
outputs) so the threshold/combination logic is tested deterministically,
without needing real trained models or network access.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.inference.sentiment import SentimentResult  # noqa: E402
from ml.risk.risk_scoring import assess_risk, load_risk_config, _risk_level_from_score  # noqa: E402


class FakePredictor:
    def __init__(self, proba: float):
        self.proba = proba

    def predict_proba(self, text: str) -> float:
        return self.proba


class FakeSentimentAnalyzer:
    def __init__(self, label="neutral", confidence=0.5):
        self._label = label
        self._confidence = confidence

    def analyze(self, text: str) -> SentimentResult:
        return SentimentResult(label=self._label, confidence=self._confidence)


@pytest.fixture
def config():
    return load_risk_config(REPO_ROOT / "configs" / "risk_config.yaml")


def test_load_risk_config_has_required_sections(config):
    for key in ("thresholds", "signal_weights", "category_detection_threshold",
                "safety_messages", "disclaimer", "model_paths", "sentiment"):
        assert key in config


def test_risk_level_from_score_boundaries(config):
    t = config["thresholds"]
    assert _risk_level_from_score(0.0, t) == "LOW"
    assert _risk_level_from_score(t["moderate"], t) == "MODERATE"
    assert _risk_level_from_score(t["high"], t) == "HIGH"
    assert _risk_level_from_score(t["critical"], t) == "CRITICAL"
    assert _risk_level_from_score(1.0, t) == "CRITICAL"


def test_assess_risk_low_when_both_signals_low(config):
    result = assess_risk(
        "just had a nice sandwich",
        FakePredictor(0.02), FakePredictor(0.03), FakeSentimentAnalyzer("positive", 0.9), config,
    )
    assert result.risk_level == "LOW"
    assert result.detected_categories == []


def test_assess_risk_critical_when_crisis_signal_very_high(config):
    result = assess_risk(
        "some crisis text", FakePredictor(0.98), FakePredictor(0.10), FakeSentimentAnalyzer("negative", 0.8), config,
    )
    assert result.risk_level == "CRITICAL"
    assert "suicide_indicator" in result.detected_categories
    assert "emotional_distress" not in result.detected_categories


def test_assess_risk_both_categories_detected_when_both_signals_high(config):
    result = assess_risk(
        "text", FakePredictor(0.9), FakePredictor(0.9), FakeSentimentAnalyzer(), config,
    )
    assert "suicide_indicator" in result.detected_categories
    assert "emotional_distress" in result.detected_categories


def test_assess_risk_never_uses_diagnostic_language(config):
    result = assess_risk(
        "text", FakePredictor(0.9), FakePredictor(0.9), FakeSentimentAnalyzer("negative", 0.9), config,
    )
    full_text = " ".join(result.explanation) + " " + result.recommendation["message"] + " " + result.disclaimer
    lowered = full_text.lower()
    for phrase in ("you are suicidal", "you have depression", "this person is", "this person has"):
        assert phrase not in lowered


def test_assess_risk_sentiment_does_not_affect_risk_score(config):
    # Same crisis/distress signals, different sentiment -> identical risk_score/level.
    r1 = assess_risk("t", FakePredictor(0.6), FakePredictor(0.4), FakeSentimentAnalyzer("positive", 0.99), config)
    r2 = assess_risk("t", FakePredictor(0.6), FakePredictor(0.4), FakeSentimentAnalyzer("negative", 0.99), config)
    assert r1.risk_score == r2.risk_score
    assert r1.risk_level == r2.risk_level
    assert r1.sentiment["label"] != r2.sentiment["label"]  # sentiment itself still differs and is reported


def test_assess_risk_low_risk_has_no_resources(config):
    result = assess_risk(
        "t", FakePredictor(0.02), FakePredictor(0.02), FakeSentimentAnalyzer(), config,
        resources_path=REPO_ROOT / "resources" / "verified_resources.json",
    )
    assert result.risk_level == "LOW"
    assert result.resources == []


def test_assess_risk_non_low_risk_includes_international_resource(config):
    result = assess_risk(
        "t", FakePredictor(0.9), FakePredictor(0.9), FakeSentimentAnalyzer(), config,
        resources_path=REPO_ROOT / "resources" / "verified_resources.json",
    )
    assert result.risk_level != "LOW"
    assert len(result.resources) >= 1
    assert any(r["country_code"] == "INTL" for r in result.resources)


def test_assess_risk_country_code_reaches_resource_filtering(config):
    result = assess_risk(
        "t", FakePredictor(0.9), FakePredictor(0.9), FakeSentimentAnalyzer(), config,
        country_code="IN", resources_path=REPO_ROOT / "resources" / "verified_resources.json",
    )
    country_codes = {r["country_code"] for r in result.resources}
    assert "IN" in country_codes
    assert "INTL" in country_codes


def test_assess_risk_confidence_in_valid_range_and_increases_with_certainty(config):
    uncertain = assess_risk("t", FakePredictor(0.5), FakePredictor(0.5), FakeSentimentAnalyzer(), config)
    assert 0.0 <= uncertain.confidence <= 1.0

    certain = assess_risk("t", FakePredictor(0.99), FakePredictor(0.01), FakeSentimentAnalyzer(), config)
    assert 0.0 <= certain.confidence <= 1.0
    assert certain.confidence > uncertain.confidence


def test_assess_risk_recommendation_matches_risk_level(config):
    result = assess_risk("t", FakePredictor(0.02), FakePredictor(0.02), FakeSentimentAnalyzer(), config)
    assert result.recommendation["type"] == config["safety_messages"]["LOW"]["type"]


def test_to_dict_matches_brief_response_shape(config):
    result = assess_risk("t", FakePredictor(0.5), FakePredictor(0.5), FakeSentimentAnalyzer(), config)
    d = result.to_dict()
    for key in ("risk_level", "risk_score", "confidence", "sentiment",
                "detected_categories", "explanation", "recommendation", "disclaimer", "resources"):
        assert key in d
