"""
risk_scoring.py

Combines the crisis-classifier probability (Komati-trained) and the
distress-classifier probability (Dreaddit-trained) into a single
probabilistic risk score and LOW/MODERATE/HIGH/CRITICAL category, per
brief §6. Sentiment is reported alongside as a secondary, informational
signal — per brief §7 it is deliberately NOT fused into the numeric
risk score, since sentiment != crisis risk in either direction
(negative sentiment != suicide risk, positive sentiment != absence of
risk). test_assess_risk_sentiment_does_not_affect_risk_score in the
test suite enforces this as a real invariant, not just a comment.

Thresholds and signal weights are read from configs/risk_config.yaml —
never hard-coded here — and are explicitly starting points requiring
real-world validation before any deployment (brief §6 requirement 4).

This module never produces diagnostic language. Output uses only the
brief's approved phrasing ("potential ... indicators detected",
"elevated risk signal", "the model estimates ...") — never "this
person has/is ...".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ml.inference.model_predictor import TextProbabilityPredictor
from ml.inference.sentiment import SentimentAnalyzer
from ml.resources.resource_service import get_resources_for_risk_level

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")


@dataclass
class RiskAssessmentResult:
    risk_level: str
    risk_score: float
    confidence: float
    sentiment: dict
    detected_categories: list
    explanation: list
    recommendation: dict
    disclaimer: str
    resources: list = field(default_factory=list)  # populated by get_resources_for_risk_level() in assess_risk()

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "sentiment": self.sentiment,
            "detected_categories": self.detected_categories,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "disclaimer": self.disclaimer,
            "resources": self.resources,
        }


def load_risk_config(path: str | Path = "configs/risk_config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _risk_level_from_score(score: float, thresholds: dict) -> str:
    if score >= thresholds["critical"]:
        return "CRITICAL"
    if score >= thresholds["high"]:
        return "HIGH"
    if score >= thresholds["moderate"]:
        return "MODERATE"
    return "LOW"


def _build_explanation(crisis_prob: float, distress_prob: float, category_threshold: float) -> list:
    explanation = []
    if crisis_prob >= category_threshold:
        explanation.append(f"Potential crisis-related language detected (model probability: {crisis_prob:.2f}).")
    if distress_prob >= category_threshold:
        explanation.append(f"Potential emotional distress indicators detected (model probability: {distress_prob:.2f}).")
    if not explanation:
        explanation.append("No strong distress or crisis-related linguistic signals detected by the model.")
    explanation.append("This is a probabilistic AI estimate, not a clinical assessment.")
    return explanation


def assess_risk(
    text: str,
    crisis_predictor: TextProbabilityPredictor,
    distress_predictor: TextProbabilityPredictor,
    sentiment_analyzer: SentimentAnalyzer,
    config: dict,
    country_code: str | None = None,
    resources_path: str | Path = "resources/verified_resources.json",
) -> RiskAssessmentResult:
    """
    country_code: an EXPLICIT, caller-provided code (e.g. "IN", "US") —
    never inferred from `text` itself. Leave as None (the default) when
    there's no reliable location signal, which correctly returns only
    the international directory rather than guessing a country from
    writing style, spelling, or anything else in the text — see
    ml/resources/resource_service.py for the full policy.
    """
    crisis_prob = crisis_predictor.predict_proba(text)
    distress_prob = distress_predictor.predict_proba(text)
    sentiment_result = sentiment_analyzer.analyze(text)

    weights = config["signal_weights"]
    risk_score = weights["crisis"] * crisis_prob + weights["distress"] * distress_prob
    risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

    risk_level = _risk_level_from_score(risk_score, config["thresholds"])

    category_threshold = config["category_detection_threshold"]
    detected_categories = []
    if crisis_prob >= category_threshold:
        detected_categories.append("suicide_indicator")
    if distress_prob >= category_threshold:
        detected_categories.append("emotional_distress")

    explanation = _build_explanation(crisis_prob, distress_prob, category_threshold)

    safety_cfg = config["safety_messages"][risk_level]
    recommendation = {"type": safety_cfg["type"], "message": safety_cfg["message"]}

    # Heuristic confidence proxy: how far each signal sits from maximum
    # uncertainty (p=0.5), combined with the same weights as risk_score.
    # This is NOT a calibrated confidence estimate (e.g. no temperature
    # scaling) — flagged as a Phase 4 limitation in the README, future
    # work rather than implemented here.
    crisis_confidence = 2 * abs(crisis_prob - 0.5)
    distress_confidence = 2 * abs(distress_prob - 0.5)
    confidence = round(weights["crisis"] * crisis_confidence + weights["distress"] * distress_confidence, 4)

    resources = get_resources_for_risk_level(risk_level, country_code=country_code, resources_path=resources_path)

    return RiskAssessmentResult(
        risk_level=risk_level,
        risk_score=risk_score,
        confidence=confidence,
        sentiment={"label": sentiment_result.label, "confidence": sentiment_result.confidence},
        detected_categories=detected_categories,
        explanation=explanation,
        recommendation=recommendation,
        disclaimer=config["disclaimer"],
        resources=resources,
    )
