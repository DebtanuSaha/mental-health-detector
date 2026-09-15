"""
explainer.py

Unified explainability entry point: picks the right explanation method
based on which kind of predictor produced the crisis/distress
probability (ml/inference/model_predictor.py) — Integrated Gradients
for a fine-tuned RoBERTa model, linear-coefficient attribution for the
Phase 2 baseline fallback. Mirrors model_predictor.py's own
roberta-with-baseline-fallback design, so explainability degrades
gracefully the same way prediction already does.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.explainability.baseline_explainer import explain_with_linear_coefficients
from ml.explainability.integrated_gradients import explain_with_integrated_gradients
from ml.inference.model_predictor import BaselinePredictor, RobertaPredictor, TextProbabilityPredictor


@dataclass
class ExplanationResult:
    method: str  # "integrated_gradients" | "linear_coefficients"
    result: object  # IntegratedGradientsResult | BaselineExplanationResult

    def to_dict(self) -> dict:
        if self.method == "integrated_gradients":
            r = self.result
            return {
                "method": self.method,
                "predicted_proba": r.predicted_proba,
                "token_attributions": [{"token": t.token, "score": t.score} for t in r.token_attributions],
                "caveats": r.caveats,
            }
        r = self.result
        return {
            "method": self.method,
            "predicted_proba": r.predicted_proba,
            "top_positive_features": [{"feature": f.feature, "score": f.score} for f in r.top_positive_features],
            "top_negative_features": [{"feature": f.feature, "score": f.score} for f in r.top_negative_features],
            "caveats": r.caveats,
        }


def explain(text: str, predictor: TextProbabilityPredictor, target_class: int = 1) -> ExplanationResult:
    if isinstance(predictor, RobertaPredictor):
        result = explain_with_integrated_gradients(
            text, predictor.tokenizer, predictor.model, target_class=target_class
        )
        return ExplanationResult(method="integrated_gradients", result=result)
    if isinstance(predictor, BaselinePredictor):
        result = explain_with_linear_coefficients(text, predictor.pipeline)
        return ExplanationResult(method="linear_coefficients", result=result)
    raise TypeError(f"No explainability method available for predictor type {type(predictor).__name__}")
