"""
baseline_explainer.py

Lightweight explanation for the Phase 2 TF-IDF + Logistic Regression
baseline predictor — used automatically as the explainability fallback
whenever ml/inference/model_predictor.py falls back to the baseline
model (e.g. before Phase 3 fine-tuning is done). NOT Integrated
Gradients — that method requires a differentiable neural network and
has no meaning for a linear bag-of-words model. Instead, this computes
the standard, well-established explanation for a linear model: each
active n-gram feature's contribution = its TF-IDF weight in this text
times the logistic regression's learned coefficient for that feature.
"""

from __future__ import annotations

from dataclasses import dataclass

BASELINE_CAVEATS = [
    "Feature contributions are computed directly from the linear model's learned coefficients times each n-gram's TF-IDF weight in this text.",
    "This reflects the linear baseline model's own feature weighting, not the deeper contextual reasoning a neural network like the fine-tuned RoBERTa model might use — it is a different (simpler) explanation for a different (simpler) model.",
    "This does NOT prove the clinical significance of any word or phrase.",
    "Computationally very cheap — no extra backward pass is needed, unlike Integrated Gradients.",
]


@dataclass
class FeatureAttribution:
    feature: str
    score: float


@dataclass
class BaselineExplanationResult:
    predicted_proba: float
    top_positive_features: list
    top_negative_features: list
    caveats: list


def explain_with_linear_coefficients(text: str, pipeline, top_k: int = 10) -> BaselineExplanationResult:
    """
    pipeline: a fitted sklearn Pipeline([("tfidf", TfidfVectorizer), ("clf", LogisticRegression)])
    as saved/loaded by ml/models/baseline.py.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    tfidf_vector = vectorizer.transform([text]).tocoo()  # sparse 1 x n_features
    coefficients = classifier.coef_[0]  # binary LogisticRegression -> one coefficient vector
    feature_names = vectorizer.get_feature_names_out()

    contributions = {}
    for feature_idx, tfidf_value in zip(tfidf_vector.col, tfidf_vector.data):
        contributions[feature_names[feature_idx]] = float(tfidf_value) * float(coefficients[feature_idx])

    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    top_positive = [FeatureAttribution(f, round(s, 4)) for f, s in ranked if s > 0][:top_k]
    top_negative = [FeatureAttribution(f, round(s, 4)) for f, s in reversed(ranked) if s < 0][:top_k]

    proba = float(pipeline.predict_proba([text])[0][1])

    return BaselineExplanationResult(
        predicted_proba=round(proba, 4),
        top_positive_features=top_positive,
        top_negative_features=top_negative,
        caveats=BASELINE_CAVEATS,
    )
