"""
baseline.py

TF-IDF + Logistic Regression baseline classifier — Experiment 1 in the
project brief's research-experiments design (§16), used as the point of
comparison for the RoBERTa fine-tune in Phase 3.

Kept as a thin wrapper around a single sklearn Pipeline so vectorizer +
classifier are always saved/loaded together (avoids the classic bug of
loading a model with a vectorizer fit on different vocabulary).
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_baseline_pipeline(tfidf_cfg: dict, logreg_cfg: dict) -> Pipeline:
    """Construct an untrained TF-IDF + LogisticRegression pipeline from config dicts."""
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 20000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 0.95),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
    )
    classifier = LogisticRegression(
        C=logreg_cfg.get("C", 1.0),
        max_iter=logreg_cfg.get("max_iter", 1000),
        class_weight=logreg_cfg.get("class_weight", "balanced"),
        solver=logreg_cfg.get("solver", "liblinear"),
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def save_pipeline(pipeline: Pipeline, output_dir: str | Path, filename: str = "baseline_pipeline.joblib") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved baseline pipeline at {path}")
    return joblib.load(path)
