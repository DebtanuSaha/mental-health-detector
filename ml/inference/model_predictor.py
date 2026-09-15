"""
model_predictor.py

Generic uniform interface for getting a crisis/distress probability from
either a Phase 2 baseline (sklearn Pipeline, .joblib) or a Phase 3
fine-tuned RoBERTa checkpoint (HF from_pretrained directory) — Phase 4
doesn't care which one produced the signal, only that it returns a
probability in [0, 1].

Falls back to the baseline automatically if a RoBERTa checkpoint isn't
present yet (e.g. Phase 3 hasn't finished training) — logged clearly,
not silent — so Phase 4 is usable incrementally rather than blocked on
Phase 3 being fully done.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class TextProbabilityPredictor(Protocol):
    def predict_proba(self, text: str) -> float: ...


class BaselinePredictor:
    """Wraps a saved Phase 2 sklearn Pipeline (TF-IDF + LogisticRegression)."""

    def __init__(self, pipeline_path: str | Path):
        from ml.models.baseline import load_pipeline
        self.pipeline = load_pipeline(pipeline_path)

    def predict_proba(self, text: str) -> float:
        return float(self.pipeline.predict_proba([text])[0][1])


class RobertaPredictor:
    """Wraps a saved Phase 3 fine-tuned RoBERTa checkpoint."""

    def __init__(self, model_dir: str | Path, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.max_length = max_length
        self._torch = torch

    def predict_proba(self, text: str) -> float:
        torch = self._torch
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1])


def load_predictor(
    roberta_dir: str | Path,
    baseline_path: str | Path,
    prefer: str = "roberta",
) -> TextProbabilityPredictor:
    """
    Load whichever real trained model is available for a given signal
    (crisis or distress). prefer="roberta" tries the fine-tuned
    checkpoint first and falls back to the Phase 2 baseline if that
    directory doesn't exist yet — with a clear log message, never
    silently. prefer="baseline" skips straight to the baseline, useful
    for testing the risk-scoring pipeline without waiting on Phase 3.
    """
    roberta_dir = Path(roberta_dir)
    baseline_path = Path(baseline_path)

    if prefer == "roberta" and roberta_dir.exists() and (roberta_dir / "config.json").exists():
        logger.info("Using fine-tuned RoBERTa model at %s", roberta_dir)
        return RobertaPredictor(roberta_dir)

    if baseline_path.exists():
        if prefer == "roberta":
            logger.warning(
                "RoBERTa checkpoint not found at %s — falling back to Phase 2 baseline at %s. "
                "Fine for testing the risk-scoring pipeline, but real fine-tuned results should "
                "replace this once Phase 3 training completes.",
                roberta_dir, baseline_path,
            )
        else:
            logger.info("Using Phase 2 baseline at %s (prefer='baseline')", baseline_path)
        return BaselinePredictor(baseline_path)

    raise FileNotFoundError(
        f"No trained model found — checked RoBERTa dir {roberta_dir} and baseline {baseline_path}. "
        f"Run Phase 2's scripts/run_phase2_baseline.py (fast) or Phase 3's "
        f"scripts/run_phase3_finetune.py (real fine-tune) first."
    )
