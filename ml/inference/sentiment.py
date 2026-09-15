"""
sentiment.py

Off-the-shelf pretrained sentiment signal, called at inference time —
NOT jointly trained with the crisis/distress classifiers. Per the
Phase 0/3 architecture decision, sentiment is a secondary signal only
and must never be the primary crisis detector (brief §7): negative
sentiment != suicide risk, positive sentiment != absence of risk.

Lazy-loaded (only downloads/loads the model on first real .analyze()
call) so constructing a SentimentAnalyzer in tests doesn't require
network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass
class SentimentResult:
    label: str        # "negative" | "neutral" | "positive"
    confidence: float


class SentimentAnalyzer:
    def __init__(self, model_name: str = DEFAULT_SENTIMENT_MODEL):
        self.model_name = model_name
        self._pipeline = None  # lazy-loaded on first analyze() call

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        try:
            self._pipeline = hf_pipeline("sentiment-analysis", model=self.model_name)
        except OSError as e:
            if "gated repo" in str(e).lower() or "401" in str(e):
                raise OSError(
                    f"'{self.model_name}' is a gated model on the Hugging Face Hub. To fix:\n"
                    f"  1. Accept its terms at https://huggingface.co/{self.model_name} while logged in.\n"
                    f"  2. Create a Read-scope token: Settings -> Access Tokens -> New token.\n"
                    f"  3. Run `hf auth login` locally and paste the token.\n"
                    f"  4. Re-run.\n"
                    f"(Original error: {e})"
                ) from e
            raise

    def analyze(self, text: str) -> SentimentResult:
        self._ensure_loaded()
        result = self._pipeline(text, truncation=True)[0]
        return SentimentResult(label=result["label"].lower(), confidence=round(float(result["score"]), 4))
