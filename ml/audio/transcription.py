"""
transcription.py

Off-the-shelf pretrained speech-to-text via OpenAI's Whisper — no
fine-tuning, no local training of any kind. This is how audio reaches
the existing fine-tuned RoBERTa crisis/distress classifiers: transcribe
to plain text, then hand that text to the SAME model_predictor.py /
risk_scoring.py / explainer.py code already used for direct text
input, completely unchanged.

Lazy-loaded (only downloads/loads the model on first real .transcribe()
call), same pattern as ml/inference/sentiment.py, so constructing a
Transcriber in tests doesn't require network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "openai/whisper-base"


@dataclass
class TranscriptionResult:
    text: str


class Transcriber:
    def __init__(self, model_name: str = DEFAULT_ASR_MODEL):
        self.model_name = model_name
        self._pipeline = None  # lazy-loaded on first transcribe() call

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        try:
            self._pipeline = hf_pipeline("automatic-speech-recognition", model=self.model_name)
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

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        self._ensure_loaded()
        # Whisper requires timestamps for long-form audio (>30 seconds).
        # They are accepted here while the pipeline still returns plain text.
        result = self._pipeline(str(audio_path), return_timestamps=True)
        text = str(result.get("text", "")).strip()
        return TranscriptionResult(text=text)
