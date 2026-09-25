"""
video_risk_pipeline.py

Full video -> risk assessment + explainability pipeline:

    video file
      -> ingest_video()            (ml/video/ingestion.py — audio + frames)
      -> Transcriber.transcribe()  (ml/audio/transcription.py — Whisper, pretrained, no fine-tune)
      -> assess_risk()             (ml/risk/risk_scoring.py — SAME code Phase 4 uses for text)
      -> explain()                 (ml/explainability/explainer.py — SAME code Phase 5 uses for text)
      -> cleanup_ingestion()       (delete extracted audio/frames — nothing persists)

Deliberately a thin orchestration layer: no new risk-scoring or
explainability logic lives here. A transcript is just text, and
everything downstream of transcription is the exact Phase 4/5 code
path already used for direct text input — this file's only job is
turning a video into that text and then calling the existing pipeline.

Facial-expression analysis is NOT included yet (a later build stage) —
this wires only the audio -> transcript -> RoBERTa path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ml.audio.transcription import Transcriber
from ml.explainability.explainer import ExplanationResult, explain
from ml.inference.model_predictor import TextProbabilityPredictor
from ml.inference.sentiment import SentimentAnalyzer
from ml.risk.risk_scoring import RiskAssessmentResult, assess_risk
from ml.video.ingestion import cleanup_ingestion, ingest_video

logger = logging.getLogger(__name__)


class NoAudioError(Exception):
    """Raised when the video has no audio track — nothing to transcribe or assess."""


@dataclass
class VideoRiskAssessmentResult:
    transcript: str
    duration_seconds: float
    risk_assessment: RiskAssessmentResult
    crisis_explanation: ExplanationResult | None
    distress_explanation: ExplanationResult | None

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "duration_seconds": self.duration_seconds,
            "risk_assessment": self.risk_assessment.to_dict(),
            "crisis_explanation": self.crisis_explanation.to_dict() if self.crisis_explanation else None,
            "distress_explanation": self.distress_explanation.to_dict() if self.distress_explanation else None,
        }


def assess_video_risk(
    video_path: str | Path,
    crisis_predictor: TextProbabilityPredictor,
    distress_predictor: TextProbabilityPredictor,
    sentiment_analyzer: SentimentAnalyzer,
    transcriber: Transcriber,
    risk_config: dict,
    frame_fps: float = 1.0,
    audio_sample_rate: int = 16000,
    max_duration_seconds: float | None = 600.0,
    max_frames: int | None = None,
    country_code: str | None = None,
    resources_path: str | Path = "res/verified_resources.json",
    include_explanations: bool = True,
) -> VideoRiskAssessmentResult:
    """
    include_explanations: Integrated Gradients (used when a RoBERTa
    model is active) costs an extra backward pass beyond the base
    prediction — pass False to skip explanation generation for a
    faster risk-only result, same cost trade-off already documented in
    Phase 5.

    country_code: same policy as the text pipeline — never inferred,
    only used if you explicitly pass one.
    """
    ingestion_result = ingest_video(
        video_path, frame_fps=frame_fps, audio_sample_rate=audio_sample_rate,
        max_duration_seconds=max_duration_seconds, max_frames=max_frames,
    )
    try:
        if ingestion_result.audio_path is None:
            raise NoAudioError(
                f"Video {video_path} has no audio track — nothing to transcribe. "
                f"Facial-expression-only analysis isn't available yet (a later build stage)."
            )

        transcription = transcriber.transcribe(ingestion_result.audio_path)
        preview = transcription.text[:200] + ("…" if len(transcription.text) > 200 else "")
        logger.info("Transcript (%d chars): %s", len(transcription.text), preview)

        if not transcription.text.strip():
            logger.warning("Transcription produced empty text — risk assessment will reflect empty input.")

        risk_result = assess_risk(
            transcription.text, crisis_predictor, distress_predictor, sentiment_analyzer, risk_config,
            country_code=country_code, resources_path=resources_path,
        )

        crisis_expl = distress_expl = None
        if include_explanations:
            crisis_expl = explain(transcription.text, crisis_predictor)
            distress_expl = explain(transcription.text, distress_predictor)

        return VideoRiskAssessmentResult(
            transcript=transcription.text,
            duration_seconds=ingestion_result.duration_seconds,
            risk_assessment=risk_result,
            crisis_explanation=crisis_expl,
            distress_explanation=distress_expl,
        )
    finally:
        cleanup_ingestion(ingestion_result)
