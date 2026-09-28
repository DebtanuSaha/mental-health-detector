"""
emotion_analyzer.py

Facial emotion analysis via DeepFace — an off-the-shelf pretrained
model, no local fine-tuning of any kind. Given a single video frame,
detects the largest/closest face (per the project's explicit design
choice when multiple people appear) and returns its emotion scores.

Verified empirically against the installed DeepFace version (not
assumed from memory): DeepFace.analyze() with enforce_detection=False
still returns a result even when NO real face is present in the frame
— it silently falls back to treating the whole frame as the "face
region" and reports face_confidence=0.0. This means face_confidence,
not the mere presence of a result, is what actually tells you whether
a person was detected; min_face_confidence below filters this out, so
the system never reports a fabricated emotion for an empty frame.

Per the project's established stance on facial expression (identical
caution already applied to sentiment analysis): this is a secondary,
informational signal, reported alongside the text/audio-driven risk
assessment but never fused into its numeric risk_score. Facial
expression recognition models are trained on posed/acted datasets
(FER2013-style) and have documented reliability and bias limitations —
treat any reading as suggestive, not diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

EMOTION_CATEGORIES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

FACIAL_EMOTION_CAVEATS = [
    "Facial expression models are trained on posed/acted datasets and have documented reliability and bias limitations — a detected expression is not a reliable indicator of someone's actual internal state.",
    "This is a secondary, informational signal, exactly like the sentiment signal — it does not change the numeric crisis/distress risk_score.",
    "Lighting, camera angle, distance, and partial face visibility all affect detection quality.",
]


@dataclass
class FaceEmotionReading:
    dominant_emotion: str
    emotion_scores: dict   # {category: 0-100 score}, keys from EMOTION_CATEGORIES
    bbox: tuple            # (x, y, w, h) in frame pixel coordinates
    face_confidence: float


def analyze_frame(
    frame,
    detector_backend: str = "opencv",
    min_face_confidence: float = 0.5,
):
    """
    frame: a BGR numpy array (OpenCV's native frame format).

    Returns None if no face meeting min_face_confidence was found in
    this frame — callers must treat None as "nobody in frame right
    now", not as an error condition.

    If multiple faces are detected, only the largest (by bounding-box
    area) is analyzed and returned — the project's explicit choice for
    handling multiple people in frame.
    """
    from deepface import DeepFace

    try:
        results = DeepFace.analyze(
            img_path=frame, actions=["emotion"], enforce_detection=False, detector_backend=detector_backend,
        )
    except Exception:
        # DeepFace can raise on some malformed/edge-case frames (e.g.
        # a corrupt or all-black frame during camera warm-up) — treat
        # as "no reading this attempt" rather than crashing the
        # capture loop over a single bad frame.
        return None

    valid = [r for r in results if r.get("face_confidence", 0.0) >= min_face_confidence]
    if not valid:
        return None

    largest = max(valid, key=lambda r: r["region"]["w"] * r["region"]["h"])
    region = largest["region"]
    scores = {k: float(v) for k, v in largest["emotion"].items()}

    return FaceEmotionReading(
        dominant_emotion=largest["dominant_emotion"],
        emotion_scores=scores,
        bbox=(region["x"], region["y"], region["w"], region["h"]),
        face_confidence=float(largest["face_confidence"]),
    )
