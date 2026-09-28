"""
emotion_aggregator.py

Aggregates a sequence of per-frame FaceEmotionReading objects
(collected live throughout a recording session) into one summary
result: the overall dominant emotion, what fraction of analyzed frames
each emotion was dominant in, and the average score per category
across the whole session.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ml.facial.emotion_analyzer import EMOTION_CATEGORIES, FACIAL_EMOTION_CAVEATS


@dataclass
class FacialEmotionResult:
    dominant_emotion: str  # None if no face was ever detected during the session
    emotion_distribution: dict   # {category: % of analyzed frames where this was dominant}
    average_scores: dict         # {category: mean 0-100 score across analyzed frames}
    n_readings: int              # attempts where a face WAS detected
    n_frames_attempted: int      # total analysis attempts during the session (readings + misses)
    caveats: list

    def to_dict(self) -> dict:
        return {
            "dominant_emotion": self.dominant_emotion,
            "emotion_distribution": self.emotion_distribution,
            "average_scores": self.average_scores,
            "n_readings": self.n_readings,
            "n_frames_attempted": self.n_frames_attempted,
            "caveats": self.caveats,
        }


def aggregate_readings(readings: list, n_frames_attempted: int) -> FacialEmotionResult:
    """
    readings: list of FaceEmotionReading | None — one entry per
    analysis attempt during the session; None entries mean no face was
    detected at that particular attempt (still counted toward
    n_frames_attempted, since "how often was nobody visible" is itself
    useful context).
    """
    valid = [r for r in readings if r is not None]

    if not valid:
        return FacialEmotionResult(
            dominant_emotion=None,
            emotion_distribution={},
            average_scores={},
            n_readings=0,
            n_frames_attempted=n_frames_attempted,
            caveats=FACIAL_EMOTION_CAVEATS,
        )

    dominant_counts = Counter(r.dominant_emotion for r in valid)
    n = len(valid)
    emotion_distribution = {emotion: round(100 * count / n, 2) for emotion, count in dominant_counts.items()}
    overall_dominant = dominant_counts.most_common(1)[0][0]

    average_scores = {}
    for category in EMOTION_CATEGORIES:
        values = [r.emotion_scores.get(category, 0.0) for r in valid]
        average_scores[category] = round(sum(values) / len(values), 2)

    return FacialEmotionResult(
        dominant_emotion=overall_dominant,
        emotion_distribution=emotion_distribution,
        average_scores=average_scores,
        n_readings=n,
        n_frames_attempted=n_frames_attempted,
        caveats=FACIAL_EMOTION_CAVEATS,
    )
