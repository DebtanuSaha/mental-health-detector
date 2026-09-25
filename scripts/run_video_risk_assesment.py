"""
run_video_risk_assessment.py

Full pipeline: video -> audio extraction -> Whisper transcription ->
existing fine-tuned RoBERTa risk assessment -> Integrated Gradients
explanation. Same output shape as Phase 4/5's text-based CLIs, just
sourced from a video instead of typed text.

Usage:
    python scripts/run_video_risk_assessment.py --video clip.mp4
    python scripts/run_video_risk_assessment.py --video clip.mp4 --country IN
    python scripts/run_video_risk_assessment.py --video clip.mp4 --no-explanations
    python scripts/run_video_risk_assessment.py --video clip.mp4 --prefer baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

import yaml  # noqa: E402

from ml.audio.transcription import Transcriber  # noqa: E402
from ml.inference.model_predictor import load_predictor  # noqa: E402
from ml.inference.sentiment import SentimentAnalyzer  # noqa: E402
from ml.risk.risk_scoring import load_risk_config  # noqa: E402
from ml.video.ingestion import VideoIngestionError  # noqa: E402
from ml.video.video_risk_pipeline import NoAudioError, assess_video_risk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def risk_output(result) -> dict:
    """Return the compact risk-assessment JSON contract for CLI consumers."""
    assessment = result.risk_assessment.to_dict()
    return {
        "risk_level": assessment["risk_level"],
        "risk_score": assessment["risk_score"],
        "confidence": assessment["confidence"],
        "sentiment": assessment["sentiment"],
        "detected_categories": assessment["detected_categories"],
        "explanation": assessment["explanation"],
    }


def existing_video_path(value: str) -> Path:
    """Return an absolute path to an existing video file for argparse."""
    video_path = Path(value).expanduser()
    if not video_path.is_file():
        raise argparse.ArgumentTypeError(f"video file not found: {value}")
    return video_path.resolve()


def positive_frame_limit(value: str) -> int:
    try:
        frame_limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max must be a positive integer") from exc
    if frame_limit < 1:
        raise argparse.ArgumentTypeError("--max must be a positive integer")
    return frame_limit


def main() -> int:
    parser = argparse.ArgumentParser(description="Full video risk assessment + explainability pipeline")
    parser.add_argument(
        "--video",
        required=True,
        type=existing_video_path,
        metavar="VIDEO_PATH",
        help="Path to the video file to assess",
    )
    parser.add_argument(
        "--max",
        dest="max_frames",
        type=positive_frame_limit,
        default=None,
        metavar="FRAMES",
        help="Maximum number of sampled video frames to extract",
    )
    parser.add_argument("--risk-config", default="configs/risk_config.yaml")
    parser.add_argument("--video-config", default="configs/video_config.yaml")
    parser.add_argument("--prefer", choices=["roberta", "baseline"], default="roberta")
    parser.add_argument("--country", default=None,
                         help="Explicit country code (e.g. IN, US, GB) — never inferred from the transcript.")
    parser.add_argument("--frame-fps", type=float, default=None, help="Override video config's ingestion.frame_fps (unused by this pipeline yet — reserved for facial expression)")
    parser.add_argument("--no-explanations", action="store_true",
                         help="Skip Integrated Gradients (faster — risk score only, no token attributions).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    risk_config = load_risk_config(repo_root / args.risk_config)
    with open(repo_root / args.video_config, "r", encoding="utf-8") as f:
        video_config = yaml.safe_load(f)
    ing_cfg = video_config["ingestion"]

    crisis_cfg = risk_config["model_paths"]["crisis"]
    distress_cfg = risk_config["model_paths"]["distress"]

    try:
        crisis_predictor = load_predictor(
            repo_root / crisis_cfg["roberta_dir"], repo_root / crisis_cfg["baseline_path"], prefer=args.prefer
        )
        distress_predictor = load_predictor(
            repo_root / distress_cfg["roberta_dir"], repo_root / distress_cfg["baseline_path"], prefer=args.prefer
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    sentiment_analyzer = SentimentAnalyzer(risk_config["sentiment"]["model_name"])
    transcriber = Transcriber()
    resources_path = repo_root / risk_config["resources_path"]

    try:
        result = assess_video_risk(
            args.video,
            crisis_predictor, distress_predictor, sentiment_analyzer, transcriber, risk_config,
            frame_fps=args.frame_fps or ing_cfg["frame_fps"],
            audio_sample_rate=ing_cfg["audio_sample_rate"],
            max_duration_seconds=ing_cfg["max_duration_seconds"],
            max_frames=args.max_frames,
            country_code=args.country,
            resources_path=resources_path,
            include_explanations=not args.no_explanations,
        )
    except (VideoIngestionError, NoAudioError) as e:
        logger.error(str(e))
        return 1

    print(json.dumps(risk_output(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
