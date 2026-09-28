"""
run_live_risk_assessment.py

Opens a live webcam + microphone preview window. Press Q (or your
configured quit key) to stop recording, or it auto-stops at
max-duration. The recording is then run through the SAME
video -> transcript -> RoBERTa risk assessment + explainability
pipeline used for uploaded video files
(ml/video/video_risk_pipeline.py) — analysis doesn't differ based on
how the video was obtained.

Usage:
    python scripts/run_live_risk_assessment.py
    python scripts/run_live_risk_assessment.py --max-duration 60 --country IN
    python scripts/run_live_risk_assessment.py --camera-index 1 --mic-index 2
    python scripts/run_live_risk_assessment.py --no-explanations
    python scripts/run_live_risk_assessment.py --no-facial
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
from ml.video.live_capture import LiveCaptureConfig, LiveCaptureError, cleanup_live_capture, capture_live_av  # noqa: E402
from ml.video.video_risk_pipeline import NoAudioError, assess_video_risk  # noqa: E402
from ml.video.ingestion import VideoIngestionError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live capture + full risk assessment pipeline")
    parser.add_argument("--risk-config", default="configs/risk_config.yaml")
    parser.add_argument("--live-config", default="configs/live_capture_config.yaml")
    parser.add_argument("--facial-config", default="configs/facial_config.yaml")
    parser.add_argument("--prefer", choices=["roberta", "baseline"], default="roberta")
    parser.add_argument("--country", default=None,
                         help="Explicit country code (e.g. IN, US, GB) — never inferred from the transcript.")
    parser.add_argument("--camera-index", type=int, default=None, help="Override live_capture.camera_index")
    parser.add_argument("--mic-index", type=int, default=None, help="Override live_capture.mic_device_index")
    parser.add_argument("--max-duration", type=float, default=None, help="Override live_capture.max_duration_seconds")
    parser.add_argument("--emotion-interval", type=float, default=None,
                         help="Override facial_emotion.analysis_interval_seconds")
    parser.add_argument("--no-explanations", action="store_true",
                         help="Skip Integrated Gradients (faster — risk score only, no token attributions).")
    parser.add_argument("--no-facial", action="store_true",
                         help="Disable live facial emotion analysis entirely.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    risk_config = load_risk_config(repo_root / args.risk_config)
    with open(repo_root / args.live_config, "r", encoding="utf-8") as f:
        live_cfg_dict = yaml.safe_load(f)["live_capture"]
    with open(repo_root / args.facial_config, "r", encoding="utf-8") as f:
        facial_cfg_dict = yaml.safe_load(f)["facial_emotion"]

    live_config = LiveCaptureConfig(
        camera_index=args.camera_index if args.camera_index is not None else live_cfg_dict["camera_index"],
        mic_device_index=args.mic_index if args.mic_index is not None else live_cfg_dict["mic_device_index"],
        fps=live_cfg_dict["fps"],
        frame_width=live_cfg_dict["frame_width"],
        frame_height=live_cfg_dict["frame_height"],
        audio_sample_rate=live_cfg_dict["audio_sample_rate"],
        audio_channels=live_cfg_dict["audio_channels"],
        audio_chunk_size=live_cfg_dict["audio_chunk_size"],
        max_duration_seconds=args.max_duration if args.max_duration is not None else live_cfg_dict["max_duration_seconds"],
        window_title=live_cfg_dict["window_title"],
        quit_key=live_cfg_dict["quit_key"],
        enable_emotion_analysis=not args.no_facial and facial_cfg_dict["enable"],
        emotion_detector_backend=facial_cfg_dict["detector_backend"],
        emotion_analysis_interval_seconds=(
            args.emotion_interval if args.emotion_interval is not None else facial_cfg_dict["analysis_interval_seconds"]
        ),
        min_face_confidence=facial_cfg_dict["min_face_confidence"],
    )

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

    logger.info("Opening camera/mic — press '%s' to stop recording (auto-stops at %.0fs).",
                live_config.quit_key, live_config.max_duration_seconds)
    try:
        capture_result = capture_live_av(live_config)
    except LiveCaptureError as e:
        logger.error(str(e))
        return 1

    logger.info("Recorded %.1fs — running risk assessment...", capture_result.duration_seconds)
    try:
        result = assess_video_risk(
            capture_result.video_path,
            crisis_predictor, distress_predictor, sentiment_analyzer, transcriber, risk_config,
            audio_sample_rate=live_config.audio_sample_rate,
            max_duration_seconds=None,  # already bounded by max_duration_seconds during capture itself
            country_code=args.country,
            resources_path=resources_path,
            include_explanations=not args.no_explanations,
        )
    except (VideoIngestionError, NoAudioError) as e:
        logger.error(str(e))
        return 1
    finally:
        cleanup_live_capture(capture_result)

    # Facial emotion is reported alongside the risk assessment (same
    # treatment as sentiment: informational, not fused into
    # risk_score) — attached here since video_risk_pipeline.py stays
    # agnostic to whether the video came from a file or live capture.
    final_output = result.to_dict()
    final_output["facial_emotion"] = capture_result.facial_emotion.to_dict() if capture_result.facial_emotion else None

    print(json.dumps(final_output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
