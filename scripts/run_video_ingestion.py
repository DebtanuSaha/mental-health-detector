"""
run_video_ingestion.py

Extracts audio + sampled frames from a video and reports a summary.
By default cleans everything up afterward (per the project's privacy
stance) — pass --keep to inspect the extracted files instead.

Usage:
    python scripts/run_video_ingestion.py --video path/to/clip.mp4
    python scripts/run_video_ingestion.py --video path/to/clip.mp4 --frame-fps 2 --keep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

import yaml  # noqa: E402

from ml.video.ingestion import VideoIngestionError, cleanup_ingestion, ingest_video  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Video ingestion: extract audio + sampled frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/video_config.yaml")
    parser.add_argument("--frame-fps", type=float, default=None, help="Override config's ingestion.frame_fps")
    parser.add_argument("--keep", action="store_true", help="Don't delete extracted audio/frames afterward (inspection only).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ing_cfg = config["ingestion"]

    frame_fps = args.frame_fps or ing_cfg["frame_fps"]

    try:
        result = ingest_video(
            args.video,
            frame_fps=frame_fps,
            audio_sample_rate=ing_cfg["audio_sample_rate"],
            max_duration_seconds=ing_cfg["max_duration_seconds"],
        )
    except VideoIngestionError as e:
        logger.error(str(e))
        return 1

    logger.info("Duration: %.1fs", result.duration_seconds)
    logger.info("Audio: %s", result.audio_path if result.audio_path else "(no audio stream)")
    logger.info("Frames: %d at %s fps", len(result.frame_paths), result.frame_fps_used)

    if args.keep:
        logger.info("--keep passed: NOT cleaning up. Files at %s", result._temp_dir)
    else:
        cleanup_ingestion(result)
        logger.info("Cleaned up extracted audio/frames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
