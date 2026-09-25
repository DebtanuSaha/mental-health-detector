"""
ingestion.py

Splits an input video into (1) an audio track suitable for ASR
(Whisper expects 16kHz mono WAV) and (2) a sampled sequence of frames
suitable for facial expression analysis — the shared infrastructure
both the speech->text->RoBERTa path and the facial-expression path
build on.

Privacy note (per the project's established stance — no raw text
stored by default, extended here to video/frame data, which is
biometric and a bigger privacy step up than text): extracted audio and
frames are written to a temporary directory by default and NOT kept
around after use. Call cleanup_ingestion() (or use ingest_video() as a
context manager) to delete them as soon as you're done — don't let
extracted frames/audio linger on disk longer than the request needs.

Requires the `ffmpeg` and `ffprobe` binaries on PATH (not a pip
package — a system dependency, checked explicitly with a clear error
if missing rather than a cryptic subprocess failure).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoIngestionError(Exception):
    """Raised for missing ffmpeg, missing/corrupt video files, or extraction failures."""


@dataclass
class VideoIngestionResult:
    video_path: Path
    audio_path: Path | None       # None if the video has no audio stream
    frame_paths: list             # list[Path], in chronological order
    duration_seconds: float
    frame_fps_used: float
    _temp_dir: Path = field(repr=False, default=None)


def check_ffmpeg_available() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise VideoIngestionError(
                f"'{binary}' not found on PATH. Install ffmpeg (includes ffprobe) — "
                f"e.g. https://ffmpeg.org/download.html — and make sure it's on your PATH, "
                f"then re-run. This is a system dependency, not a pip package."
            )


def probe_video(video_path: str | Path) -> dict:
    """Returns {duration_seconds, has_audio, width, height} via ffprobe."""
    check_ffmpeg_available()
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoIngestionError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoIngestionError(
            f"ffprobe failed on {video_path} (is it a valid video file?): {result.stderr.strip()}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VideoIngestionError(f"ffprobe returned unparseable output for {video_path}: {e}") from e

    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    duration = float(info.get("format", {}).get("duration", 0.0))
    width = int(video_stream["width"]) if video_stream and "width" in video_stream else None
    height = int(video_stream["height"]) if video_stream and "height" in video_stream else None

    return {"duration_seconds": duration, "has_audio": has_audio, "width": width, "height": height}


def _extract_audio(video_path: Path, output_path: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoIngestionError(f"ffmpeg audio extraction failed for {video_path}: {result.stderr.strip()}")


def _extract_frames(video_path: Path, output_dir: Path, fps: float, max_frames: int | None = None) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%05d.jpg"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}"]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(str(pattern))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoIngestionError(f"ffmpeg frame extraction failed for {video_path}: {result.stderr.strip()}")
    return sorted(output_dir.glob("frame_*.jpg"))


def ingest_video(
    video_path: str | Path,
    frame_fps: float = 1.0,
    audio_sample_rate: int = 16000,
    max_duration_seconds: float | None = 600.0,
    max_frames: int | None = None,
) -> VideoIngestionResult:
    """
    Extracts audio (16kHz mono WAV, ready for Whisper) and sampled
    frames (JPEG, ready for face detection) into a fresh temp
    directory. Call cleanup_ingestion(result) — or use ingest_video_ctx()
    as a context manager — to delete them once you're done; nothing
    here persists on its own.

    max_duration_seconds: a safety cap (default 10 minutes) against
    accidentally processing a very long video — raise this explicitly
    if you have a real use case for longer input, don't just remove it.

    max_frames: optional cap on the number of sampled video frames
    extracted for downstream processing.
    """
    check_ffmpeg_available()
    if max_frames is not None and max_frames < 1:
        raise VideoIngestionError("max_frames must be at least 1")
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoIngestionError(f"Video file not found: {video_path}")

    info = probe_video(video_path)
    if max_duration_seconds is not None and info["duration_seconds"] > max_duration_seconds:
        raise VideoIngestionError(
            f"Video is {info['duration_seconds']:.1f}s, exceeds max_duration_seconds="
            f"{max_duration_seconds}. Pass a higher explicit limit if this is expected."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="video_ingest_"))
    logger.info("Ingesting %s into temp dir %s", video_path, temp_dir)

    audio_path = None
    if info["has_audio"]:
        audio_path = temp_dir / "audio.wav"
        _extract_audio(video_path, audio_path, audio_sample_rate)
    else:
        logger.warning("Video %s has no audio stream — audio/transcript signal will be unavailable.", video_path)

    frames_dir = temp_dir / "frames"
    frame_paths = _extract_frames(video_path, frames_dir, frame_fps, max_frames=max_frames)
    logger.info("Extracted %d frames at %s fps, audio=%s", len(frame_paths), frame_fps, audio_path is not None)

    return VideoIngestionResult(
        video_path=video_path,
        audio_path=audio_path,
        frame_paths=frame_paths,
        duration_seconds=info["duration_seconds"],
        frame_fps_used=frame_fps,
        _temp_dir=temp_dir,
    )


def cleanup_ingestion(result: VideoIngestionResult) -> None:
    """Deletes the temp directory (audio + frames) created by ingest_video()."""
    if result._temp_dir and result._temp_dir.exists():
        shutil.rmtree(result._temp_dir)
        logger.info("Cleaned up temp ingestion dir %s", result._temp_dir)


@contextmanager
def ingest_video_ctx(*args, **kwargs):
    """Context-manager wrapper: guarantees cleanup even if processing raises."""
    result = ingest_video(*args, **kwargs)
    try:
        yield result
    finally:
        cleanup_ingestion(result)
