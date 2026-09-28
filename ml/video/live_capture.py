"""
live_capture.py

Live webcam + microphone capture: opens a preview window (OpenCV) and
records simultaneously from the camera and default microphone
(PyAudio) until the user presses the quit key (default 'q') or
max_duration_seconds elapses. The captured frames + audio are muxed
into a single temporary MP4 via ffmpeg — the exact format
ml/video/ingestion.py and ml/video/video_risk_pipeline.py already
expect — so a live recording feeds straight into the existing
assess_video_risk() pipeline unchanged, exactly like an uploaded video
file would. No new risk-scoring or explainability logic lives here.

While recording, this also runs live facial emotion analysis
(ml/facial/emotion_analyzer.py, DeepFace) at a configurable interval,
draws a bounding box + emotion label overlay on the PREVIEW only (the
frames actually written to the output video stay clean, unmodified),
and aggregates the readings across the whole session into a
FacialEmotionResult attached to LiveCaptureResult.facial_emotion.

Several of the building blocks below (write_frames_to_video,
write_audio_to_wav, mux_audio_video, _draw_overlay) are pure functions
that don't need a real camera or microphone and are fully unit-tested
with synthetic data. Only capture_live_av() itself opens real hardware
(cv2.VideoCapture + a PyAudio input stream) and cannot be exercised in
an environment with no camera/mic attached — verify that part on your
own machine.

Privacy: exactly like video ingestion, the recording is written to a
temp directory and must be deleted via cleanup_live_capture() (or the
live_capture_ctx() context manager) as soon as processing is done —
nothing here persists on its own.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ml.facial.emotion_aggregator import FacialEmotionResult, aggregate_readings
from ml.facial.emotion_analyzer import analyze_frame

logger = logging.getLogger(__name__)


class LiveCaptureError(Exception):
    """Raised for missing camera/mic/dependencies, or capture/muxing failures."""


@dataclass
class LiveCaptureConfig:
    camera_index: int = 0
    mic_device_index: int | None = None  # None = system default input device
    fps: int = 20
    frame_width: int = 640
    frame_height: int = 480
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_size: int = 1024
    max_duration_seconds: float = 120.0
    window_title: str = "Recording - press Q to stop"
    quit_key: str = "q"
    # Facial emotion analysis (see configs/facial_config.yaml)
    enable_emotion_analysis: bool = True
    emotion_detector_backend: str = "opencv"
    emotion_analysis_interval_seconds: float = 1.0
    min_face_confidence: float = 0.5


@dataclass
class LiveCaptureResult:
    video_path: Path
    duration_seconds: float
    facial_emotion: FacialEmotionResult | None = None
    _temp_dir: Path = field(repr=False, default=None)


# --- Pure helpers: no camera/mic required, fully unit-testable ---

def write_frames_to_video(frames: list, output_path: str | Path, fps: float, frame_size: tuple) -> None:
    """frames: list of BGR numpy arrays (OpenCV's native frame format)."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise LiveCaptureError(f"Could not open VideoWriter for {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def write_audio_to_wav(
    audio_chunks: list, output_path: str | Path, sample_rate: int, channels: int, sample_width: int = 2
) -> None:
    """audio_chunks: list of raw PCM bytes (as returned by a PyAudio stream.read())."""
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(audio_chunks))


def mux_audio_video(video_only_path: str | Path, audio_wav_path: str | Path, output_path: str | Path) -> None:
    """Combines a video-only file and a WAV into one MP4 via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise LiveCaptureError("'ffmpeg' not found on PATH — required to combine captured video and audio.")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only_path), "-i", str(audio_wav_path),
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LiveCaptureError(f"ffmpeg muxing failed: {result.stderr.strip()}")


def _draw_overlay(frame, bbox: tuple | None, label: str | None):
    """
    Returns a NEW frame (copy) with a bounding box + emotion label
    drawn on it — never mutates the input frame. This matters because
    the caller writes the ORIGINAL (clean, un-overlaid) frame to the
    output video and only shows the overlaid copy in the live preview
    window — the recorded video should not have the overlay burned in.
    """
    import cv2

    display = frame.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if label:
            cv2.putText(display, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return display


# --- The real live capture — needs actual hardware, not testable in a sandbox ---

def _check_capture_dependencies() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise LiveCaptureError(
            "opencv-python not installed. Run `pip install opencv-python` "
            "(NOT opencv-python-headless — that build has no GUI/window support, "
            "which this needs for the live preview window)."
        ) from e
    try:
        import pyaudio  # noqa: F401
    except ImportError as e:
        raise LiveCaptureError(
            "PyAudio not installed. On Windows: `pip install pyaudio` usually works "
            "directly; if it fails to build, install a prebuilt wheel instead — see "
            "requirements notes for the exact command."
        ) from e


def capture_live_av(config: LiveCaptureConfig = LiveCaptureConfig()) -> LiveCaptureResult:
    """
    Opens a live preview window and records from the default camera +
    microphone until the quit key is pressed or max_duration_seconds
    elapses. Requires a real camera and microphone attached to the
    machine this runs on.
    """
    _check_capture_dependencies()
    import cv2
    import pyaudio

    temp_dir = Path(tempfile.mkdtemp(prefix="live_capture_"))
    video_only_path = temp_dir / "video_only.mp4"
    audio_wav_path = temp_dir / "audio.wav"
    final_path = temp_dir / "capture.mp4"

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise LiveCaptureError(f"Could not open camera at index {config.camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=config.audio_channels, rate=config.audio_sample_rate,
            input=True, input_device_index=config.mic_device_index, frames_per_buffer=config.audio_chunk_size,
        )
    except Exception as e:
        cap.release()
        pa.terminate()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise LiveCaptureError(f"Could not open microphone: {e}") from e

    frames: list = []
    audio_chunks: list = []
    stop_event = threading.Event()

    def audio_loop():
        while not stop_event.is_set():
            try:
                data = stream.read(config.audio_chunk_size, exception_on_overflow=False)
                audio_chunks.append(data)
            except Exception as e:
                logger.warning("Audio capture chunk failed: %s", e)
                break

    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()

    # Facial emotion state: readings accumulate for the whole-session
    # aggregate; last_bbox/last_label are held over between analysis
    # attempts so the overlay doesn't flicker every single frame.
    emotion_readings: list = []
    n_emotion_attempts = 0
    last_analysis_time = 0.0  # 0.0 forces an analysis attempt on the very first frame
    last_bbox = None
    last_label = None

    start_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera frame read failed — stopping capture.")
                break
            frames.append(frame)  # the CLEAN frame — no overlay burned in

            now = time.monotonic()

            if config.enable_emotion_analysis and (now - last_analysis_time) >= config.emotion_analysis_interval_seconds:
                reading = analyze_frame(frame, config.emotion_detector_backend, config.min_face_confidence)
                emotion_readings.append(reading)
                n_emotion_attempts += 1
                last_analysis_time = now
                last_bbox, last_label = (reading.bbox, reading.dominant_emotion) if reading else (None, None)

            display_frame = _draw_overlay(frame, last_bbox, last_label) if config.enable_emotion_analysis else frame
            cv2.imshow(config.window_title, display_frame)

            elapsed = now - start_time
            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.quit_key) or elapsed >= config.max_duration_seconds:
                break
    finally:
        stop_event.set()
        audio_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        pa.terminate()

    duration = time.monotonic() - start_time
    if not frames:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise LiveCaptureError("No frames captured — recording ended immediately.")

    try:
        frame_size = (frames[0].shape[1], frames[0].shape[0])
        write_frames_to_video(frames, video_only_path, config.fps, frame_size)
        write_audio_to_wav(audio_chunks, audio_wav_path, config.audio_sample_rate, config.audio_channels)
        mux_audio_video(video_only_path, audio_wav_path, final_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        video_only_path.unlink(missing_ok=True)
        audio_wav_path.unlink(missing_ok=True)

    facial_result = aggregate_readings(emotion_readings, n_emotion_attempts) if config.enable_emotion_analysis else None

    return LiveCaptureResult(
        video_path=final_path, duration_seconds=duration, facial_emotion=facial_result, _temp_dir=temp_dir,
    )


def cleanup_live_capture(result: LiveCaptureResult) -> None:
    if result._temp_dir and result._temp_dir.exists():
        shutil.rmtree(result._temp_dir)


@contextmanager
def live_capture_ctx(config: LiveCaptureConfig = LiveCaptureConfig()):
    result = capture_live_av(config)
    try:
        yield result
    finally:
        cleanup_live_capture(result)
