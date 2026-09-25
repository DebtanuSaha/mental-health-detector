"""
build_test_video_fixture.py (dev utility — NOT part of the production pipeline)

Builds a tiny, fully synthetic test video (ffmpeg's built-in testsrc
color-bar pattern + a sine-wave tone) entirely from ffmpeg's own
generators — no real video file needed, nothing copyrighted, fully
reproducible. Used to smoke-test ml/video/ingestion.py without
depending on an external video file.

Run once (already run — output is committed under tests/fixtures/):
    python scripts/build_test_video_fixture.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_test_video.mp4"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(OUT_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    print(f"Tiny synthetic test video written to {OUT_PATH}")


if __name__ == "__main__":
    main()
