"""
list_audio_devices.py

Lists available microphone input devices (index + name), so you can
pick the right mic_device_index in configs/live_capture_config.yaml if
the system default isn't the one you want.

Usage:
    python scripts/list_audio_devices.py
"""

from __future__ import annotations


def main() -> None:
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        print(f"{'Index':<6} {'Max Input Channels':<20} Name")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(f"{i:<6} {info['maxInputChannels']:<20} {info['name']}")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
