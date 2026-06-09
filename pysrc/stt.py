#!/usr/bin/env python
"""On-device speech-to-text for BWOC voice input, via MLX Whisper.

Transcribes an audio file and prints the recognized text to stdout (only the
text — so callers can capture it cleanly). Runs on Apple Silicon; the default
model handles Thai well. First run downloads the model.

Usage:
    python stt.py <audio.wav> [--model REPO] [--language th]
Env:
    BWOC_SPEAKER_STT_MODEL  default model repo
"""
import argparse
import os
import sys

import mlx_whisper

DEFAULT_MODEL = os.environ.get(
    "BWOC_SPEAKER_STT_MODEL", "mlx-community/whisper-large-v3-turbo"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", default=None, help="force a language (e.g. th); omit to auto-detect")
    a = ap.parse_args()

    # verbose=False keeps Whisper's segment dump off stdout, so the only thing
    # we print is the final text — callers capture stdout cleanly.
    kwargs = {"path_or_hf_repo": a.model, "verbose": False}
    if a.language:
        kwargs["language"] = a.language

    result = mlx_whisper.transcribe(a.audio, **kwargs)
    text = (result.get("text") or "").strip()
    print(text)
    return 0 if text else 1


if __name__ == "__main__":
    raise SystemExit(main())
