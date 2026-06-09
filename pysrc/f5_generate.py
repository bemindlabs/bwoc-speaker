#!/usr/bin/env python
"""Drop-in F5-TTS-MLX generator that also accepts a **local model directory**.

The stock `python -m f5_tts_mlx.generate` only takes a Hugging Face repo id for
`--model` (its `fetch_from_hub` runs `snapshot_download`, which rejects a local
path). BWOC ships fine-tuned voices (e.g. a converted F5-TTS-THAI checkpoint) as
a local dir, so we monkeypatch `fetch_from_hub` to short-circuit to that dir when
it exists, then call the *official* `generate()` unchanged — reusing all of its
weight-conversion, vocoder and sampling logic.

Args mirror the upstream CLI exactly, so `bwoc-speaker`'s `f5_mlx` backend can
target this script in place of the module.
"""
import argparse
from pathlib import Path

import f5_tts_mlx.cfm as cfm
from f5_tts_mlx.generate import generate

_orig_fetch = cfm.fetch_from_hub


def _smart_fetch(repo, quantization_bits=None):
    p = Path(repo).expanduser()
    if p.is_dir():
        return p  # local fine-tune: use it as-is, skip the hub
    return _orig_fetch(repo, quantization_bits)


cfm.fetch_from_hub = _smart_fetch


def main() -> int:
    ap = argparse.ArgumentParser(description="F5-TTS-MLX generate (local-dir aware)")
    ap.add_argument("--model", default="lucasnewman/f5-tts-mlx")
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--duration", type=float)
    ap.add_argument("--estimate-duration", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--method", default="rk4")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway-coef", type=float, default=-1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--q", type=int)
    a = ap.parse_args()

    generate(
        generation_text=a.text,
        duration=a.duration,
        estimate_duration=a.estimate_duration,
        model_name=a.model,
        ref_audio_path=a.ref_audio,
        ref_audio_text=a.ref_text,
        steps=a.steps,
        method=a.method,
        cfg_strength=a.cfg,
        sway_sampling_coef=a.sway_coef,
        speed=a.speed,
        seed=a.seed,
        quantization_bits=a.q,
        output_path=a.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
