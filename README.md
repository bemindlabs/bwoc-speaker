<div align="center">

# 🜂 bwoc-speaker

**Voice for BWOC agents — turn an agent's replies into spoken audio.**

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Backends](https://img.shields.io/badge/tts-f5--mlx%20%C2%B7%20f5--remote%20%C2%B7%20mac--say-555.svg)](#-backends)

A small daemon that reads the [`bwoc_core::chat_proto`](../bwoc-framework/crates/bwoc-core/src/chat_proto.rs)
stream the harness already speaks and renders each assistant turn as audio,
in a **per-agent voice**, through a **pluggable** TTS backend.

</div>

---

## ✨ What it is

`bwoc-speaker` is a thin **narrator** over the protocol the framework already
speaks. Point it at a `bwoc-harness --chat` stream (or send it lines on a
socket) and it speaks every completed assistant `Message` — never the streaming
`Token`s, never tool noise, never replayed history. It also **listens**:
push-to-talk voice input transcribed on-device (MLX Whisper) and handed back to
an agent — the full talk-back loop, ฟัง + พูด.

The **backend** and the **voice** are resolved per agent from `voices.toml`, so
บุษบา can sound like a soft female voice while a team lead uses the cloned "P1".

## 🔊 Backends

Picked per voice; an unavailable backend degrades to `mac_say` unless `--no-fallback`.

| Backend | Engine | Where it runs | Use it for |
| --- | --- | --- | --- |
| `mac_say` | macOS `say` (Kanya, th_TH) | on-device, ANE | instant, offline, the **universal fallback** — needs nothing |
| `f5_mlx` | [F5-TTS-MLX](https://github.com/lucasnewman/f5-tts-mlx) | on-device, Apple Silicon | high-quality **voice clone** on a Mac (RTF ~0.15 on M3/M4); `pip install f5-tts-mlx` |
| `f5_remote` | F5-TTS-THAI over SSH | remote GPU (bemind A6000) | top quality / the locked **P1** voice; offloads the laptop |

> **Thai note.** Kokoro-82M is lighter but doesn't speak Thai yet, so for Thai
> agents the on-device pick is F5-TTS-MLX with a Thai checkpoint. `mac_say`/Kanya
> always works as a floor.

### On-device Thai (f5_mlx) — one-time setup

Thai F5 checkpoints ship as PyTorch `.pt`; F5-TTS-MLX wants MLX `safetensors`.
Convert once, then point a voice's `model` at the local dir:

```bash
uv venv ~/.bwoc/speaker/venv --python 3.12
uv pip install --python ~/.bwoc/speaker/venv f5-tts-mlx torch safetensors huggingface_hub
~/.bwoc/speaker/venv/bin/python pysrc/convert_thai.py   # → ~/.bwoc/speaker/models/f5-thai

export BWOC_SPEAKER_PYTHON=~/.bwoc/speaker/venv/bin/python
export BWOC_SPEAKER_F5_SCRIPT="$PWD/pysrc/f5_generate.py"   # local-dir loader
```

`pysrc/f5_generate.py` is a drop-in wrapper around F5-TTS-MLX that also accepts a
**local model dir** (the stock module only takes HF repo ids). `convert_thai.py`
lifts the EMA weights into the safetensors layout the MLX loader then remaps.

### Voice input (listen) — setup

```bash
uv pip install --python ~/.bwoc/speaker/venv mlx-whisper
export BWOC_SPEAKER_STT_SCRIPT="$PWD/pysrc/stt.py"
# optional: BWOC_SPEAKER_MIC (avfoundation device, default ":0")
#           BWOC_SPEAKER_STT_MODEL (default mlx-community/whisper-large-v3-turbo)
```

`listen` records with ffmpeg's avfoundation input, so the **terminal needs mic
permission** (System Settings → Privacy → Microphone). `pysrc/stt.py` transcribes
with MLX Whisper (Thai works well) and prints only the text. A convenience
`~/.bwoc/speaker/env.sh` exports all of the above — `source` it once.

## 🚀 Install

```bash
cargo install --path . --force        # → bwoc-speaker on your PATH
cp voices.example.toml ~/.bwoc/speaker/voices.toml   # then edit
```

## 🖱️ Usage

```bash
# one-shot — test a voice
bwoc-speaker say --agent agent-home "สวัสดีค่ะที่รัก"

# narrate a live agent: pipe the harness chat stream in
bwoc-harness --chat agent-yudi | bwoc-speaker pipe --agent agent-yudi

# background voice for the whole fleet: a socket anything can send to
bwoc-speaker daemon                                   # ~/.bwoc/speaker.sock
echo '{"agent":"agent-home","text":"งานเสร็จแล้วค่ะ"}' | nc -U ~/.bwoc/speaker.sock

# voice INPUT — push-to-talk: speak, press Enter, it transcribes (MLX Whisper)
bwoc-speaker listen --language th                      # prints the transcript
bwoc-speaker listen --language th --send agent-home    # …and hands it to the agent
```

Global flags: `--config <path>` · `--backend mac_say|f5_mlx|f5_remote` (force one) ·
`--no-fallback` (error instead of degrading to `say`).

## ⚙️ Config — `voices.toml`

`[agents]` maps an agent id to a voice key; `[voices.<key>]` defines the backend
+ params. See [`voices.example.toml`](voices.example.toml). Lookup order:
`--config` → `$BWOC_SPEAKER_CONFIG` → `~/.bwoc/speaker/voices.toml` → `./voices.toml`
→ built-in Kanya.

```toml
default = "kanya"
[agents]
agent-home = "bussaba"
[voices.kanya]
backend = "mac_say"
[voices.bussaba]
backend = "f5_mlx"
ref_audio = "~/ai-stack/refs/bussaba.wav"
ref_text  = "สวัสดีค่ะ บุษบาเองนะคะ"
speed = 0.9
```

## 🧩 How it fits

```
bwoc-harness --chat ──chat_proto──▶ bwoc-speaker ──▶ TTS backend ──▶ afplay
   (owns the session)                (this crate)      (per voice)     (audio out)
```

The harness owns the session, tools, and model calls. This crate only listens,
filters to spoken-worthy events, and narrates — the same separation `bwoc-chat`
keeps as a *visual* renderer.

## License

MIT
