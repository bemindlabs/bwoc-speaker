//! F5-TTS-MLX backend — on-device voice cloning on Apple Silicon.
//!
//! Shells out to F5-TTS-MLX (`pip install f5-tts-mlx`). On an M3/M4 this runs
//! faster than real time (RTF ~0.15). The bundled model is English; set
//! `voice.model` to a fine-tune to clone another voice/language.
//!
//! Two env vars steer it:
//!   - `BWOC_SPEAKER_PYTHON` — the interpreter (a venv with f5-tts-mlx).
//!   - `BWOC_SPEAKER_F5_SCRIPT` — path to the bundled `pysrc/f5_generate.py`
//!     wrapper. The wrapper is **required for a local-dir `model`** (a Thai
//!     fine-tune): the stock `f5_tts_mlx.generate` module only accepts HF repo
//!     ids. Without the env set we fall back to the module (HF ids only).
//!
//! When no `duration` is given we pass `--estimate-duration`, since fine-tunes
//! (incl. F5-TTS-THAI) ship no duration predictor and would otherwise error.

use super::{TtsBackend, unique_stem};
use crate::config::VoiceProfile;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct F5Mlx;

fn python() -> String {
    std::env::var("BWOC_SPEAKER_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// The bundled wrapper script, if configured (enables local-dir models).
fn wrapper_script() -> Option<String> {
    std::env::var("BWOC_SPEAKER_F5_SCRIPT").ok()
}

/// Expand a leading `~/` to `$HOME` so config paths like `~/.bwoc/...` work
/// (the python side doesn't expand tildes passed as argv).
fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}/{}", home.to_string_lossy(), rest);
        }
    }
    p.to_string()
}

impl TtsBackend for F5Mlx {
    fn name(&self) -> &'static str {
        "f5_mlx"
    }

    fn available(&self, _voice: &VoiceProfile) -> bool {
        // arm64 macOS + the module importable.
        if !cfg!(target_os = "macos") {
            return false;
        }
        Command::new(python())
            .args(["-c", "import f5_tts_mlx"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn synth(&self, text: &str, voice: &VoiceProfile, out_dir: &Path) -> Result<PathBuf> {
        let out = out_dir.join(format!("{}.wav", unique_stem("f5mlx")));
        let mut cmd = Command::new(python());
        match wrapper_script() {
            Some(script) => {
                cmd.arg(script);
            }
            None => {
                cmd.args(["-m", "f5_tts_mlx.generate"]);
            }
        }
        cmd.arg("--text").arg(text);
        cmd.arg("--output").arg(&out);
        if let Some(model) = &voice.model {
            cmd.arg("--model").arg(expand_tilde(model));
        }
        if let Some(ref_audio) = &voice.ref_audio {
            cmd.arg("--ref-audio").arg(expand_tilde(ref_audio));
        }
        if let Some(ref_text) = &voice.ref_text {
            cmd.arg("--ref-text").arg(ref_text);
        }
        if let Some(speed) = voice.speed {
            cmd.arg("--speed").arg(speed.to_string());
        }
        if let Some(seed) = voice.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }
        if let Some(q) = voice.quantize {
            cmd.arg("--q").arg(q.to_string());
        }
        cmd.arg("--steps").arg(voice.steps.unwrap_or(32).to_string());
        match voice.duration {
            Some(d) => {
                cmd.arg("--duration").arg(d.to_string());
            }
            // No predictor in fine-tunes → let the wrapper estimate from text.
            None => {
                cmd.arg("--estimate-duration");
            }
        }
        let status = cmd
            .status()
            .with_context(|| format!("launching f5-tts-mlx via {}", python()))?;
        if !status.success() {
            bail!("f5-tts-mlx exited with {status}");
        }
        if !out.exists() {
            bail!("f5-tts-mlx reported success but wrote no file at {out:?}");
        }
        Ok(out)
    }
}
