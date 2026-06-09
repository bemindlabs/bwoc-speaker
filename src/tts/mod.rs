//! Pluggable TTS backends.
//!
//! Every backend turns `text` + a [`VoiceProfile`] into a playable audio file
//! on disk. The dispatcher picks the backend named by the profile, and — unless
//! disabled — falls back to macOS `say` when the chosen backend isn't available
//! (e.g. `f5-tts-mlx` not installed, or the remote box unreachable), so an agent
//! never goes silent on a dev machine.

mod f5_mlx;
mod f5_remote;
mod mac_say;

use crate::config::{Backend, VoiceProfile};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A TTS engine.
pub trait TtsBackend {
    fn name(&self) -> &'static str;
    /// Whether this backend can run right now (tools installed, host set, …).
    fn available(&self, voice: &VoiceProfile) -> bool;
    /// Render `text` to an audio file under `out_dir`, returning its path.
    fn synth(&self, text: &str, voice: &VoiceProfile, out_dir: &Path) -> Result<PathBuf>;
}

fn backend(b: Backend) -> Box<dyn TtsBackend> {
    match b {
        Backend::MacSay => Box::new(mac_say::MacSay),
        Backend::F5Mlx => Box::new(f5_mlx::F5Mlx),
        Backend::F5Remote => Box::new(f5_remote::F5Remote),
    }
}

/// Synthesize `text` for `voice`, returning a playable file path.
///
/// `allow_fallback` controls whether an unavailable backend silently degrades
/// to macOS `say`. With it off, an unavailable backend is a hard error (useful
/// in CI / when you want to know the real backend ran).
pub fn synth(
    text: &str,
    voice: &VoiceProfile,
    out_dir: &Path,
    allow_fallback: bool,
) -> Result<(PathBuf, &'static str)> {
    let chosen = backend(voice.backend);
    let (path, name) = if chosen.available(voice) {
        (chosen.synth(text, voice, out_dir)?, chosen.name())
    } else {
        if !allow_fallback {
            anyhow::bail!(
                "backend `{}` is not available and fallback is disabled",
                voice.backend.as_str()
            );
        }
        eprintln!(
            "[bwoc-speaker] backend `{}` unavailable — falling back to mac_say",
            voice.backend.as_str()
        );
        let fb = mac_say::MacSay;
        (fb.synth(text, &VoiceProfile::default(), out_dir)?, fb.name())
    };
    let path = apply_pitch(path, voice.pitch, out_dir);
    Ok((path, name))
}

/// Shift the pitch of `src` by `ratio` (duration preserved) via ffmpeg, giving
/// each agent a distinct voice. Returns the shifted file; on any failure (no
/// ffmpeg, bad ratio) it returns the original path so audio is never lost.
fn apply_pitch(src: PathBuf, ratio: Option<f32>, out_dir: &Path) -> PathBuf {
    let p = match ratio {
        Some(p) if (p - 1.0).abs() > 0.001 && p > 0.1 => p,
        _ => return src,
    };
    // Probe the sample rate (asetrate needs it); default 24k if probe fails.
    let sr = std::process::Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "a:0", "-show_entries",
            "stream=sample_rate", "-of", "csv=p=0",
        ])
        .arg(&src)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(24000);

    let out = out_dir.join(format!("{}.wav", unique_stem("pitch")));
    // asetrate scales pitch+tempo; atempo restores tempo → pure pitch shift.
    let filter = format!(
        "asetrate={sr}*{p},aresample={sr},atempo={inv}",
        inv = 1.0 / p
    );
    let ok = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(&src)
        .args(["-af", &filter])
        .arg(&out)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok && out.exists() {
        let _ = std::fs::remove_file(&src);
        out
    } else {
        src
    }
}

/// A short, unique, collision-free stem for an output file.
pub fn unique_stem(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{prefix}-{t}-{n}")
}
