//! macOS `say` backend — the universal fallback. Offline, instant, on the ANE.
//! Default voice `Kanya` speaks Thai (th_TH). Outputs AIFF, which `afplay` plays.

use super::{TtsBackend, unique_stem};
use crate::config::VoiceProfile;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct MacSay;

impl TtsBackend for MacSay {
    fn name(&self) -> &'static str {
        "mac_say"
    }

    fn available(&self, _voice: &VoiceProfile) -> bool {
        cfg!(target_os = "macos")
    }

    fn synth(&self, text: &str, voice: &VoiceProfile, out_dir: &Path) -> Result<PathBuf> {
        let voice_name = voice.say_voice.as_deref().unwrap_or("Kanya");
        let out = out_dir.join(format!("{}.aiff", unique_stem("say")));
        let status = Command::new("say")
            .arg("-v")
            .arg(voice_name)
            .arg("-o")
            .arg(&out)
            .arg(text)
            .status()
            .context("launching `say` (is this macOS?)")?;
        if !status.success() {
            bail!("`say` exited with {status}");
        }
        Ok(out)
    }
}
