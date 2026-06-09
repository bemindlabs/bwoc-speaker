//! Remote F5-TTS-THAI backend — synth on a GPU box (the bemind A6000) over SSH.
//!
//! The exact F5-TTS-THAI inference invocation lives on the server and varies by
//! setup, so rather than hard-code it we run a user-supplied `remote_cmd`
//! template that must write a wav to `{out}` on the remote; we then `scp` it
//! back. Placeholders: `{text} {ref_audio} {ref_text} {speed} {seed} {out}`.
//!
//! Example `remote_cmd` (voices.toml):
//! ```text
//! remote_cmd = "cd ~/ai-stack && conda run -n f5tts-thai \
//!   python f5_tts/infer.py --ref '{ref_audio}' --ref-text '{ref_text}' \
//!   --gen '{text}' --speed {speed} --seed {seed} --out '{out}'"
//! ```

use super::{TtsBackend, unique_stem};
use crate::config::VoiceProfile;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct F5Remote;

const DEFAULT_REMOTE_OUT: &str = "/tmp/bwoc-speaker-out.wav";

impl TtsBackend for F5Remote {
    fn name(&self) -> &'static str {
        "f5_remote"
    }

    fn available(&self, voice: &VoiceProfile) -> bool {
        // Needs a host, a command template, and `ssh` on PATH. We don't probe
        // the network here (that would stall every turn); a dead host surfaces
        // as a synth error and — if enabled — falls back to mac_say.
        voice.remote_host.is_some()
            && voice.remote_cmd.is_some()
            && Command::new("sh")
                .args(["-c", "command -v ssh"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }

    fn synth(&self, text: &str, voice: &VoiceProfile, out_dir: &Path) -> Result<PathBuf> {
        let host = voice
            .remote_host
            .as_deref()
            .context("f5_remote: `remote_host` not set")?;
        let template = voice
            .remote_cmd
            .as_deref()
            .context("f5_remote: `remote_cmd` not set")?;
        let remote_out = voice.remote_out.as_deref().unwrap_or(DEFAULT_REMOTE_OUT);

        let remote_cmd = template
            .replace("{text}", &shell_escape(text))
            .replace("{ref_audio}", voice.ref_audio.as_deref().unwrap_or(""))
            .replace("{ref_text}", voice.ref_text.as_deref().unwrap_or(""))
            .replace(
                "{speed}",
                &voice.speed.map(|s| s.to_string()).unwrap_or_default(),
            )
            .replace(
                "{seed}",
                &voice.seed.map(|s| s.to_string()).unwrap_or_default(),
            )
            .replace("{out}", remote_out);

        // 1. run the synth on the remote
        let status = Command::new("ssh")
            .arg(host)
            .arg(&remote_cmd)
            .status()
            .with_context(|| format!("ssh {host}"))?;
        if !status.success() {
            bail!("remote synth command exited with {status}");
        }

        // 2. copy the result back
        let local = out_dir.join(format!("{}.wav", unique_stem("f5remote")));
        let status = Command::new("scp")
            .arg(format!("{host}:{remote_out}"))
            .arg(&local)
            .status()
            .with_context(|| format!("scp {host}:{remote_out}"))?;
        if !status.success() {
            bail!("scp of remote wav exited with {status}");
        }
        if !local.exists() {
            bail!("remote synth produced no local file at {local:?}");
        }
        Ok(local)
    }
}

/// Single-quote a string for safe interpolation into a remote shell command.
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}
