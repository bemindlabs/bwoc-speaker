//! Audio playback — shell out to the platform player, no audio crates.
//!
//! macOS: `afplay` (handles both .aiff from `say` and .wav). Linux: try
//! `paplay` (PulseAudio/PipeWire) then `aplay` (ALSA).

use anyhow::{Result, bail};
use std::path::Path;
use std::process::Command;

/// Play `file` to completion (blocking). Errors if no player is available.
pub fn play(file: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        run("afplay", &[file])
    }
    #[cfg(not(target_os = "macos"))]
    {
        if which("paplay") {
            run("paplay", &[file])
        } else if which("aplay") {
            run("aplay", &[file])
        } else {
            bail!("no audio player found (install pulseaudio/pipewire or alsa-utils)")
        }
    }
}

fn run(bin: &str, args: &[&Path]) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to launch {bin}: {e}"))?;
    if !status.success() {
        bail!("{bin} exited with {status}");
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn which(bin: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {bin}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
