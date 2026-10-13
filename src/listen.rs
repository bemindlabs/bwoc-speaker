//! Voice input — push-to-talk capture + on-device STT (MLX Whisper).
//!
//! `record_ptt` records from the mic with ffmpeg's avfoundation input and stops
//! when the operator presses Enter (a terminal-friendly push-to-talk: Enter to
//! stop the take). `transcribe` shells the bundled `pysrc/stt.py` wrapper, which
//! prints only the recognized text. The mic device is `BWOC_SPEAKER_MIC`
//! (avfoundation spec, default `:0` = audio device 0, no video).

use anyhow::{Context, Result, bail};
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// Record until the operator presses Enter. Writes a 24 kHz mono wav to `out`.
pub fn record_ptt(out: &Path) -> Result<()> {
    let mic = std::env::var("BWOC_SPEAKER_MIC").unwrap_or_else(|_| ":0".to_string());
    eprintln!("🎤 อัดเสียงอยู่… พูดได้เลย แล้วกด Enter เพื่อหยุด");
    let mut child = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-y",
            "-f",
            "avfoundation",
            "-i",
            &mic,
            "-ar",
            "24000",
            "-ac",
            "1",
        ])
        .arg(out)
        .stdin(Stdio::piped())
        .spawn()
        .context("launching ffmpeg (avfoundation) — is ffmpeg installed?")?;

    // Block until the operator hits Enter.
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).ok();

    // `q` on ffmpeg's stdin makes it finalize the file cleanly.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"q\n");
        let _ = stdin.flush();
    }
    let _ = child.wait().context("waiting for ffmpeg")?;
    if !out.exists() || std::fs::metadata(out).map(|m| m.len()).unwrap_or(0) == 0 {
        bail!("recording produced no audio (check mic device BWOC_SPEAKER_MIC)");
    }
    Ok(())
}

/// Which engine turns the recording into text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttProvider {
    /// On-device MLX Whisper (default; audio never leaves the machine).
    Whisper,
    /// Wispr Flow cloud API (opt-in; audio is uploaded to Wispr).
    Wispr,
}

impl SttProvider {
    /// Parse `BWOC_SPEAKER_STT_PROVIDER` (unset/empty = whisper).
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("whisper") | Some("mlx") => Ok(Self::Whisper),
            Some("wispr") | Some("wispr-flow") => Ok(Self::Wispr),
            Some(other) => bail!("unknown BWOC_SPEAKER_STT_PROVIDER `{other}` (whisper|wispr)"),
        }
    }

    pub fn from_env() -> Result<Self> {
        Self::parse(std::env::var("BWOC_SPEAKER_STT_PROVIDER").ok().as_deref())
    }
}

/// Transcribe `audio` with the selected provider.
pub fn transcribe(audio: &Path, language: Option<&str>) -> Result<String> {
    match SttProvider::from_env()? {
        SttProvider::Whisper => transcribe_whisper(audio, language),
        SttProvider::Wispr => crate::wispr::transcribe(audio, language),
    }
}

/// Transcribe `audio` via the MLX Whisper wrapper, returning the text.
fn transcribe_whisper(audio: &Path, language: Option<&str>) -> Result<String> {
    let python = std::env::var("BWOC_SPEAKER_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = std::env::var("BWOC_SPEAKER_STT_SCRIPT")
        .context("BWOC_SPEAKER_STT_SCRIPT not set (path to pysrc/stt.py)")?;
    let mut cmd = Command::new(python);
    cmd.arg(script).arg(audio);
    if let Some(l) = language {
        cmd.arg("--language").arg(l);
    }
    let output = cmd.output().context("launching stt.py")?;
    if !output.status.success() {
        bail!(
            "transcription failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_selection() {
        assert_eq!(SttProvider::parse(None).unwrap(), SttProvider::Whisper);
        assert_eq!(SttProvider::parse(Some("")).unwrap(), SttProvider::Whisper);
        assert_eq!(
            SttProvider::parse(Some("Whisper")).unwrap(),
            SttProvider::Whisper
        );
        assert_eq!(
            SttProvider::parse(Some(" WISPR ")).unwrap(),
            SttProvider::Wispr
        );
        assert!(SttProvider::parse(Some("bogus")).is_err());
    }
}
