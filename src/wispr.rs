//! Opt-in STT provider: Wispr Flow's Voice Interface API (REST, cloud).
//!
//! Docs: <https://api-docs.wisprflow.ai/> (`POST /api`, bearer API key, base64
//! 16 kHz PCM16 wav, max 25 MB / 6 min). Selected with
//! `BWOC_SPEAKER_STT_PROVIDER=wispr`; the API key comes only from the
//! `BWOC_SPEAKER_WISPR_API_KEY` env var. The key is handed to `curl` over stdin
//! (`-K -`), so it never reaches argv, disk, or logs.
//!
//! PRIVACY: with this provider the recorded audio (and the language hint) is
//! uploaded to Wispr's servers. Nothing is sent unless it is selected.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::{Command, Stdio};

pub const DEFAULT_URL: &str = "https://platform-api.wisprflow.ai/api/v1/dash/api";
const MAX_BYTES: usize = 25 * 1024 * 1024;

/// Standard (padded) base64 — tiny on purpose, to avoid a new dependency.
pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Request body for `POST /api` (REST v2 schema).
pub fn build_request(wav: &[u8], language: Option<&str>) -> Value {
    json!({
        "audio": base64(wav),
        "language": language.map(|l| vec![l]).unwrap_or_default(),
        "context": { "app": { "type": "ai" } },
    })
}

/// Pull `text` out of the API response.
pub fn parse_response(body: &str) -> Result<String> {
    let v: Value = serde_json::from_str(body).context("Wispr response was not JSON")?;
    match v.get("text").and_then(Value::as_str) {
        Some(t) => Ok(t.trim().to_string()),
        None => bail!("Wispr response had no `text` field"),
    }
}

/// An API key is embedded in a curl config line; refuse anything that could
/// break out of the quoted value.
pub fn validate_key(key: &str) -> Result<&str> {
    let key = key.trim();
    if key.is_empty() {
        bail!("BWOC_SPEAKER_WISPR_API_KEY is empty");
    }
    if key.chars().any(|c| c.is_control() || c == '"' || c == '\\') {
        bail!("BWOC_SPEAKER_WISPR_API_KEY contains invalid characters");
    }
    Ok(key)
}

/// Transcribe `audio` through Wispr Flow's REST API.
pub fn transcribe(audio: &Path, language: Option<&str>) -> Result<String> {
    let key_env = std::env::var("BWOC_SPEAKER_WISPR_API_KEY")
        .context("BWOC_SPEAKER_WISPR_API_KEY not set (required for STT provider `wispr`)")?;
    let key = validate_key(&key_env)?;
    let url = std::env::var("BWOC_SPEAKER_WISPR_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());

    // The API wants 16 kHz PCM16 mono wav; the recorder writes 24 kHz.
    let resampled = audio.with_extension("wispr16k.wav");
    let status = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(audio)
        .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
        .arg(&resampled)
        .status()
        .context("launching ffmpeg to resample for Wispr")?;
    if !status.success() {
        bail!("ffmpeg resample failed ({status})");
    }
    let wav = std::fs::read(&resampled);
    let _ = std::fs::remove_file(&resampled);
    let wav = wav.context("reading resampled audio")?;
    if wav.len() > MAX_BYTES * 3 / 4 {
        bail!("recording too long for Wispr (limit 25 MB / 6 min)");
    }

    // Body holds the audio: owner-only file, removed straight after the call.
    let body_path = audio.with_extension("wispr.json");
    let result = (|| -> Result<String> {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&body_path)
            .context("writing request body")?;
        f.write_all(build_request(&wav, language).to_string().as_bytes())?;
        drop(f);

        let mut child = Command::new("curl")
            .args(["-sS", "--fail-with-body", "--max-time", "60", "-K", "-"])
            .args(["-H", "Content-Type: application/json", "--data-binary"])
            .arg(format!("@{}", body_path.display()))
            .arg(&url)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("launching curl — is it installed?")?;
        child
            .stdin
            .take()
            .context("curl stdin")?
            .write_all(format!("header = \"Authorization: Bearer {key}\"\n").as_bytes())?;
        let out = child.wait_with_output().context("waiting for curl")?;
        if !out.status.success() {
            bail!(
                "Wispr request failed: {} {}",
                String::from_utf8_lossy(&out.stderr).trim(),
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
        parse_response(&String::from_utf8_lossy(&out.stdout))
    })();
    let _ = std::fs::remove_file(&body_path);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn request_shape() {
        let v = build_request(b"foo", Some("th"));
        assert_eq!(v["audio"], "Zm9v");
        assert_eq!(v["language"], json!(["th"]));
        assert_eq!(build_request(b"", None)["language"], json!([]));
    }

    #[test]
    fn response_parsing() {
        assert_eq!(
            parse_response(r#"{"text":" hi there ","id":"x"}"#).unwrap(),
            "hi there"
        );
        assert!(parse_response(r#"{"id":"x"}"#).is_err());
        assert!(parse_response("nope").is_err());
    }

    #[test]
    fn key_validation() {
        assert_eq!(validate_key(" sk-abc \n").unwrap(), "sk-abc");
        assert!(validate_key("").is_err());
        assert!(validate_key("a\"b").is_err());
        assert!(validate_key("a\nb").is_err());
    }
}
