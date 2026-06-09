//! Per-agent voice configuration.
//!
//! `bwoc-speaker` owns its own voice config rather than extending the framework
//! `Manifest` — a voice is a property of *how we narrate* an agent, not of the
//! agent's runtime identity, and we don't want to drag the (backend-neutral)
//! core manifest into TTS concerns. The config maps `agent-id → voice key` and
//! defines each voice's backend + parameters.
//!
//! Resolution order for the config file:
//!   1. `--config <path>` (CLI)
//!   2. `$BWOC_SPEAKER_CONFIG`
//!   3. `~/.bwoc/speaker/voices.toml`
//!   4. `./voices.toml`
//! If none exist we fall back to a single built-in voice: macOS `say` / Kanya
//! (th_TH), which needs nothing installed.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Which TTS engine renders a voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// macOS `say` — instant, offline, on the Neural Engine. Robotic but always
    /// available on a Mac (Kanya = th_TH). The universal fallback.
    MacSay,
    /// F5-TTS-MLX — on-device voice-clone TTS on Apple Silicon (RTF ~0.15 on
    /// M3/M4). High quality, needs `pip install f5-tts-mlx` + a reference clip.
    F5Mlx,
    /// F5-TTS-THAI on a remote GPU box (the bemind A6000) over SSH. Highest
    /// quality / the locked "P1" voice; offloads compute off the laptop.
    F5Remote,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::MacSay => "mac_say",
            Backend::F5Mlx => "f5_mlx",
            Backend::F5Remote => "f5_remote",
        }
    }
}

/// One named voice. Most fields are backend-specific and optional; an empty
/// profile is a valid `mac_say`/Kanya voice.
#[derive(Debug, Clone, Deserialize)]
pub struct VoiceProfile {
    #[serde(default = "default_backend")]
    pub backend: Backend,

    // --- mac_say ---
    /// `say -v <voice>` name. Defaults to `Kanya` (th_TH).
    #[serde(default)]
    pub say_voice: Option<String>,

    // --- f5 (mlx + remote share these) ---
    /// Reference audio for voice cloning (mono 24 kHz wav, ~5–10 s).
    #[serde(default)]
    pub ref_audio: Option<String>,
    /// Transcript of `ref_audio`.
    #[serde(default)]
    pub ref_text: Option<String>,
    /// Speaking rate. F5-TTS-THAI "P1" = 0.86.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Pitch shift applied to the rendered audio (a post-process, works for
    /// every backend). A ratio: `1.0` = unchanged, `<1` lower/deeper (e.g. an
    /// authoritative voice), `>1` higher/younger. Duration is preserved. This is
    /// how distinct agents get distinct voices on-device when only one Thai
    /// timbre is available.
    #[serde(default)]
    pub pitch: Option<f32>,
    /// RNG seed for reproducible prosody. "P1" = 123.
    #[serde(default)]
    pub seed: Option<i64>,

    // --- f5_mlx only ---
    /// Optional HF repo id or **local dir** of a non-default checkpoint (e.g. a
    /// converted F5-TTS-THAI fine-tune). Omit for the bundled English model.
    /// Local dirs require the wrapper script (see `BWOC_SPEAKER_F5_SCRIPT`).
    #[serde(default)]
    pub model: Option<String>,
    /// Quantization for the MLX model: `4` or `8` bit. Omit for full precision.
    #[serde(default)]
    pub quantize: Option<u8>,
    /// Sampling steps for the neural ODE (more = better/slower). Default 32.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Explicit target duration (seconds). If unset, the backend estimates it
    /// from the text — required for checkpoints without a duration predictor
    /// (most fine-tunes, including F5-TTS-THAI).
    #[serde(default)]
    pub duration: Option<f32>,

    // --- f5_remote only ---
    /// SSH target, e.g. `bmt@192.168.1.113`.
    #[serde(default)]
    pub remote_host: Option<String>,
    /// Shell command run on the remote that must write a wav to `{out}`.
    /// Placeholders substituted before execution: `{text} {ref_audio}
    /// {ref_text} {speed} {seed} {out}`. The local side then `scp`s `{out}` back.
    #[serde(default)]
    pub remote_cmd: Option<String>,
    /// Remote path the command writes to (default `/tmp/bwoc-speaker-out.wav`).
    #[serde(default)]
    pub remote_out: Option<String>,
}

fn default_backend() -> Backend {
    Backend::MacSay
}

impl Default for VoiceProfile {
    fn default() -> Self {
        VoiceProfile {
            backend: Backend::MacSay,
            say_voice: None,
            ref_audio: None,
            ref_text: None,
            speed: None,
            pitch: None,
            seed: None,
            model: None,
            quantize: None,
            steps: None,
            duration: None,
            remote_host: None,
            remote_cmd: None,
            remote_out: None,
        }
    }
}

/// The whole `voices.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerConfig {
    /// Voice key used when an agent has no explicit mapping.
    #[serde(default = "default_voice_key")]
    pub default: String,
    /// `voice-key → profile`.
    #[serde(default)]
    pub voices: HashMap<String, VoiceProfile>,
    /// `agent-id → voice-key`.
    #[serde(default)]
    pub agents: HashMap<String, String>,
}

fn default_voice_key() -> String {
    "kanya".to_string()
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        let mut voices = HashMap::new();
        voices.insert("kanya".to_string(), VoiceProfile::default());
        SpeakerConfig {
            default: "kanya".to_string(),
            voices,
            agents: HashMap::new(),
        }
    }
}

impl SpeakerConfig {
    /// Load from an explicit path, the env var, or the standard locations.
    /// Returns the built-in default config when nothing is found.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        if let Some(p) = Self::resolve_path(explicit) {
            let text = std::fs::read_to_string(&p)
                .with_context(|| format!("reading speaker config {}", p.display()))?;
            let cfg: SpeakerConfig = toml::from_str(&text)
                .with_context(|| format!("parsing speaker config {}", p.display()))?;
            Ok(cfg)
        } else {
            Ok(SpeakerConfig::default())
        }
    }

    fn resolve_path(explicit: Option<&Path>) -> Option<PathBuf> {
        if let Some(p) = explicit {
            return Some(p.to_path_buf());
        }
        if let Ok(p) = std::env::var("BWOC_SPEAKER_CONFIG") {
            return Some(PathBuf::from(p));
        }
        if let Some(home) = home_dir() {
            let p = home.join(".bwoc/speaker/voices.toml");
            if p.exists() {
                return Some(p);
            }
        }
        let cwd = PathBuf::from("voices.toml");
        if cwd.exists() {
            return Some(cwd);
        }
        None
    }

    /// Resolve the voice for an agent id (e.g. `agent-home`). Falls back to the
    /// `default` voice, then to a built-in Kanya profile, so this never fails.
    pub fn voice_for(&self, agent: &str) -> VoiceProfile {
        let key = self
            .agents
            .get(agent)
            .cloned()
            .unwrap_or_else(|| self.default.clone());
        self.voices
            .get(&key)
            .or_else(|| self.voices.get(&self.default))
            .cloned()
            .unwrap_or_default()
    }
}

/// Minimal `$HOME` lookup without pulling in a crate.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
