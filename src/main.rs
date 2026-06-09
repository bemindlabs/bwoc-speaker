//! `bwoc-speaker` — give BWOC agents a voice.
//!
//! Three modes, one pluggable TTS pipeline:
//!   - `say`    one-shot: synth + play a line in an agent's voice (great for tests)
//!   - `pipe`   read a `bwoc-harness --chat` event stream on stdin, speak each
//!              assistant turn (compose with the harness or `bwoc-chat`)
//!   - `daemon` listen on a Unix socket; any process can send `{agent,text}`
//!              JSON lines to be spoken — one background voice for the fleet
//!
//! The backend (on-device F5-TTS-MLX / remote F5-TTS-THAI / macOS `say`) and the
//! voice are resolved per agent from `voices.toml`. See `config.rs`.

mod config;
mod listen;
mod player;
mod stream;
mod tts;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use config::{Backend, SpeakerConfig};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "bwoc-speaker", version, about = "Voice for BWOC agents.")]
struct Cli {
    #[command(flatten)]
    common: Common,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// Path to voices.toml (else $BWOC_SPEAKER_CONFIG, ~/.bwoc/speaker/voices.toml, ./voices.toml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Force a backend for every voice: mac_say | f5_mlx | f5_remote.
    #[arg(long, global = true)]
    backend: Option<String>,
    /// Disable the automatic mac_say fallback when a backend is unavailable.
    #[arg(long, global = true)]
    no_fallback: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Speak a single line in an agent's voice.
    Say {
        /// Agent id (default: the config `default` voice).
        #[arg(long)]
        agent: Option<String>,
        /// The text to speak.
        text: Vec<String>,
    },
    /// Read a chat_proto event stream on stdin and speak each assistant turn.
    Pipe {
        /// Pin the voice to this agent id (overrides the stream's Ready). Omit
        /// to follow whichever agent the stream announces.
        #[arg(long)]
        agent: Option<String>,
    },
    /// Run as a background voice service listening on a Unix socket.
    Daemon {
        /// Socket path (default: ~/.bwoc/speaker.sock).
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Push-to-talk: record from the mic, transcribe (MLX Whisper), print the
    /// text — optionally hand it straight to an agent with `--send`.
    Listen {
        /// Force a language for STT (e.g. `th`); omit to auto-detect.
        #[arg(long)]
        language: Option<String>,
        /// Send the transcript to this agent via `bwoc send <agent>`.
        #[arg(long)]
        send: Option<String>,
    },
}

/// Owns the config + render settings and turns text into played audio.
struct Speaker {
    config: SpeakerConfig,
    backend_override: Option<Backend>,
    allow_fallback: bool,
    out_dir: PathBuf,
}

impl Speaker {
    fn speak(&self, agent: &str, text: &str) {
        let mut voice = self.config.voice_for(agent);
        if let Some(b) = self.backend_override {
            voice.backend = b;
        }
        let result = tts::synth(text, &voice, &self.out_dir, self.allow_fallback)
            .and_then(|(path, used)| {
                eprintln!("[bwoc-speaker] {agent} · {used} · {}", preview(text));
                let r = player::play(&path);
                let _ = std::fs::remove_file(&path);
                r
            });
        if let Err(e) = result {
            eprintln!("[bwoc-speaker] failed to speak for {agent}: {e:#}");
        }
    }
}

/// A serialized speak queue: a worker thread plays one utterance at a time so
/// voices never overlap, while producers (stdin / socket) keep reading.
struct SpeakQueue {
    tx: mpsc::Sender<(String, String)>,
    worker: std::thread::JoinHandle<()>,
}

impl SpeakQueue {
    fn new(speaker: Arc<Speaker>) -> Self {
        let (tx, rx) = mpsc::channel::<(String, String)>();
        let worker = std::thread::spawn(move || {
            for (agent, text) in rx {
                speaker.speak(&agent, &text);
            }
        });
        SpeakQueue { tx, worker }
    }

    fn enqueue(&self, agent: String, text: String) {
        let _ = self.tx.send((agent, text));
    }

    /// Drop the sender and wait for the queue to drain.
    fn finish(self) {
        drop(self.tx);
        let _ = self.worker.join();
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let backend_override = match cli.common.backend.as_deref() {
        None => None,
        Some(s) => Some(parse_backend(s)?),
    };
    let out_dir = std::env::temp_dir().join("bwoc-speaker");
    std::fs::create_dir_all(&out_dir).context("creating temp output dir")?;

    let speaker = Arc::new(Speaker {
        config: SpeakerConfig::load(cli.common.config.as_deref())?,
        backend_override,
        allow_fallback: !cli.common.no_fallback,
        out_dir,
    });

    match cli.cmd {
        Cmd::Say { agent, text } => {
            let line = text.join(" ");
            if line.trim().is_empty() {
                bail!("nothing to say — provide text");
            }
            let agent = agent.unwrap_or_else(|| speaker.config.default.clone());
            speaker.speak(&agent, &line);
        }
        Cmd::Pipe { agent } => {
            let queue = SpeakQueue::new(Arc::clone(&speaker));
            let fallback = speaker.config.default.clone();
            let stdin = std::io::stdin();
            stream::pump(stdin.lock(), &fallback, agent.as_deref(), |a, t| {
                queue.enqueue(a.to_string(), t.to_string())
            });
            queue.finish();
        }
        Cmd::Daemon { socket } => run_daemon(speaker, socket)?,
        Cmd::Listen { language, send } => run_listen(&speaker, language.as_deref(), send.as_deref())?,
    }
    Ok(())
}

fn run_listen(speaker: &Speaker, language: Option<&str>, send: Option<&str>) -> Result<()> {
    let wav = speaker.out_dir.join(format!("{}.wav", tts::unique_stem("listen")));
    listen::record_ptt(&wav)?;
    eprintln!("[bwoc-speaker] กำลังถอดเสียง…");
    let text = listen::transcribe(&wav, language)?;
    let _ = std::fs::remove_file(&wav);
    if text.is_empty() {
        bail!("got no speech from the recording");
    }
    // The transcript is the command's output (stdout) — composable in a pipe.
    println!("{text}");

    if let Some(agent) = send {
        eprintln!("[bwoc-speaker] ส่งให้ {agent}…");
        let status = Command::new(bwoc_core::exec::binary_or_name("bwoc"))
            .arg("send")
            .arg(agent)
            .arg(&text)
            .status()
            .context("launching `bwoc send`")?;
        if !status.success() {
            bail!("`bwoc send` exited with {status}");
        }
    }
    Ok(())
}

fn run_daemon(speaker: Arc<Speaker>, socket: Option<PathBuf>) -> Result<()> {
    use std::os::unix::net::UnixListener;

    let path = match socket {
        Some(p) => p,
        None => config::home_dir()
            .context("no $HOME for default socket path")?
            .join(".bwoc/speaker.sock"),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Clear a stale socket from a previous run.
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
    let listener = UnixListener::bind(&path)
        .with_context(|| format!("binding speaker socket {}", path.display()))?;
    eprintln!("[bwoc-speaker] daemon listening on {}", path.display());

    let queue = SpeakQueue::new(Arc::clone(&speaker));
    for conn in listener.incoming() {
        let conn = match conn {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[bwoc-speaker] accept error: {e}");
                continue;
            }
        };
        // Each client streams JSON lines: {"agent":"agent-home","text":"…"}.
        let reader = BufReader::new(conn);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<SpeakRequest>(line) {
                Ok(req) => queue.enqueue(
                    req.agent.unwrap_or_else(|| speaker.config.default.clone()),
                    req.text,
                ),
                Err(e) => eprintln!("[bwoc-speaker] bad request: {e}"),
            }
        }
    }
    queue.finish();
    Ok(())
}

#[derive(serde::Deserialize)]
struct SpeakRequest {
    #[serde(default)]
    agent: Option<String>,
    text: String,
}

fn parse_backend(s: &str) -> Result<Backend> {
    Ok(match s {
        "mac_say" | "say" => Backend::MacSay,
        "f5_mlx" | "mlx" => Backend::F5Mlx,
        "f5_remote" | "remote" => Backend::F5Remote,
        other => bail!("unknown backend `{other}` (mac_say | f5_mlx | f5_remote)"),
    })
}

/// A short one-line preview of an utterance for the log.
fn preview(text: &str) -> String {
    const MAX: usize = 60;
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX {
        flat
    } else {
        let head: String = flat.chars().take(MAX).collect();
        format!("{head}…")
    }
}
