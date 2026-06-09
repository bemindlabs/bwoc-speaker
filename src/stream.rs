//! Consume a `bwoc-harness --chat` event stream and surface the bits worth
//! speaking. We read [`ChatEvent`] JSON lines and emit `(agent, text)` for every
//! completed assistant [`ChatEvent::Message`] — not `Token` deltas (we speak
//! whole turns, not partial words) and not `Restored` history (already spoken,
//! or never was).

use bwoc_core::chat_proto::ChatEvent;
use std::io::BufRead;

/// Drive `reader` to EOF, invoking `on_utterance(agent, text)` for each complete
/// assistant message.
///
/// If `pinned` is `Some`, that agent id is used for every utterance (an explicit
/// `--agent` pins the voice). Otherwise `agent` tracks the latest
/// [`ChatEvent::Ready`], starting from `fallback_agent`.
pub fn pump<R: BufRead>(
    reader: R,
    fallback_agent: &str,
    pinned: Option<&str>,
    mut on_utterance: impl FnMut(&str, &str),
) {
    let mut agent = pinned.unwrap_or(fallback_agent).to_string();
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<ChatEvent>(line) {
            // A pinned agent wins; otherwise follow the stream's Ready.
            Ok(ChatEvent::Ready { agent: a, .. }) => {
                if pinned.is_none() {
                    agent = a;
                }
            }
            Ok(ChatEvent::Message { text }) => {
                let spoken = clean(&text);
                if !spoken.is_empty() {
                    on_utterance(&agent, &spoken);
                }
            }
            Ok(ChatEvent::Bye) => break,
            // Token / ToolCall / ToolResult / Restored / … — not spoken.
            Ok(_) => {}
            // Forward-compat: ignore unparseable / unknown lines.
            Err(_) => {}
        }
    }
}

/// Strip the parts of an assistant message that shouldn't be read aloud:
/// fenced code blocks and the noisier markdown punctuation. Keeps it readable
/// for a TTS engine without trying to be a full markdown renderer.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("```") {
            in_fence = !in_fence;
            if in_fence {
                out.push_str(" (โค้ด) ");
            }
            continue;
        }
        if in_fence {
            continue;
        }
        out.push_str(line);
        out.push(' ');
    }
    out.chars()
        .map(|c| match c {
            '*' | '_' | '`' | '#' | '>' | '|' => ' ',
            other => other,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
