//! Wire types shared by the HTTP service and its clients.
//!
//! This module exists so a client can depend on the *protocol* without
//! depending on the engine. It has no CUDA types, no runtime types, and no
//! feature gates: the TUI links against this and nothing else from the
//! inference side, which is what keeps the client/server boundary real rather
//! than merely intended.
//!
//! Both directions are derived here rather than duplicated in the client.
//! Duplicating them is how a server and its client quietly drift apart, and the
//! drift only shows up as a field silently deserialising to its default.

use serde::{Deserialize, Serialize};

/// `GET /health`. Stable facts about the service, fetched on connect rather
/// than polled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub model: String,
    pub device: String,
    pub max_batch: usize,
    pub context: usize,
    pub kv_pages: usize,
    /// Decoding modes the server supports.
    ///
    /// Was a bare string when greedy was the only option. Kept as a field of
    /// the same name so existing clients still find something there, but it is
    /// now structured: advertising "greedy" while accepting temperature would
    /// be a lie, and a capability list is the smallest honest replacement.
    pub sampling: SamplingCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCapabilities {
    pub greedy: bool,
    pub temperature: bool,
    pub top_k: bool,
    pub seed: bool,
    /// What a request gets when it specifies nothing.
    pub default_mode: String,
}

impl Default for SamplingCapabilities {
    fn default() -> Self {
        Self {
            greedy: true,
            temperature: true,
            top_k: true,
            seed: true,
            default_mode: "greedy".into(),
        }
    }
}

impl SamplingCapabilities {
    /// One-line summary for a status bar.
    pub fn summary(&self) -> String {
        if self.temperature {
            "greedy + top-k".into()
        } else {
            "greedy".into()
        }
    }
}

/// `GET /metrics`. Counters are cumulative since server start.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub failed_requests: u64,
    pub kv_pages_used: usize,
    pub kv_pages_free: usize,
    pub last_batch_size: usize,
    pub decode_steps: u64,
    pub aggregate_tokens_generated: u64,
    pub average_batch_size: f64,
    pub uptime_seconds: f64,
    /// Cumulative request counts by decoding mode. Two counters, incremented
    /// once at admission -- not per token, and not per request labels.
    #[serde(default)]
    pub greedy_requests: u64,
    #[serde(default)]
    pub sampled_requests: u64,
    /// Requests holding pages and still working through their prompt.
    ///
    /// Prefill is scheduled work now, so it has a queue depth of its own.
    /// All four come from the scheduler's own state once per step rather than
    /// from the GPU, so none of them costs a synchronisation.
    #[serde(default)]
    pub prefilling_requests: usize,
    #[serde(default)]
    pub prefill_chunks: u64,
    #[serde(default)]
    pub prefill_tokens: u64,
    #[serde(default)]
    pub last_prefill_chunk_tokens: usize,
}

impl Metrics {
    /// Fraction of the KV pool in use, 0.0 to 1.0.
    pub fn kv_usage(&self) -> f64 {
        let total = self.kv_pages_used + self.kv_pages_free;
        if total == 0 {
            0.0
        } else {
            self.kv_pages_used as f64 / total as f64
        }
    }

    pub fn kv_total(&self) -> usize {
        self.kv_pages_used + self.kv_pages_free
    }
}

/// Body of `POST /v1/generate` and `/v1/generate/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Omitted means greedy.
    ///
    /// This is the backward-compatibility hinge: every client written before
    /// sampling existed omits it and must keep getting exactly what it got
    /// before. A default of 0.8 here would silently change every existing
    /// caller's output.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Omitted uses a fixed documented seed, so a sampled request without one
    /// is still reproducible. Entropy-seeded randomness is deliberately not
    /// offered: reproducibility is worth more here than convenience.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl GenerateRequest {
    /// Whether this request asked for sampling at all.
    pub fn wants_sampling(&self) -> bool {
        matches!(self.temperature, Some(t) if t > 0.0)
    }
}

pub fn default_max_tokens() -> usize {
    64
}

/// Body of a non-streaming `POST /v1/generate` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: String,
    pub prompt_tokens: usize,
}

/// Error body returned for a 4xx or 5xx.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
}

// --- SSE payloads -----------------------------------------------------------

/// `event: token`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    pub token_id: usize,
    pub text: String,
}

/// `event: done`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoneEvent {
    pub finish_reason: String,
    pub tokens_generated: usize,
    /// Any text still buffered by the server's incremental UTF-8 decoder. A
    /// client that ignores this drops the tail of a multi-byte character.
    #[serde(default)]
    pub text: String,
}

/// `event: error`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    pub error: String,
}

/// One decoded server-sent event.
#[derive(Debug, Clone, PartialEq)]
pub enum SseEvent {
    Token { token_id: usize, text: String },
    Done { finish_reason: String, tokens_generated: usize, text: String },
    Error { error: String },
    /// A well-formed event this client does not know about. Ignored rather than
    /// treated as a failure, so adding an event type to the server does not
    /// break older clients.
    Unknown(String),
}

/// Parse one SSE block: the lines between blank-line separators.
///
/// Deliberately tolerant in the ways the spec requires and strict in the ways
/// that matter. Comment lines (`:`) and unknown fields are skipped, multiple
/// `data:` lines concatenate with newlines, and a block whose payload is not
/// valid JSON is an error the caller can surface rather than a panic.
pub fn parse_sse_block(block: &str) -> Result<Option<SseEvent>, String> {
    let mut event: Option<&str> = None;
    let mut data = String::new();

    for line in block.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => event = Some(value),
            "data" => {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value);
            }
            _ => {}
        }
    }

    if data.is_empty() && event.is_none() {
        return Ok(None);
    }

    let parse = |what: &str| -> String {
        format!("malformed {what} event from server: {data}")
    };

    match event {
        Some("token") => {
            let t: TokenEvent =
                serde_json::from_str(&data).map_err(|_| parse("token"))?;
            Ok(Some(SseEvent::Token {
                token_id: t.token_id,
                text: t.text,
            }))
        }
        Some("done") => {
            let d: DoneEvent = serde_json::from_str(&data).map_err(|_| parse("done"))?;
            Ok(Some(SseEvent::Done {
                finish_reason: d.finish_reason,
                tokens_generated: d.tokens_generated,
                text: d.text,
            }))
        }
        Some("error") => {
            let e: StreamError =
                serde_json::from_str(&data).map_err(|_| parse("error"))?;
            Ok(Some(SseEvent::Error { error: e.error }))
        }
        Some(other) => Ok(Some(SseEvent::Unknown(other.to_string()))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_token_event() {
        let ev = parse_sse_block("event: token\ndata: {\"token_id\":42,\"text\":\" Paris\"}")
            .unwrap()
            .unwrap();
        assert_eq!(
            ev,
            SseEvent::Token {
                token_id: 42,
                text: " Paris".into()
            }
        );
    }

    #[test]
    fn parses_a_done_event_including_the_tail() {
        let ev = parse_sse_block(
            "event: done\ndata: {\"finish_reason\":\"length\",\"tokens_generated\":8,\"text\":\"!\"}",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            ev,
            SseEvent::Done {
                finish_reason: "length".into(),
                tokens_generated: 8,
                text: "!".into()
            }
        );
    }

    #[test]
    fn done_without_a_tail_field_still_parses() {
        // The tail is optional; an older or minimal server may omit it.
        let ev = parse_sse_block(
            "event: done\ndata: {\"finish_reason\":\"cancelled\",\"tokens_generated\":3}",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(ev, SseEvent::Done { ref text, .. } if text.is_empty()));
    }

    #[test]
    fn tolerates_crlf_comments_and_unknown_fields() {
        let block = ": keep-alive\r\nid: 7\r\nevent: token\r\ndata: {\"token_id\":1,\"text\":\"a\"}\r\n";
        let ev = parse_sse_block(block).unwrap().unwrap();
        assert_eq!(ev, SseEvent::Token { token_id: 1, text: "a".into() });
    }

    #[test]
    fn concatenates_multiple_data_lines() {
        let block = "event: token\ndata: {\"token_id\":1,\ndata: \"text\":\"x\"}";
        let ev = parse_sse_block(block).unwrap().unwrap();
        assert_eq!(ev, SseEvent::Token { token_id: 1, text: "x".into() });
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        let err = parse_sse_block("event: token\ndata: {not json").unwrap_err();
        assert!(err.contains("malformed token"), "{err}");
    }

    #[test]
    fn unknown_event_types_are_ignored_rather_than_fatal() {
        let ev = parse_sse_block("event: heartbeat\ndata: {}").unwrap().unwrap();
        assert!(matches!(ev, SseEvent::Unknown(ref s) if s == "heartbeat"));
    }

    #[test]
    fn an_empty_block_yields_nothing() {
        assert_eq!(parse_sse_block("").unwrap(), None);
        assert_eq!(parse_sse_block(": comment only").unwrap(), None);
    }

    #[test]
    fn kv_usage_is_safe_when_the_pool_is_unknown() {
        let m = Metrics::default();
        assert_eq!(m.kv_usage(), 0.0);
        let m = Metrics {
            kv_pages_used: 78,
            kv_pages_free: 234,
            ..Default::default()
        };
        assert_eq!(m.kv_total(), 312);
        assert!((m.kv_usage() - 0.25).abs() < 1e-9);
    }
}
