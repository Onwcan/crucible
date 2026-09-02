//! HTTP and SSE client for the Crucible service.
//!
//! The only thing in the TUI that talks to the server, and it talks to it
//! purely through `crate::protocol`. Nothing here knows the server runs on the
//! same machine, or that there is a GPU behind it.
//!
//! SSE is parsed here rather than pulled in as a framework: the protocol is
//! seven lines of framing, and a dependency that reconnects on its own would
//! fight the explicit reconnect policy the app already has.

use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::protocol::{parse_sse_block, ErrorBody, GenerateRequest, Health, Metrics, SseEvent};

/// Anything that can go wrong talking to the service, phrased for a status bar
/// rather than for a log file.
#[derive(Debug, Clone)]
pub enum ClientError {
    /// Could not reach the server at all.
    Unreachable(String),
    /// The server answered, and refused.
    Rejected { status: u16, message: String },
    /// The connection dropped or the body was unreadable mid-stream.
    Transport(String),
    /// The server said something this client could not parse.
    Protocol(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Unreachable(e) => write!(f, "Server unavailable: {e}"),
            ClientError::Rejected { status: 429, message } => {
                write!(f, "Server busy: {message}")
            }
            ClientError::Rejected { status, message } => {
                write!(f, "Request rejected ({status}): {message}")
            }
            ClientError::Transport(e) => write!(f, "Stream disconnected: {e}"),
            ClientError::Protocol(e) => write!(f, "{e}"),
        }
    }
}

/// Strip the noise reqwest puts in front of a connection failure.
fn reachability(e: &reqwest::Error) -> ClientError {
    if e.is_connect() {
        ClientError::Unreachable("connection refused".into())
    } else if e.is_timeout() {
        ClientError::Unreachable("timed out".into())
    } else {
        ClientError::Unreachable(e.to_string())
    }
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    pub fn new(base: impl Into<String>) -> Result<Self, ClientError> {
        let base = base.into();
        let http = reqwest::Client::builder()
            // Polling must fail fast: a hung request should degrade the status
            // indicator quickly, not stall it for the default timeout.
            .connect_timeout(Duration::from_millis(1500))
            .build()
            .map_err(|e| ClientError::Unreachable(e.to_string()))?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub async fn health(&self) -> Result<Health, ClientError> {
        let r = self
            .http
            .get(format!("{}/health", self.base))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| reachability(&e))?;
        if !r.status().is_success() {
            return Err(status_error(r).await);
        }
        r.json::<Health>()
            .await
            .map_err(|e| ClientError::Protocol(format!("unreadable /health response: {e}")))
    }

    pub async fn metrics(&self) -> Result<Metrics, ClientError> {
        let r = self
            .http
            .get(format!("{}/metrics", self.base))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| reachability(&e))?;
        if !r.status().is_success() {
            return Err(status_error(r).await);
        }
        r.json::<Metrics>()
            .await
            .map_err(|e| ClientError::Protocol(format!("unreadable /metrics response: {e}")))
    }

    /// Open a generation stream, forwarding decoded events into `out`.
    ///
    /// Runs until the stream ends, the server sends `done`, or the returned
    /// task is aborted. Aborting drops the HTTP response, which closes the
    /// connection; the server sees the disconnect and cancels the request at
    /// its next scheduler boundary. That is the entire cancellation path --
    /// there is deliberately no second protocol for it.
    ///
    /// No timeout on the stream itself. A queued request behind fifteen others
    /// can legitimately wait, and killing it on a fixed deadline would make the
    /// client the reason a valid request failed.
    pub async fn stream(
        &self,
        prompt: String,
        max_tokens: usize,
        sampling: Option<(f32, usize, u64)>,
        out: mpsc::Sender<StreamMessage>,
    ) {
        // None sends no sampling fields at all, so the request is byte-for-byte
        // what this client sent before sampling existed.
        let req = GenerateRequest {
            prompt,
            max_tokens,
            temperature: sampling.map(|(t, _, _)| t),
            top_k: sampling.map(|(_, k, _)| k),
            seed: sampling.map(|(_, _, s)| s),
        };
        let resp = match self
            .http
            .post(format!("{}/v1/generate/stream", self.base))
            .json(&req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = out.send(StreamMessage::Failed(reachability(&e))).await;
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = out.send(StreamMessage::Failed(status_error(resp).await)).await;
            return;
        }

        let mut buf = String::new();
        let mut body = resp.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = out
                        .send(StreamMessage::Failed(ClientError::Transport(e.to_string())))
                        .await;
                    return;
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // Events are separated by a blank line. Anything after the last
            // separator is a partial event and stays buffered.
            while let Some(idx) = find_separator(&buf) {
                let (block, rest) = buf.split_at(idx.0);
                let block = block.to_string();
                buf = rest[idx.1..].to_string();
                match parse_sse_block(&block) {
                    Ok(Some(SseEvent::Token { token_id, text })) => {
                        if out
                            .send(StreamMessage::Token { token_id, text })
                            .await
                            .is_err()
                        {
                            return; // receiver gone; drop the stream
                        }
                    }
                    Ok(Some(SseEvent::Done {
                        finish_reason,
                        tokens_generated,
                        text,
                    })) => {
                        let _ = out
                            .send(StreamMessage::Done {
                                finish_reason,
                                tokens_generated,
                                text,
                            })
                            .await;
                        return;
                    }
                    Ok(Some(SseEvent::Error { error })) => {
                        let _ = out
                            .send(StreamMessage::Failed(ClientError::Protocol(error)))
                            .await;
                        return;
                    }
                    // Unknown event types and keep-alive comments are skipped
                    // rather than treated as failures.
                    Ok(Some(SseEvent::Unknown(_))) | Ok(None) => {}
                    Err(e) => {
                        let _ = out
                            .send(StreamMessage::Failed(ClientError::Protocol(e)))
                            .await;
                        return;
                    }
                }
            }
        }

        // Body ended without `done`: a cancellation or a dropped connection.
        let _ = out.send(StreamMessage::Ended).await;
    }
}

/// Find the end of the first complete event block.
///
/// Returns (block length, separator length). Handles both `\n\n` and `\r\n\r\n`
/// so a proxy that rewrites line endings cannot stall the stream.
fn find_separator(s: &str) -> Option<(usize, usize)> {
    let lf = s.find("\n\n").map(|i| (i, 2));
    let crlf = s.find("\r\n\r\n").map(|i| (i, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

async fn status_error(r: reqwest::Response) -> ClientError {
    let status = r.status().as_u16();
    let body = r.text().await.unwrap_or_default();
    let message = serde_json::from_str::<ErrorBody>(&body)
        .map(|e| e.error)
        .unwrap_or_else(|_| {
            if body.is_empty() {
                "no detail".into()
            } else {
                body.chars().take(200).collect()
            }
        });
    ClientError::Rejected { status, message }
}

/// What the streaming task reports back to the application.
#[derive(Debug)]
pub enum StreamMessage {
    Token { token_id: usize, text: String },
    Done { finish_reason: String, tokens_generated: usize, text: String },
    Failed(ClientError),
    /// Body closed without a `done` event.
    Ended,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separator_detection_handles_both_line_endings() {
        assert_eq!(find_separator("a\n\nb"), Some((1, 2)));
        assert_eq!(find_separator("a\r\n\r\nb"), Some((1, 4)));
        assert_eq!(find_separator("no separator yet"), None);
        // A partial event must not be consumed.
        assert_eq!(find_separator("event: token\ndata: {"), None);
    }

    #[test]
    fn separator_prefers_whichever_terminator_comes_first() {
        // A block ending in \n\n followed later by \r\n\r\n.
        let s = "one\n\ntwo\r\n\r\n";
        assert_eq!(find_separator(s), Some((3, 2)));
    }

    #[test]
    fn errors_render_for_humans_not_for_logs() {
        let e = ClientError::Rejected {
            status: 429,
            message: "server queue is full (64 waiting); retry shortly".into(),
        };
        assert!(e.to_string().starts_with("Server busy:"), "{e}");

        let e = ClientError::Unreachable("connection refused".into());
        assert_eq!(e.to_string(), "Server unavailable: connection refused");

        let e = ClientError::Rejected {
            status: 400,
            message: "prompt is empty after tokenisation".into(),
        };
        assert!(e.to_string().contains("400"), "{e}");
    }
}
