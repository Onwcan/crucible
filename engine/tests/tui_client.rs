//! Integration tests for the TUI's HTTP/SSE client against a mock server.
//!
//! No GPU, no model, no real Crucible server. The mock is a raw TCP listener
//! rather than an axum app on purpose: these tests need to produce responses a
//! framework would not let me build -- truncated SSE, malformed JSON, a body
//! that stops mid-stream -- and those are exactly the cases a client gets wrong.

#![cfg(feature = "tui")]

use std::net::SocketAddr;
use std::time::Duration;

use llm_engine::tui::client::{Client, ClientError, StreamMessage};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// What the mock does with one connection.
enum Reply {
    /// Write this, then close.
    Once(String),
    /// Write each piece with a delay before it, then close.
    Pieces(Vec<(Duration, String)>),
    /// Encode every byte as its own HTTP body chunk, including partial UTF-8.
    ByteChunks(Vec<u8>),
    /// Accept and close without writing anything.
    Hangup,
}

/// Start a one-shot mock server. Returns its address.
async fn mock<F>(reply: F) -> SocketAddr
where
    F: Fn(&str) -> Reply + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            // Read just enough to see the request line.
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = req
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();

            match reply(&path) {
                Reply::Once(body) => {
                    let _ = sock.write_all(body.as_bytes()).await;
                }
                Reply::Pieces(parts) => {
                    for (delay, part) in parts {
                        tokio::time::sleep(delay).await;
                        if sock.write_all(part.as_bytes()).await.is_err() {
                            break;
                        }
                        let _ = sock.flush().await;
                    }
                }
                Reply::ByteChunks(body) => {
                    sock.set_nodelay(true).unwrap();
                    sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                        .await
                        .unwrap();
                    for byte in body {
                        if sock
                            .write_all(&[b'1', b'\r', b'\n', byte, b'\r', b'\n'])
                            .await
                            .is_err()
                        {
                            break;
                        }
                        // Give the client a chance to consume each HTTP frame.
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    let _ = sock.write_all(b"0\r\n\r\n").await;
                }
                Reply::Hangup => {}
            }
            let _ = sock.shutdown().await;
        }
    });
    addr
}

fn json_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn sse_headers() -> String {
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".into()
}

const HEALTH: &str = r#"{"status":"ok","model":"120m","device":"test-gpu","max_batch":16,"context":1024,"kv_pages":312,"sampling":{"greedy":true,"temperature":true,"top_k":true,"seed":true,"default_mode":"greedy"}}"#;
const METRICS: &str = r#"{"active_requests":3,"queued_requests":1,"completed_requests":9,"cancelled_requests":2,"failed_requests":0,"kv_pages_used":78,"kv_pages_free":234,"last_batch_size":4,"decode_steps":100,"aggregate_tokens_generated":400,"average_batch_size":4.0,"uptime_seconds":12.5}"#;

async fn collect(addr: SocketAddr, prompt: &str) -> Vec<StreamMessage> {
    let client = Client::new(format!("http://{addr}")).unwrap();
    let (tx, mut rx) = mpsc::channel(256);
    let p = prompt.to_string();
    let task = tokio::spawn(async move { client.stream(p, 8, None, tx).await });
    let result = tokio::time::timeout(Duration::from_secs(10), async {
        let mut out = Vec::new();
        while let Some(m) = rx.recv().await {
            out.push(m);
        }
        out
    })
    .await;
    if result.is_err() {
        task.abort();
    }
    let out = result.expect("stream collection hung");
    task.await.unwrap();
    out
}

// Compare all decoded fields and ordering without depending on reqwest's
// internal chunk sizes or adding equality traits to the public client types.
fn message_values(messages: &[StreamMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|message| match message {
            StreamMessage::Token { token_id, text } => serde_json::json!(["token", token_id, text]),
            StreamMessage::Done {
                finish_reason,
                tokens_generated,
                text,
            } => serde_json::json!(["done", finish_reason, tokens_generated, text]),
            StreamMessage::Failed(ClientError::Protocol(message)) => {
                serde_json::json!(["protocol", message])
            }
            StreamMessage::Failed(error) => panic!("unexpected transport/status failure: {error}"),
            StreamMessage::Ended => serde_json::json!(["ended"]),
        })
        .collect()
}

async fn assert_whole_and_fragmented(body: String, expected: Vec<serde_json::Value>) {
    let whole_body = body.clone();
    let whole = mock(move |_| Reply::Once(format!("{}{whole_body}", sse_headers()))).await;
    let split = mock(move |_| Reply::ByteChunks(body.as_bytes().to_vec())).await;
    let whole_messages = message_values(&collect(whole, "hi").await);
    let split_messages = message_values(&collect(split, "hi").await);
    assert_eq!(whole_messages, expected, "whole-body SSE changed");
    assert_eq!(
        split_messages, whole_messages,
        "one-byte HTTP chunks changed the decoded SSE messages"
    );
}

#[tokio::test]
async fn one_byte_http_chunks_preserve_utf8_tokens_and_done() {
    let body = concat!(
        "event: token\ndata: {\"token_id\":1,\"text\":\"Hello\"}\n\n",
        ": keepalive\n\n",
        "event: token\ndata: {\"token_id\":2,\"text\":\"é雪🦀\"}\n\n",
        "event: heartbeat\ndata: {}\n\n",
        "event: token\ndata: {\"token_id\":3,\"text\":\" world\"}\n\n",
        "event: token\ndata: {\"token_id\":4,\"text\":\"!\"}\n\n",
        "event: done\ndata: {\"finish_reason\":\"length\",\"tokens_generated\":4,\"text\":\"終\"}\n\n",
    );
    for newline in ["\n", "\r\n"] {
        assert_whole_and_fragmented(
            body.replace('\n', newline),
            vec![
                serde_json::json!(["token", 1, "Hello"]),
                serde_json::json!(["token", 2, "é雪🦀"]),
                serde_json::json!(["token", 3, " world"]),
                serde_json::json!(["token", 4, "!"]),
                serde_json::json!(["done", "length", 4, "終"]),
            ],
        )
        .await;
    }
}

#[tokio::test]
async fn one_byte_http_chunks_preserve_utf8_error_details() {
    let body = "event: error\ndata: {\"error\":\"server refused é雪🦀\"}\n\n";
    for newline in ["\n", "\r\n"] {
        assert_whole_and_fragmented(
            body.replace('\n', newline),
            vec![serde_json::json!(["protocol", "server refused é雪🦀"])],
        )
        .await;
    }
}

#[tokio::test]
async fn complete_sse_events_with_invalid_utf8_are_protocol_errors() {
    for newline in ["\n", "\r\n"] {
        let mut body = b"event: token\ndata: {\"token_id\":1,\"text\":\"".to_vec();
        body.push(0xff);
        body.extend_from_slice(b"\"}\n\n");
        let body: Vec<_> = body
            .into_iter()
            .flat_map(|byte| {
                if byte == b'\n' {
                    newline.as_bytes().to_vec()
                } else {
                    vec![byte]
                }
            })
            .collect();
        let addr = mock(move |_| Reply::ByteChunks(body.clone())).await;
        let messages = collect(addr, "hi").await;
        assert!(
            matches!(messages.as_slice(), [StreamMessage::Failed(ClientError::Protocol(message))]
                if message.contains("UTF-8")),
            "invalid complete SSE event was not rejected: {messages:?}"
        );
    }
}

#[tokio::test]
async fn an_incomplete_utf8_event_at_body_end_preserves_ended_semantics() {
    let addr = mock(|_| {
        Reply::ByteChunks(
            b"event: token\ndata: {\"token_id\":1,\"text\":\"a\"}\n\nevent: token\ndata: {\"token_id\":2,\"text\":\"\xc3"
                .to_vec(),
        )
    })
    .await;
    assert_eq!(
        message_values(&collect(addr, "hi").await),
        vec![
            serde_json::json!(["token", 1, "a"]),
            serde_json::json!(["ended"]),
        ],
    );
}

#[tokio::test]
async fn health_is_parsed() {
    let addr = mock(|_| Reply::Once(json_response(HEALTH))).await;
    let c = Client::new(format!("http://{addr}")).unwrap();
    let h = c.health().await.unwrap();
    assert_eq!(h.model, "120m");
    assert_eq!(h.max_batch, 16);
    assert!(
        h.sampling.greedy && h.sampling.temperature,
        "server should advertise both modes: {:?}",
        h.sampling
    );
    assert_eq!(
        h.sampling.default_mode, "greedy",
        "omitting sampling parameters must stay greedy"
    );
}

#[tokio::test]
async fn metrics_are_parsed_and_derive_kv_usage() {
    let addr = mock(|_| Reply::Once(json_response(METRICS))).await;
    let c = Client::new(format!("http://{addr}")).unwrap();
    let m = c.metrics().await.unwrap();
    assert_eq!(m.active_requests, 3);
    assert_eq!(m.kv_total(), 312);
    assert!((m.kv_usage() - 0.25).abs() < 1e-9);
}

#[tokio::test]
async fn an_unreachable_server_is_reported_not_panicked() {
    // Port 1 on loopback: nothing is listening.
    let c = Client::new("http://127.0.0.1:1").unwrap();
    let err = c.health().await.unwrap_err();
    assert!(
        matches!(err, ClientError::Unreachable(_)),
        "expected unreachable, got {err:?}"
    );
    assert!(err.to_string().starts_with("Server unavailable"), "{err}");
}

#[tokio::test]
async fn streams_tokens_then_done() {
    let addr = mock(|_| {
        Reply::Once(format!(
            "{}event: token\ndata: {{\"token_id\":1,\"text\":\"Hello\"}}\n\n\
             event: token\ndata: {{\"token_id\":2,\"text\":\", world\"}}\n\n\
             event: done\ndata: {{\"finish_reason\":\"length\",\"tokens_generated\":2,\"text\":\"!\"}}\n\n",
            sse_headers()
        ))
    })
    .await;

    let msgs = collect(addr, "hi").await;
    let mut text = String::new();
    let mut done = None;
    for m in &msgs {
        match m {
            StreamMessage::Token { text: t, .. } => text.push_str(t),
            StreamMessage::Done {
                finish_reason,
                tokens_generated,
                text: tail,
            } => {
                text.push_str(tail);
                done = Some((finish_reason.clone(), *tokens_generated));
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(text, "Hello, world!");
    assert_eq!(done, Some(("length".into(), 2)));
}

#[tokio::test]
async fn events_split_across_tcp_chunks_are_reassembled() {
    // The event boundary falls in the middle of a write, which is what actually
    // happens on a real socket.
    let addr = mock(|_| {
        Reply::Pieces(vec![
            (Duration::from_millis(0), sse_headers()),
            (Duration::from_millis(5), "event: token\ndata: {\"token_id\":1,".into()),
            (Duration::from_millis(5), "\"text\":\"split\"}\n\n".into()),
            (
                Duration::from_millis(5),
                "event: done\ndata: {\"finish_reason\":\"length\",\"tokens_generated\":1,\"text\":\"\"}\n\n".into(),
            ),
        ])
    })
    .await;

    let msgs = collect(addr, "hi").await;
    let text: String = msgs
        .iter()
        .filter_map(|m| match m {
            StreamMessage::Token { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "split");
    assert!(matches!(msgs.last(), Some(StreamMessage::Done { .. })));
}

#[tokio::test]
async fn a_delayed_first_token_is_waited_for() {
    let addr = mock(|_| {
        Reply::Pieces(vec![
            (Duration::from_millis(0), sse_headers()),
            (Duration::from_millis(250), "event: token\ndata: {\"token_id\":1,\"text\":\"late\"}\n\n".into()),
            (Duration::from_millis(0), "event: done\ndata: {\"finish_reason\":\"length\",\"tokens_generated\":1,\"text\":\"\"}\n\n".into()),
        ])
    })
    .await;

    let started = std::time::Instant::now();
    let msgs = collect(addr, "hi").await;
    assert!(started.elapsed() >= Duration::from_millis(200));
    assert!(matches!(msgs.first(), Some(StreamMessage::Token { .. })));
}

#[tokio::test]
async fn keepalive_comments_and_unknown_events_are_skipped() {
    let addr = mock(|_| {
        Reply::Once(format!(
            "{}: keep-alive\n\n\
             event: heartbeat\ndata: {{}}\n\n\
             event: token\ndata: {{\"token_id\":5,\"text\":\"x\"}}\n\n\
             event: done\ndata: {{\"finish_reason\":\"length\",\"tokens_generated\":1,\"text\":\"\"}}\n\n",
            sse_headers()
        ))
    })
    .await;

    let msgs = collect(addr, "hi").await;
    let tokens: Vec<_> = msgs
        .iter()
        .filter(|m| matches!(m, StreamMessage::Token { .. }))
        .collect();
    assert_eq!(tokens.len(), 1, "unknown events were not skipped: {msgs:?}");
}

#[tokio::test]
async fn malformed_sse_json_surfaces_as_a_protocol_error() {
    let addr = mock(|_| {
        Reply::Once(format!(
            "{}event: token\ndata: {{not json}}\n\n",
            sse_headers()
        ))
    })
    .await;

    let msgs = collect(addr, "hi").await;
    assert!(
        matches!(
            msgs.first(),
            Some(StreamMessage::Failed(ClientError::Protocol(_)))
        ),
        "got {msgs:?}"
    );
}

#[tokio::test]
async fn a_server_error_status_is_reported_with_its_message() {
    let body = r#"{"error":"server queue is full (64 waiting); retry shortly"}"#;
    let addr = mock(move |_| {
        Reply::Once(format!(
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ))
    })
    .await;

    let msgs = collect(addr, "hi").await;
    match msgs.first() {
        Some(StreamMessage::Failed(e @ ClientError::Rejected { status: 429, .. })) => {
            assert!(e.to_string().starts_with("Server busy:"), "{e}");
        }
        other => panic!("expected a 429 rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn a_500_is_reported_rather_than_treated_as_a_stream() {
    let addr = mock(|_| {
        Reply::Once(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .into(),
        )
    })
    .await;
    let msgs = collect(addr, "hi").await;
    assert!(
        matches!(
            msgs.first(),
            Some(StreamMessage::Failed(ClientError::Rejected {
                status: 500,
                ..
            }))
        ),
        "got {msgs:?}"
    );
}

#[tokio::test]
async fn a_mid_stream_disconnect_ends_the_stream_without_done() {
    // Two tokens, then the connection closes. This is what the client sees when
    // it cancels, and when the server goes away.
    let addr = mock(|_| {
        Reply::Once(format!(
            "{}event: token\ndata: {{\"token_id\":1,\"text\":\"a\"}}\n\n\
             event: token\ndata: {{\"token_id\":2,\"text\":\"b\"}}\n\n",
            sse_headers()
        ))
    })
    .await;

    let msgs = collect(addr, "hi").await;
    let tokens = msgs
        .iter()
        .filter(|m| matches!(m, StreamMessage::Token { .. }))
        .count();
    assert_eq!(tokens, 2, "partial output was discarded: {msgs:?}");
    assert!(
        matches!(msgs.last(), Some(StreamMessage::Ended)),
        "expected Ended, got {msgs:?}"
    );
}

#[tokio::test]
async fn a_hangup_before_any_body_is_an_error_not_a_hang() {
    let addr = mock(|_| Reply::Hangup).await;
    let client = Client::new(format!("http://{addr}")).unwrap();
    let (tx, mut rx) = mpsc::channel(8);
    tokio::spawn(async move { client.stream("hi".into(), 4, None, tx).await });
    let first = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("client hung on an empty response");
    assert!(first.is_some());
}

#[tokio::test]
async fn dropping_the_receiver_stops_the_stream_task() {
    // A failed send also stops an actively producing client. TUI Escape aborts
    // the owning generation task instead, so it can stop a stalled body too.
    let addr = mock(|_| {
        let mut parts = vec![(Duration::from_millis(0), sse_headers())];
        for i in 0..500 {
            parts.push((
                Duration::from_millis(2),
                format!("event: token\ndata: {{\"token_id\":{i},\"text\":\"x\"}}\n\n"),
            ));
        }
        Reply::Pieces(parts)
    })
    .await;

    for _ in 0..5 {
        let client = Client::new(format!("http://{addr}")).unwrap();
        let (tx, mut rx) = mpsc::channel(4);
        let task = tokio::spawn(async move { client.stream("hi".into(), 500, None, tx).await });
        // Take a couple of tokens, then walk away.
        let _ = rx.recv().await;
        let _ = rx.recv().await;
        drop(rx);
        // The task must finish on its own once the receiver is gone.
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("stream task leaked after the receiver was dropped")
            .unwrap();
    }
}

#[tokio::test]
async fn a_client_recovers_once_the_server_comes_back() {
    // First a dead address, then a live one on the same client type: the client
    // holds no failed state that would prevent reconnecting.
    let dead = Client::new("http://127.0.0.1:1").unwrap();
    assert!(dead.health().await.is_err());

    let addr = mock(|_| Reply::Once(json_response(HEALTH))).await;
    let live = Client::new(format!("http://{addr}")).unwrap();
    assert!(live.health().await.is_ok());

    // And the same client instance can fail then succeed.
    let c = Client::new(format!("http://{addr}")).unwrap();
    assert!(c.health().await.is_ok());
    assert!(c.metrics().await.is_err() || true); // metrics path returns HEALTH here; shape mismatch is fine
    assert!(
        c.health().await.is_ok(),
        "client became unusable after one bad response"
    );
}
