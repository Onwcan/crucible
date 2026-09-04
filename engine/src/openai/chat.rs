//! Chat Completions: `POST /v1/chat/completions`.
//!
//! # Protocol compatibility is not a claim about the model
//!
//! This endpoint exists so that chat-shaped clients can talk to Crucible. The
//! checkpoint behind it is a **base language model** trained on FineWeb-Edu.
//! It has never been instruction-tuned, has no chat fine-tune, and ships no
//! trained chat template. It will continue the text it is given, which for a
//! transcript-shaped prompt means producing something transcript-shaped -- not
//! following instructions.
//!
//! Saying so is part of the implementation. The alternative -- serving this
//! endpoint silently and letting the shape of the API imply a capability -- is
//! the kind of thing that makes a benchmark look fine and a user's expectations
//! wrong.
//!
//! # Serialization
//!
//! Messages become one plain-text transcript. The rules, chosen for the GPT-2
//! BPE vocabulary this model was trained with:
//!
//! ```text
//! System: you are terse
//!
//! User: hello
//!
//! Assistant: hi
//!
//! User: and again
//!
//! Assistant:
//! ```
//!
//! - **No pseudo-special tokens.** No `<|im_start|>`, no `<|system|>`. Those
//!   are not in the GPT-2 vocabulary, so they would tokenise into several
//!   unrelated pieces the model has never seen in that arrangement. Plain
//!   English role labels tokenise as ordinary words that do occur in dialogue
//!   on the web, which is the only prior this model actually has.
//! - **Blank line between turns**, which is how transcripts are separated in
//!   the training distribution.
//! - **The trailing prime has no space after the colon.** GPT-2 merges a
//!   leading space into the following token -- `" hi"` is one token, `"hi"` is
//!   another -- so ending the prompt with `"Assistant: "` would force the model
//!   to emit a space token and then a word token that almost never follows one.
//!   Ending with `"Assistant:"` lets it emit `" hi"` as the single token the
//!   distribution expects. This is the one detail here that is genuinely about
//!   GPT-2 rather than about taste.
//! - **A trailing assistant message is a continuation**, not a new turn: the
//!   transcript ends with its content and the model carries on from there.
//!   That is what upstream calls prefix continuation and it costs nothing to
//!   support honestly.
//!
//! One function does this, used by the streaming and non-streaming handlers
//! alike, so the two cannot disagree about what was sent to the model.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use super::types::{
    ChatChoice, ChatChunk, ChatChunkChoice, ChatDelta, ChatMessage, ChatRequest, ChatResponse,
    ChatResponseMessage, MessageContent, Usage,
};
use super::{check_model, json_is_empty, new_id, reject_if_set, unix_now, ApiError};
use crate::server::{finish_reason_str, submit_openai, AppState, StreamItem};
use crate::tokenizer::Tokenizer;

/// Roles this adapter can place in a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::System => "System:",
            Role::User => "User:",
            Role::Assistant => "Assistant:",
        }
    }
}

/// Map an incoming role name.
///
/// `developer` is upstream's newer name for `system` and carries the same
/// meaning for a text-only model, so it is mapped rather than refused. `tool`
/// and `function` are refused: representing a tool result in a transcript would
/// require inventing a convention the model was never trained on, and a
/// plausible-looking one is worse than an error.
fn role_of(name: &str) -> Result<Role, ApiError> {
    match name {
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" | "function" => Err(ApiError::unsupported(
            "messages",
            format!(
                "Role '{name}' requires tool calling, which this server does not \
                 implement. Send only system, user and assistant messages."
            ),
        )),
        other => Err(ApiError::invalid(
            format!("Unknown message role '{other}'."),
            Some("messages"),
        )),
    }
}

/// Flatten one message's content to text, refusing anything that is not text.
///
/// An image or audio part is never stringified into the prompt. Feeding a
/// client the completion of `[object]`-shaped filler while reporting success
/// would be indistinguishable from working, which is exactly why it is a 400.
fn content_text(msg: &ChatMessage, index: usize) -> Result<String, ApiError> {
    match &msg.content {
        None | Some(MessageContent::Null) => Ok(String::new()),
        Some(MessageContent::Text(s)) => Ok(s.clone()),
        Some(MessageContent::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if part.kind != "text" {
                    return Err(ApiError::unsupported(
                        "messages",
                        format!(
                            "Message {index} contains a '{}' content part. This server \
                             is text-only; send a string or text content parts.",
                            part.kind
                        ),
                    ));
                }
                out.push_str(part.text.as_deref().unwrap_or(""));
            }
            Ok(out)
        }
    }
}

/// Turn a message list into the exact string submitted to the model.
///
/// Deterministic and total: the same messages always produce the same prompt,
/// which is what makes a seeded chat request reproducible and what lets a test
/// compare this endpoint against the native one.
pub fn serialize(messages: &[ChatMessage]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid(
            "'messages' must contain at least one message.",
            Some("messages"),
        ));
    }

    let mut turns: Vec<(Role, String)> = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        if m.tool_calls.as_ref().is_some_and(|v| !json_is_empty(v))
            || m.function_call.as_ref().is_some_and(|v| !json_is_empty(v))
        {
            return Err(ApiError::unsupported(
                "messages",
                format!("Message {i} carries tool calls, which this server does not implement."),
            ));
        }
        turns.push((role_of(&m.role)?, content_text(m, i)?));
    }

    let mut out = String::new();
    for (role, text) in &turns {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(role.label());
        if !text.is_empty() {
            out.push(' ');
            out.push_str(text);
        }
    }
    // A trailing assistant turn is a continuation of that turn, so it is not
    // followed by a fresh prime.
    if turns.last().map(|(r, _)| *r) != Some(Role::Assistant) {
        out.push_str("\n\nAssistant:");
    }
    Ok(out)
}

/// Everything a chat request must be checked for before it can be served.
struct Prepared {
    prompt: String,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stream: bool,
    include_usage: bool,
}

fn prepare(req: &ChatRequest, st: &AppState) -> Result<Prepared, ApiError> {
    check_model(req.model.as_deref(), &st.model_id)?;

    let messages = req.messages.as_ref().ok_or_else(|| ApiError::missing("messages"))?;

    // Refuse what is not implemented, by name, before doing any work.
    reject_if_set(req.n, |n| *n == 1, "n",
        "This server returns exactly one choice per request.")?;
    reject_if_set(req.top_p, |v| *v == 1.0, "top_p",
        "Crucible samples with top-k; nucleus sampling is not implemented.")?;
    reject_if_set(req.frequency_penalty, |v| *v == 0.0, "frequency_penalty",
        "Repetition penalties are not implemented.")?;
    reject_if_set(req.presence_penalty, |v| *v == 0.0, "presence_penalty",
        "Repetition penalties are not implemented.")?;
    reject_if_set(req.logprobs, |v| !*v, "logprobs",
        "Token log probabilities are not returned by this server.")?;
    reject_if_set(req.top_logprobs, |_| false, "top_logprobs",
        "Token log probabilities are not returned by this server.")?;
    reject_if_set(req.stop.as_ref(), |v| json_is_empty(v), "stop",
        "Stop sequences would have to be matched inside the scheduler across \
         token boundaries; the server does not do that yet, so accepting them \
         would mean generating past the stop and trimming afterwards.")?;
    reject_if_set(req.logit_bias.as_ref(), |v| json_is_empty(v), "logit_bias",
        "Logit bias is not implemented.")?;
    reject_if_set(req.tools.as_ref(), |v| json_is_empty(v), "tools",
        "This model has no tool-calling training; it would never emit a valid call.")?;
    reject_if_set(req.tool_choice.as_ref(), |v| json_is_empty(v), "tool_choice",
        "Tool calling is not implemented.")?;
    reject_if_set(req.functions.as_ref(), |v| json_is_empty(v), "functions",
        "Function calling is not implemented.")?;
    reject_if_set(req.function_call.as_ref(), |v| json_is_empty(v), "function_call",
        "Function calling is not implemented.")?;
    reject_if_set(req.modalities.as_ref(), |v| json_is_empty(v), "modalities",
        "This server is text-only.")?;
    reject_if_set(req.audio.as_ref(), |v| json_is_empty(v), "audio",
        "This server is text-only.")?;
    reject_if_set(req.reasoning_effort.as_ref(), |v| json_is_empty(v), "reasoning_effort",
        "This is not a reasoning model.")?;
    reject_if_set(
        req.response_format.as_ref(),
        |v| json_is_empty(v) || v.get("type").and_then(|t| t.as_str()) == Some("text"),
        "response_format",
        "Structured output requires constrained decoding, which is not implemented.",
    )?;

    // max_tokens is deprecated upstream but still what most clients send.
    // Accept either spelling, refuse a disagreement rather than pick a winner.
    let max_tokens = match (req.max_tokens, req.max_completion_tokens) {
        (Some(a), Some(b)) if a != b => {
            return Err(ApiError::invalid(
                "'max_tokens' and 'max_completion_tokens' were both given with \
                 different values.",
                Some("max_completion_tokens"),
            ))
        }
        (_, Some(b)) => b,
        (Some(a), None) => a,
        (None, None) => st.limits.max_new_tokens.min(256),
    };

    Ok(Prepared {
        prompt: serialize(messages)?,
        max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        seed: req.seed.map(|s| s as u64),
        stream: req.stream.unwrap_or(false),
        include_usage: req
            .stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
    })
}

pub(crate) async fn chat_completions(
    State(st): State<AppState>,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return ApiError::invalid(
                format!("Could not parse request body: {}", e.body_text()),
                None,
            )
            .into_response()
        }
    };

    let prep = match prepare(&req, &st) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };

    let submitted = submit_openai(
        &st,
        &tokenizer,
        &prep.prompt,
        prep.max_tokens,
        prep.temperature,
        prep.top_k,
        prep.seed,
    )
    .await;
    let (mut rx, prompt_tokens) = match submitted {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };

    let id = new_id("chatcmpl");
    let created = unix_now();

    if !prep.stream {
        let mut text = String::new();
        let mut generated = 0usize;
        let mut finish = "length";
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token { text: t, .. } => text.push_str(&t),
                StreamItem::Done { reason, generated: g, tail } => {
                    text.push_str(&tail);
                    generated = g;
                    finish = finish_reason_str(reason);
                    break;
                }
                StreamItem::Failed(e) => return ApiError::server(e).into_response(),
            }
        }
        return Json(ChatResponse {
            id,
            object: "chat.completion",
            created,
            model: st.model_id.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant",
                    content: text,
                    refusal: None,
                },
                logprobs: None,
                finish_reason: finish,
            }],
            usage: Usage::new(prompt_tokens, generated),
        })
        .into_response();
    }

    // Streaming. The role delta goes first, as its own chunk with no content:
    // that is what the schema shows and what clients that build a message
    // incrementally expect to see before any text arrives.
    let model_id = st.model_id.to_string();
    let include_usage = prep.include_usage;
    let head = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_id.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: Some(String::new()),
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: include_usage.then_some(None),
    };

    let stream = async_stream::stream! {
        yield Ok::<Event, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&head).unwrap_or_default()),
        );

        let mut generated = 0usize;
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token { text, .. } => {
                    // Never skip an empty delta silently: a token whose bytes
                    // are half a character decodes to "" and the next one
                    // carries both. Emitting it keeps the chunk count equal to
                    // the token count, which the tests rely on.
                    let chunk = ChatChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta { role: None, content: Some(text) },
                            logprobs: None,
                            finish_reason: None,
                        }],
                        usage: include_usage.then_some(None),
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default()));
                }
                StreamItem::Done { reason, generated: g, tail } => {
                    generated = g;
                    if !tail.is_empty() {
                        let chunk = ChatChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatDelta { role: None, content: Some(tail) },
                                logprobs: None,
                                finish_reason: None,
                            }],
                            usage: include_usage.then_some(None),
                        };
                        yield Ok(Event::default()
                            .data(serde_json::to_string(&chunk).unwrap_or_default()));
                    }
                    let last = ChatChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta::default(),
                            logprobs: None,
                            finish_reason: Some(finish_reason_str(reason)),
                        }],
                        usage: include_usage.then_some(None),
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&last).unwrap_or_default()));
                    break;
                }
                StreamItem::Failed(_) => {
                    // The status line is long gone, so the only honest signal
                    // left is to stop without [DONE]. A client sees a truncated
                    // stream, which is what happened.
                    return;
                }
            }
        }

        if include_usage {
            let tail = ChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: Vec::new(),
                usage: Some(Some(Usage::new(prompt_tokens, generated))),
            };
            yield Ok(Event::default()
                .data(serde_json::to_string(&tail).unwrap_or_default()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            function_call: None,
        }
    }

    #[test]
    fn a_single_user_turn_is_primed_for_the_assistant() {
        let p = serialize(&[msg("user", "hello")]).unwrap();
        assert_eq!(p, "User: hello\n\nAssistant:");
        // No trailing space: GPT-2 merges the leading space into the first
        // generated token, so priming with one would split it.
        assert!(!p.ends_with(' '));
    }

    #[test]
    fn a_full_transcript_round_trips_in_order() {
        let p = serialize(&[
            msg("system", "be terse"),
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "again"),
        ])
        .unwrap();
        assert_eq!(
            p,
            "System: be terse\n\nUser: hi\n\nAssistant: hello\n\nUser: again\n\nAssistant:"
        );
    }

    #[test]
    fn a_trailing_assistant_message_continues_rather_than_reprimes() {
        let p = serialize(&[msg("user", "count"), msg("assistant", "one two")]).unwrap();
        assert_eq!(p, "User: count\n\nAssistant: one two");
        assert!(!p.ends_with("Assistant:"));
    }

    #[test]
    fn developer_is_treated_as_system() {
        let a = serialize(&[msg("developer", "x"), msg("user", "y")]).unwrap();
        let b = serialize(&[msg("system", "x"), msg("user", "y")]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn serialization_is_deterministic() {
        let m = vec![msg("system", "s"), msg("user", "u"), msg("assistant", "a"), msg("user", "v")];
        assert_eq!(serialize(&m).unwrap(), serialize(&m).unwrap());
    }

    #[test]
    fn empty_content_still_produces_a_labelled_turn() {
        let p = serialize(&[msg("system", ""), msg("user", "hi")]).unwrap();
        assert_eq!(p, "System:\n\nUser: hi\n\nAssistant:");
    }

    #[test]
    fn text_content_parts_are_accepted_and_concatenated() {
        let m = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                super::super::types::ContentPart { kind: "text".into(), text: Some("a".into()) },
                super::super::types::ContentPart { kind: "text".into(), text: Some("b".into()) },
            ])),
            name: None,
            tool_calls: None,
            function_call: None,
        };
        assert_eq!(serialize(&[m]).unwrap(), "User: ab\n\nAssistant:");
    }

    #[test]
    fn an_image_part_is_refused_rather_than_stringified() {
        let m = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![super::super::types::ContentPart {
                kind: "image_url".into(),
                text: None,
            }])),
            name: None,
            tool_calls: None,
            function_call: None,
        };
        let e = serialize(&[m]).unwrap_err();
        assert_eq!(e.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(e.body.message.contains("image_url"), "{}", e.body.message);
    }

    #[test]
    fn tool_and_function_roles_are_refused() {
        for role in ["tool", "function"] {
            let e = serialize(&[msg(role, "{}")]).unwrap_err();
            assert_eq!(e.body.code.as_deref(), Some("unsupported_parameter"));
        }
    }

    #[test]
    fn an_unknown_role_is_a_400() {
        let e = serialize(&[msg("wizard", "x")]).unwrap_err();
        assert_eq!(e.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn an_assistant_message_with_tool_calls_is_refused() {
        let mut m = msg("assistant", "");
        m.tool_calls = Some(serde_json::json!([{"id": "call_1"}]));
        assert!(serialize(&[m]).is_err());
    }

    #[test]
    fn an_empty_message_list_is_refused() {
        assert!(serialize(&[]).is_err());
    }

    #[test]
    fn null_content_is_an_empty_turn_not_an_error() {
        let m = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Null),
            name: None,
            tool_calls: None,
            function_call: None,
        };
        assert_eq!(serialize(&[msg("user", "x"), m]).unwrap(), "User: x\n\nAssistant:");
    }

    #[test]
    fn multibyte_content_survives_serialization_unchanged() {
        let p = serialize(&[msg("user", "héllo 世界 🌍")]).unwrap();
        assert!(p.contains("héllo 世界 🌍"));
    }
}
