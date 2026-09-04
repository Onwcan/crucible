//! `POST /v1/messages` and `POST /v1/messages/count_tokens`.
//!
//! # Serialization
//!
//! Anthropic keeps the system prompt at the top level rather than inside the
//! message list; both become turns for `chat_template`, which produces the
//! identical prompt the OpenAI adapter produces for the same conversation.
//! That equivalence is the point of the shared module and is tested across the
//! two surfaces.
//!
//! # Sampling
//!
//! There is no standard sampling parameter to map. `temperature`, `top_p` and
//! `top_k` are absent from the current SDK's Messages types entirely, so this
//! endpoint is **greedy by default and takes no standard knob**. Crucible's
//! sampler is still reachable, through fields named `crucible_temperature`,
//! `crucible_top_k` and `crucible_seed` -- prefixed so nobody can mistake them
//! for Anthropic fields, and never required: the official SDK works without
//! ever sending one.
//!
//! A request that *does* carry `temperature`, `top_p` or `top_k` is refused
//! rather than ignored, with a message pointing at the extensions. Ignoring
//! them would mean a caller who asked for sampling silently received greedy
//! output.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use super::types::{
    ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent, CountTokensRequest,
    CountTokensResponse, InMessage, MessageContent, MessageDeltaBody, MessageDeltaEvent,
    MessageDeltaUsage, MessageResponse, MessageStartEvent, MessageStopEvent, MessagesRequest,
    SystemField, TextBlock, TextDelta, Usage,
};
use super::{check_version, new_message_id, new_request_id, ApiError, REQUEST_ID_HEADER};
use crate::chat_template::{self, Role, Turn};
use crate::server::{submit_compat, AppState, StreamItem};
use crate::tokenizer::Tokenizer;

/// The stop reason, in Anthropic's vocabulary.
///
/// `max_tokens` and nothing else. The runtime stops when the token budget runs
/// out: this checkpoint has no trained end-of-turn token, so `end_turn` would
/// be a claim about a semantic the model never acquired, and `stop_sequence`,
/// `tool_use`, `pause_turn` and `refusal` all describe machinery that does not
/// exist here.
fn stop_reason(reason: crate::runtime::FinishReason) -> &'static str {
    use crate::runtime::FinishReason as F;
    match reason {
        // A cancelled request has already lost its connection, so its stop
        // reason is never delivered; the arm exists to keep this exhaustive
        // rather than to describe something a client can observe.
        F::Length | F::Cancelled => "max_tokens",
    }
}

fn blocks_to_text(
    blocks: &[super::types::ContentBlockIn],
    where_: &str,
) -> Result<String, ApiError> {
    let mut out = String::new();
    for b in blocks {
        if b.kind != "text" {
            return Err(ApiError::invalid(format!(
                "{where_} contains a {:?} content block. This server is text-only; \
                 send a string or text blocks.",
                b.kind
            )));
        }
        out.push_str(b.text.as_deref().unwrap_or(""));
    }
    Ok(out)
}

fn message_text(m: &InMessage, index: usize) -> Result<String, ApiError> {
    match &m.content {
        None => Ok(String::new()),
        Some(MessageContent::Text(s)) => Ok(s.clone()),
        Some(MessageContent::Blocks(b)) => blocks_to_text(b, &format!("messages[{index}]")),
    }
}

fn system_text(system: Option<&SystemField>) -> Result<Option<String>, ApiError> {
    match system {
        None => Ok(None),
        Some(SystemField::Text(s)) => Ok(Some(s.clone())),
        Some(SystemField::Blocks(b)) => Ok(Some(blocks_to_text(b, "system")?)),
    }
}

/// Build the conversation this request describes.
///
/// Shared by `/v1/messages` and `/v1/messages/count_tokens`, which is what lets
/// the count endpoint promise the exact number the generate endpoint will later
/// report rather than an estimate of it.
pub fn conversation(
    system: Option<&SystemField>,
    messages: &[InMessage],
) -> Result<Vec<Turn>, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid(
            "messages: at least one message is required.",
        ));
    }
    let mut turns = Vec::with_capacity(messages.len() + 1);
    if let Some(text) = system_text(system)? {
        turns.push(Turn::new(Role::System, text));
    }
    for (i, m) in messages.iter().enumerate() {
        let role = match m.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => {
                return Err(ApiError::invalid(format!(
                    "messages[{i}].role: 'system' is not a message role in this API. \
                     Use the top-level 'system' parameter."
                )))
            }
            other => {
                return Err(ApiError::invalid(format!(
                    "messages[{i}].role: {other:?} is not supported. This server \
                     accepts 'user' and 'assistant'."
                )))
            }
        };
        turns.push(Turn::new(role, message_text(m, i)?));
    }
    Ok(turns)
}

/// One typed SSE event.
///
/// A free function rather than a macro used inside the `stream!` block:
/// `async_stream` is a proc macro that rewrites `yield` where it can see it,
/// and a `yield` nested inside a `macro_rules!` body is not somewhere it can.
fn sse(name: &'static str, value: &impl serde::Serialize) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default()
        .event(name)
        .data(serde_json::to_string(value).unwrap_or_default()))
}

/// Refuse a field that is present and not at its no-op value.
fn refuse_if_set(
    value: Option<&serde_json::Value>,
    name: &str,
    detail: &str,
) -> Result<(), ApiError> {
    if let Some(v) = value {
        let empty = matches!(v, serde_json::Value::Null)
            || v.as_array().is_some_and(|a| a.is_empty())
            || v.as_object().is_some_and(|o| o.is_empty());
        if !empty {
            return Err(ApiError::invalid(format!("{name}: {detail}")));
        }
    }
    Ok(())
}

struct Prepared {
    prompt: String,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stream: bool,
}

fn prepare(req: &MessagesRequest, st: &AppState) -> Result<Prepared, ApiError> {
    if let Some(m) = req.model.as_deref() {
        if m != &*st.model_id {
            return Err(ApiError::not_found(format!(
                "model: {m}. This server serves one model: {}.",
                st.model_id
            )));
        }
    } else {
        return Err(ApiError::invalid("model: this field is required."));
    }

    let max_tokens = match req.max_tokens {
        None => return Err(ApiError::invalid("max_tokens: this field is required.")),
        Some(v) if v < 1 => {
            return Err(ApiError::invalid(format!(
                "max_tokens: {v} is less than the minimum of 1."
            )))
        }
        Some(v) => v as usize,
    };

    // Sampling controls the current API does not define. Refused with a route
    // to the extension rather than ignored.
    for (present, name) in [
        (req.temperature.is_some(), "temperature"),
        (req.top_p.is_some(), "top_p"),
        (req.top_k.is_some(), "top_k"),
    ] {
        if present {
            return Err(ApiError::invalid(format!(
                "{name}: not a parameter of this API. This server decodes greedily \
                 by default; to sample, use the Crucible extensions \
                 crucible_temperature, crucible_top_k and crucible_seed."
            )));
        }
    }

    refuse_if_set(req.stop_sequences.as_ref().map(|s| serde_json::json!(s)).as_ref(),
        "stop_sequences",
        "stop sequences are not implemented. Matching them would have to happen \
         inside the scheduler across token boundaries; generating past the \
         sequence and trimming afterwards would leave the KV cache and token \
         accounting describing text you never received.")?;
    refuse_if_set(req.tools.as_ref(), "tools",
        "tool use is not implemented. This model has no tool-calling training.")?;
    refuse_if_set(req.tool_choice.as_ref(), "tool_choice",
        "tool use is not implemented.")?;
    refuse_if_set(req.thinking.as_ref(), "thinking",
        "extended thinking is not implemented. This is not a reasoning model.")?;
    refuse_if_set(req.output_config.as_ref(), "output_config",
        "structured output and effort levels are not implemented.")?;
    refuse_if_set(req.container.as_ref(), "container",
        "code execution containers are not implemented.")?;
    refuse_if_set(req.cache_control.as_ref(), "cache_control",
        "prompt caching is not implemented; this server has no prompt cache.")?;

    let messages = req
        .messages
        .as_ref()
        .ok_or_else(|| ApiError::invalid("messages: this field is required."))?;
    let turns = conversation(req.system.as_ref(), messages)?;

    Ok(Prepared {
        prompt: chat_template::serialize(&turns),
        max_tokens,
        temperature: req.crucible_temperature,
        top_k: req.crucible_top_k,
        seed: req.crucible_seed,
        stream: req.stream.unwrap_or(false),
    })
}

fn with_request_id(id: &str, response: Response) -> Response {
    let mut response = response;
    if let Ok(v) = axum::http::HeaderValue::from_str(id) {
        response.headers_mut().insert(
            axum::http::HeaderName::from_static(REQUEST_ID_HEADER),
            v,
        );
    }
    response
}

pub(crate) async fn messages(
    State(st): State<AppState>,
    headers: HeaderMap,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    body: Result<Json<MessagesRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = check_version(&headers) {
        return e.into_response();
    }
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return ApiError::invalid(format!("Could not parse request body: {}", e.body_text()))
                .into_response()
        }
    };
    let prep = match prepare(&req, &st) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };

    let submitted = submit_compat(
        &st,
        &tokenizer,
        &prep.prompt,
        prep.max_tokens,
        prep.temperature,
        prep.top_k,
        prep.seed,
    )
    .await;
    let (mut rx, input_tokens) = match submitted {
        Ok(v) => v,
        Err(e) => return ApiError::from(e).into_response(),
    };

    let id = new_message_id();
    let request_id = new_request_id();
    let model = st.model_id.to_string();

    if !prep.stream {
        let mut text = String::new();
        let mut output_tokens = 0usize;
        let mut stop = "max_tokens";
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token { text: t, .. } => text.push_str(&t),
                StreamItem::Done { reason, generated, tail } => {
                    text.push_str(&tail);
                    output_tokens = generated;
                    stop = stop_reason(reason);
                    break;
                }
                StreamItem::Failed(e) => return ApiError::api_error(e).into_response(),
            }
        }
        return with_request_id(
            &request_id,
            Json(MessageResponse {
                id,
                kind: "message",
                role: "assistant",
                model,
                content: vec![TextBlock::new(text)],
                stop_reason: Some(stop),
                stop_sequence: None,
                usage: Usage {
                    input_tokens,
                    output_tokens,
                },
            })
            .into_response(),
        );
    }

    // Streaming. Anthropic's protocol is a sequence of *typed* events, not
    // OpenAI chunks, and it has no [DONE] sentinel: message_stop terminates it.
    let head = MessageStartEvent {
        kind: "message_start",
        message: MessageResponse {
            id: id.clone(),
            kind: "message",
            role: "assistant",
            model: model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        },
    };

    let stream = async_stream::stream! {
        yield sse("message_start", &head);
        yield sse("content_block_start", &ContentBlockStartEvent {
            kind: "content_block_start",
            index: 0,
            content_block: TextBlock::new(""),
        });

        let mut output_tokens = 0usize;
        let mut stop = "max_tokens";
        let mut finished = false;
        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token { text, .. } => {
                    // Emitted even when empty: a token whose bytes are half a
                    // character decodes to "" and the next one carries both.
                    // The incremental decoder owns that, not this loop.
                    yield sse("content_block_delta", &ContentBlockDeltaEvent {
                        kind: "content_block_delta",
                        index: 0,
                        delta: TextDelta { kind: "text_delta", text },
                    });
                }
                StreamItem::Done { reason, generated, tail } => {
                    if !tail.is_empty() {
                        yield sse("content_block_delta", &ContentBlockDeltaEvent {
                            kind: "content_block_delta",
                            index: 0,
                            delta: TextDelta { kind: "text_delta", text: tail },
                        });
                    }
                    output_tokens = generated;
                    stop = stop_reason(reason);
                    finished = true;
                    break;
                }
                StreamItem::Failed(e) => {
                    // The status line is long gone. Anthropic's protocol has an
                    // error event for exactly this, so the client learns the
                    // stream ended badly rather than seeing it simply stop.
                    yield sse("error", &serde_json::json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": e},
                    }));
                    return;
                }
            }
        }
        if !finished {
            // The receiver went away mid-generation: the client is gone, so
            // there is nobody to send a terminator to.
            return;
        }

        yield sse("content_block_stop", &ContentBlockStopEvent {
            kind: "content_block_stop",
            index: 0,
        });
        yield sse("message_delta", &MessageDeltaEvent {
            kind: "message_delta",
            delta: MessageDeltaBody { stop_reason: Some(stop), stop_sequence: None },
            usage: MessageDeltaUsage { input_tokens, output_tokens },
        });
        yield sse("message_stop", &MessageStopEvent { kind: "message_stop" });
    };

    with_request_id(
        &request_id,
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response(),
    )
}

/// `POST /v1/messages/count_tokens`.
///
/// Cheap and exact, because this server owns the tokenizer that will process
/// the prompt. It runs the same conversation builder, the same template and the
/// same tokenizer `/v1/messages` runs, so the number it returns is the number a
/// later `usage.input_tokens` will report -- not an estimate of it. There is
/// deliberately no second counting implementation to drift.
pub(crate) async fn count_tokens(
    State(st): State<AppState>,
    headers: HeaderMap,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    body: Result<Json<CountTokensRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(e) = check_version(&headers) {
        return e.into_response();
    }
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return ApiError::invalid(format!("Could not parse request body: {}", e.body_text()))
                .into_response()
        }
    };
    if let Some(m) = req.model.as_deref() {
        if m != &*st.model_id {
            return ApiError::not_found(format!(
                "model: {m}. This server serves one model: {}.",
                st.model_id
            ))
            .into_response();
        }
    }
    if let Err(e) = refuse_if_set(req.tools.as_ref(), "tools", "tool use is not implemented.")
        .and_then(|_| {
            refuse_if_set(req.tool_choice.as_ref(), "tool_choice", "tool use is not implemented.")
        })
        .and_then(|_| {
            refuse_if_set(
                req.thinking.as_ref(),
                "thinking",
                "extended thinking is not implemented.",
            )
        })
    {
        return e.into_response();
    }

    let messages = match req.messages.as_ref() {
        Some(m) => m,
        None => return ApiError::invalid("messages: this field is required.").into_response(),
    };
    let turns = match conversation(req.system.as_ref(), messages) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    let prompt = chat_template::serialize(&turns);
    let input_tokens = match tokenizer.encode(&prompt) {
        Ok(ids) => ids.len(),
        Err(e) => {
            return ApiError::invalid(format!("Could not tokenise the prompt: {e}")).into_response()
        }
    };
    with_request_id(
        &new_request_id(),
        Json(CountTokensResponse { input_tokens }).into_response(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, text: &str) -> InMessage {
        InMessage {
            role: role.into(),
            content: Some(MessageContent::Text(text.into())),
        }
    }

    fn prompt_of(system: Option<SystemField>, messages: &[InMessage]) -> String {
        chat_template::serialize(&conversation(system.as_ref(), messages).unwrap())
    }

    #[test]
    fn a_top_level_system_becomes_the_leading_turn() {
        let p = prompt_of(
            Some(SystemField::Text("be terse".into())),
            &[msg("user", "hi")],
        );
        assert_eq!(p, "System: be terse\n\nUser: hi\n\nAssistant:");
    }

    #[test]
    fn the_prompt_matches_what_the_openai_adapter_builds() {
        // The reason chat_template exists. Anthropic carries the system prompt
        // at the top level and OpenAI carries it in the message list; the model
        // must not be able to tell which client asked.
        let anthropic = prompt_of(
            Some(SystemField::Text("be terse".into())),
            &[msg("user", "hi"), msg("assistant", "hello"), msg("user", "again")],
        );
        let openai = chat_template::serialize(&[
            Turn::new(Role::System, "be terse"),
            Turn::new(Role::User, "hi"),
            Turn::new(Role::Assistant, "hello"),
            Turn::new(Role::User, "again"),
        ]);
        assert_eq!(anthropic, openai);
    }

    #[test]
    fn system_blocks_are_concatenated_like_a_string() {
        let blocks = SystemField::Blocks(vec![
            super::super::types::ContentBlockIn { kind: "text".into(), text: Some("a".into()) },
            super::super::types::ContentBlockIn { kind: "text".into(), text: Some("b".into()) },
        ]);
        assert_eq!(
            prompt_of(Some(blocks), &[msg("user", "hi")]),
            prompt_of(Some(SystemField::Text("ab".into())), &[msg("user", "hi")])
        );
    }

    #[test]
    fn a_trailing_assistant_message_is_a_prefill() {
        let p = prompt_of(None, &[msg("user", "count"), msg("assistant", "one two")]);
        assert_eq!(p, "User: count\n\nAssistant: one two");
    }

    #[test]
    fn consecutive_same_role_messages_are_allowed() {
        // The current API permits them; nothing here merges or reorders.
        let p = prompt_of(None, &[msg("user", "one"), msg("user", "two")]);
        assert_eq!(p, "User: one\n\nUser: two\n\nAssistant:");
    }

    #[test]
    fn a_system_role_inside_messages_points_at_the_top_level_field() {
        let e = conversation(None, &[msg("system", "x")]).unwrap_err();
        assert!(e.message.contains("top-level 'system'"), "{}", e.message);
    }

    #[test]
    fn an_unknown_role_is_refused() {
        assert!(conversation(None, &[msg("tool", "{}")]).is_err());
        assert!(conversation(None, &[msg("wizard", "x")]).is_err());
    }

    #[test]
    fn an_empty_message_list_is_refused() {
        assert!(conversation(None, &[]).is_err());
    }

    #[test]
    fn a_non_text_block_is_refused_rather_than_stringified() {
        let m = InMessage {
            role: "user".into(),
            content: Some(MessageContent::Blocks(vec![
                super::super::types::ContentBlockIn { kind: "image".into(), text: None },
            ])),
        };
        let e = conversation(None, &[m]).unwrap_err();
        assert!(e.message.contains("image"), "{}", e.message);
        assert!(e.message.contains("text-only"), "{}", e.message);
    }

    #[test]
    fn multibyte_content_survives() {
        let p = prompt_of(None, &[msg("user", "héllo 世界 🌍")]);
        assert!(p.contains("héllo 世界 🌍"));
    }

    #[test]
    fn the_stop_reason_is_always_max_tokens() {
        use crate::runtime::FinishReason;
        assert_eq!(stop_reason(FinishReason::Length), "max_tokens");
        assert_eq!(stop_reason(FinishReason::Cancelled), "max_tokens");
    }
}
