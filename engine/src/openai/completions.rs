//! Legacy Completions: `POST /v1/completions`.
//!
//! The closest thing to what Crucible natively does -- text in, continuation
//! out -- so the mapping is nearly direct: `prompt` is submitted unchanged,
//! with no template and no role labels.
//!
//! Unlike chat, the upstream schema gives the streamed and non-streamed
//! responses the *same* object shape (`text_completion`), so both handlers
//! build the same struct and differ only in whether `text` holds the whole
//! completion or one delta.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use super::types::{CompletionChoice, CompletionRequest, CompletionResponse, PromptField, Usage};
use super::{check_model, json_is_empty, new_id, reject_if_set, unix_now, ApiError};
use crate::server::{finish_reason_str, submit_compat, AppState, StreamItem};
use crate::tokenizer::Tokenizer;

struct Prepared {
    prompt: String,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stream: bool,
    include_usage: bool,
}

fn prepare(req: &CompletionRequest, st: &AppState) -> Result<Prepared, ApiError> {
    check_model(req.model.as_deref(), &st.model_id)?;

    let prompt = match req.prompt.as_ref() {
        None => return Err(ApiError::missing("prompt")),
        Some(PromptField::Text(s)) => s.clone(),
        // A prompt array is several independent completions in one call, which
        // is a different response shape with several choices. Serving only the
        // first element would look like success and silently drop the rest.
        Some(PromptField::Many(items)) => {
            return Err(ApiError::unsupported(
                "prompt",
                format!(
                    "A batch of {} prompts was given. This server accepts one string \
                     prompt per request; send them as separate requests, which batch \
                     together in the scheduler anyway.",
                    items.len()
                ),
            ))
        }
        Some(PromptField::Other(v)) => {
            return Err(ApiError::invalid(
                format!(
                    "'prompt' must be a string; got {}. Token-id prompts are not \
                     accepted.",
                    match v {
                        serde_json::Value::Number(_) => "a number",
                        serde_json::Value::Bool(_) => "a boolean",
                        serde_json::Value::Object(_) => "an object",
                        _ => "an unsupported value",
                    }
                ),
                Some("prompt"),
            ))
        }
    };

    reject_if_set(req.n, |n| *n == 1, "n",
        "This server returns exactly one choice per request.")?;
    reject_if_set(req.best_of, |n| *n == 1, "best_of",
        "Generating several candidates and ranking them is not implemented.")?;
    reject_if_set(req.echo, |v| !*v, "echo",
        "Echoing the prompt back is not implemented.")?;
    reject_if_set(req.suffix.as_ref(), |s| s.is_empty(), "suffix",
        "Infilling is not implemented; this model was not trained for it.")?;
    reject_if_set(req.top_p, |v| *v == 1.0, "top_p",
        "Crucible samples with top-k; nucleus sampling is not implemented.")?;
    reject_if_set(req.frequency_penalty, |v| *v == 0.0, "frequency_penalty",
        "Repetition penalties are not implemented.")?;
    reject_if_set(req.presence_penalty, |v| *v == 0.0, "presence_penalty",
        "Repetition penalties are not implemented.")?;
    reject_if_set(req.logprobs, |_| false, "logprobs",
        "Token log probabilities are not returned by this server.")?;
    reject_if_set(req.stop.as_ref(), |v| json_is_empty(v), "stop",
        "Stop sequences would have to be matched inside the scheduler across \
         token boundaries; the server does not do that yet.")?;
    reject_if_set(req.logit_bias.as_ref(), |v| json_is_empty(v), "logit_bias",
        "Logit bias is not implemented.")?;

    Ok(Prepared {
        prompt,
        max_tokens: req.max_tokens.unwrap_or(16),
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

pub(crate) async fn completions(
    State(st): State<AppState>,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    body: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
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
    let (mut rx, prompt_tokens) = match submitted {
        Ok(v) => v,
        Err(e) => return ApiError::from(e).into_response(),
    };

    let id = new_id("cmpl");
    let created = unix_now();
    let model_id = st.model_id.to_string();

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
        return Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model: model_id,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: Some(finish),
            }],
            usage: Some(Some(Usage::new(prompt_tokens, generated))),
        })
        .into_response();
    }

    let include_usage = prep.include_usage;
    let stream = async_stream::stream! {
        let mut generated = 0usize;
        // One helper rather than four near-identical literals: the streamed
        // and final objects differ only in `text` and `finish_reason`.
        macro_rules! chunk {
            ($text:expr, $finish:expr) => {
                CompletionResponse {
                    id: id.clone(),
                    object: "text_completion",
                    created,
                    model: model_id.clone(),
                    choices: vec![CompletionChoice {
                        text: $text,
                        index: 0,
                        logprobs: None,
                        finish_reason: $finish,
                    }],
                    usage: include_usage.then_some(None),
                }
            };
        }

        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token { text, .. } => {
                    yield Ok::<Event, std::convert::Infallible>(Event::default()
                        .data(serde_json::to_string(&chunk!(text, None)).unwrap_or_default()));
                }
                StreamItem::Done { reason, generated: g, tail } => {
                    generated = g;
                    let last = chunk!(tail, Some(finish_reason_str(reason)));
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&last).unwrap_or_default()));
                    break;
                }
                StreamItem::Failed(_) => return,
            }
        }

        if include_usage {
            let mut tail = chunk!(String::new(), None);
            tail.choices.clear();
            tail.usage = Some(Some(Usage::new(prompt_tokens, generated)));
            yield Ok(Event::default()
                .data(serde_json::to_string(&tail).unwrap_or_default()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}
