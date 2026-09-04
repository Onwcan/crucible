//! Anthropic-compatible Messages API, as a second adapter over the same service.
//!
//! # What this is
//!
//! A third wire protocol on one runtime:
//!
//! ```text
//! native handlers    ─┐
//! OpenAI adapter     ─┼─> submit_compat ─> one queue ─> one inference thread
//! Anthropic adapter  ─┘                                  one scheduler, one GPU
//! ```
//!
//! It is deliberately *not* built on the OpenAI adapter. The two protocols
//! disagree about where the system prompt lives, whether `max_tokens` is
//! required, how content is shaped, and what streaming looks like; routing one
//! through the other would make OpenAI's DTOs the interface Anthropic is
//! written against, and every upstream divergence a refactor. What they truly
//! share -- turning a conversation into prompt text -- lives in `chat_template`
//! and is shared *there*, which is what makes the two surfaces produce
//! identical prompts for equivalent conversations.
//!
//! # What this is not
//!
//! Claude. The checkpoint behind this endpoint is a 120M-parameter base
//! language model trained on FineWeb-Edu with the GPT-2 tokenizer. It is not
//! instruction-tuned, not RLHF-trained, not tool-trained, and not a reasoning
//! model. This is protocol compatibility so that Anthropic-shaped clients have
//! something to talk to; it says nothing about what the model can do.
//!
//! Verified against the official `anthropic` Python SDK, version 1.3.0.

pub mod messages;
pub mod types;

use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// The API version this adapter implements.
///
/// What the current SDK sends on every request, and the only value whose
/// semantics are implemented here. Accepting a newer version string would be
/// claiming to implement whatever changed in it.
pub const SUPPORTED_VERSION: &str = "2023-06-01";

/// The header carrying it.
pub const VERSION_HEADER: &str = "anthropic-version";

/// The header the SDK reads a request id back from.
pub const REQUEST_ID_HEADER: &str = "request-id";

// --- errors -----------------------------------------------------------------

/// Anthropic's error envelope: `{type: "error", error: {type, message}, request_id}`.
///
/// A different shape from OpenAI's, and built separately for that reason. The
/// `type` values are drawn from the union the SDK can actually parse --
/// `invalid_request_error`, `not_found_error`, `rate_limit_error`,
/// `overloaded_error`, `api_error` -- so an error never arrives as an
/// unrecognised variant.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorEnvelope {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub error: ApiErrorBody,
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub request_id: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            request_id: new_request_id(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", message)
    }

    /// 529, which is Anthropic's overload status and what the SDK maps to
    /// `OverloadedError`.
    ///
    /// Not 429: a full queue here is the server being busy, not a caller
    /// exceeding a quota, and this server has no quotas. The OpenAI surface
    /// answers the same condition with 429 because that is what *its* clients
    /// retry on -- one failure, two vocabularies.
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::from_u16(529).expect("529 is a valid status code"),
            "overloaded_error",
            message,
        )
    }

    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", message)
    }

    pub fn envelope(&self) -> ApiErrorEnvelope {
        ApiErrorEnvelope {
            kind: "error",
            error: ApiErrorBody {
                kind: self.kind,
                message: self.message.clone(),
            },
            request_id: self.request_id.clone(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&self.request_id) {
            headers.insert(HeaderName::from_static("request-id"), v);
        }
        (self.status, headers, Json(self.envelope())).into_response()
    }
}

/// Render a submission failure into this protocol's envelope.
impl From<crate::server::SubmitError> for ApiError {
    fn from(e: crate::server::SubmitError) -> Self {
        use crate::server::SubmitError as S;
        let msg = e.message().to_string();
        match e {
            S::Tokenise(_) | S::Invalid(_) | S::TooLarge(_) | S::Sampling(_) => {
                ApiError::invalid(msg)
            }
            S::QueueFull(_) => ApiError::overloaded(msg),
            S::Unavailable(_) => ApiError::api_error(msg),
        }
    }
}

// --- identifiers ------------------------------------------------------------

static SEQ: AtomicU64 = AtomicU64::new(0);

fn hex_id(prefix: &str) -> String {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let base = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!(
        "{prefix}{:016x}{:08x}",
        base ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        n as u32
    )
}

/// `msg_...`, stable across every event of one stream.
///
/// Derived from a counter and the clock, never from a scheduler slot: slots are
/// reused by `swap_remove` between decode steps, so a slot-derived id would
/// eventually be handed to two unrelated conversations. Nothing here exposes a
/// page id, a pointer or any other internal state.
pub fn new_message_id() -> String {
    hex_id("msg_")
}

/// `req_...`, for the `request-id` header and error bodies.
pub fn new_request_id() -> String {
    hex_id("req_")
}

// --- version negotiation ----------------------------------------------------

/// Check the `anthropic-version` header.
///
/// Absent is accepted: `curl` users have no reason to send it and the endpoint
/// behaves identically without it. A *different* version is refused, because
/// answering a 2099 request with 2023-06-01 semantics would be claiming to
/// implement changes this adapter has never seen.
pub fn check_version(headers: &HeaderMap) -> Result<(), ApiError> {
    match headers.get(VERSION_HEADER).and_then(|v| v.to_str().ok()) {
        None => Ok(()),
        Some(v) if v == SUPPORTED_VERSION => Ok(()),
        Some(v) => Err(ApiError::invalid(format!(
            "Unsupported anthropic-version: {v:?}. This server implements \
             {SUPPORTED_VERSION}."
        ))),
    }
}

/// Whether a request is speaking Anthropic.
///
/// Used only to disambiguate `/v1/models`, which both protocols claim with
/// incompatible response schemas. The current Anthropic SDK sends this header
/// on every request and the OpenAI SDK never does, so the signal is reliable;
/// absence keeps the existing OpenAI behaviour, which is what makes this
/// non-breaking.
pub fn wants_anthropic(headers: &HeaderMap) -> bool {
    headers.contains_key(VERSION_HEADER)
}

// --- models -----------------------------------------------------------------

fn model_info(st: &crate::server::AppState) -> types::ModelInfo {
    types::ModelInfo {
        id: st.model_id.to_string(),
        kind: "model",
        display_name: st.model_id.to_string(),
        created_at: rfc3339(st.model_created),
        max_tokens: st.limits.max_new_tokens,
        max_input_tokens: st.limits.max_prompt_tokens,
    }
}

/// Anthropic's models list, served only when the version header says so.
pub(crate) async fn models_list(State(st): State<crate::server::AppState>) -> Response {
    let info = model_info(&st);
    let id = info.id.clone();
    Json(types::ModelPage {
        data: vec![info],
        has_more: false,
        first_id: Some(id.clone()),
        last_id: Some(id),
    })
    .into_response()
}

pub(crate) async fn models_get(st: &crate::server::AppState, model: &str) -> Response {
    if model != &*st.model_id {
        return ApiError::not_found(format!(
            "model: {model}. This server serves one model: {}.",
            st.model_id
        ))
        .into_response();
    }
    Json(model_info(st)).into_response()
}

/// Unix seconds as an RFC 3339 UTC timestamp.
///
/// Hand-rolled rather than pulling in a date crate for one field. Civil-date
/// arithmetic from the days-since-epoch, which is exact and needs no timezone
/// database because the output is always UTC.
pub fn rfc3339(unix: i64) -> String {
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days, shifted to a March-based year so the
    // leap day lands at the end and needs no special case.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_str(value).unwrap(),
        );
        h
    }

    #[test]
    fn the_supported_version_and_an_absent_header_are_both_fine() {
        assert!(check_version(&HeaderMap::new()).is_ok());
        assert!(check_version(&headers_with(SUPPORTED_VERSION)).is_ok());
    }

    #[test]
    fn an_unknown_version_is_refused_rather_than_assumed() {
        let e = check_version(&headers_with("2099-01-01")).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert_eq!(e.kind, "invalid_request_error");
        assert!(e.message.contains("2023-06-01"), "{}", e.message);
    }

    #[test]
    fn the_version_header_is_what_distinguishes_the_two_models_apis() {
        assert!(wants_anthropic(&headers_with(SUPPORTED_VERSION)));
        assert!(!wants_anthropic(&HeaderMap::new()));
    }

    #[test]
    fn ids_are_prefixed_unique_and_free_of_internal_state() {
        let a = new_message_id();
        let b = new_message_id();
        assert_ne!(a, b);
        assert!(a.starts_with("msg_"), "{a}");
        assert!(new_request_id().starts_with("req_"));
        assert!(a["msg_".len()..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn many_ids_do_not_collide() {
        let ids: std::collections::HashSet<String> =
            (0..10_000).map(|_| new_message_id()).collect();
        assert_eq!(ids.len(), 10_000);
    }

    #[test]
    fn the_error_envelope_matches_the_published_shape() {
        let e = ApiError::invalid("nope");
        let v = serde_json::to_value(e.envelope()).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "nope");
        assert!(v["request_id"].as_str().unwrap().starts_with("req_"));
    }

    #[test]
    fn a_full_queue_is_529_here_and_not_429() {
        // The OpenAI surface answers the same condition with 429. Each client
        // library retries on its own protocol's status.
        let e = ApiError::overloaded("busy");
        assert_eq!(e.status.as_u16(), 529);
        assert_eq!(e.kind, "overloaded_error");
    }

    #[test]
    fn rfc3339_matches_known_timestamps() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_000_000_000), "2001-09-09T01:46:40Z");
        assert_eq!(rfc3339(1_700_000_000), "2023-11-14T22:13:20Z");
        // Leap day, which the March-based civil arithmetic exists to get right.
        assert_eq!(rfc3339(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }
}
