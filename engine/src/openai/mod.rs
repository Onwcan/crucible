//! OpenAI-compatible API, as an adapter over the native service.
//!
//! # What this is
//!
//! A translation layer, not a second engine. An OpenAI-shaped request is parsed
//! here, converted into the same job the native `/v1/generate` endpoint builds,
//! and handed to the same bounded queue, the same inference thread and the same
//! continuous-batching scheduler. The token stream that comes back is reshaped
//! into OpenAI's chunk format on the way out.
//!
//! ```text
//! OpenAI request -> DTO -> native job -> [one runtime, one scheduler] -> tokens -> OpenAI chunks
//! ```
//!
//! There is no path from these handlers to `GpuModel`, no second scheduler and
//! no separate queue. A request arriving here batches with native requests and
//! with the TUI, and is subject to the same backpressure and the same
//! cancellation.
//!
//! # What this is not
//!
//! A claim to implement the OpenAI API. It implements a subset: the parts that
//! correspond to something Crucible actually does. Everything else is refused
//! by name with a 4xx rather than accepted and quietly ignored, because a
//! server that accepts `top_p` and then samples with top-k has told the client
//! something false about its own output.
//!
//! Schemas follow the official OpenAI OpenAPI specification, version 2.3.0.

pub mod chat;
pub mod completions;
pub mod types;

use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Default public identifier for the served checkpoint.
///
/// Not the model directory: exposing a filesystem path as a model id leaks
/// server layout into a field clients paste into config files. `120m` is what
/// this repository has called this checkpoint since it was trained, so it is
/// the stable name rather than a new one invented here. Serving a different
/// checkpoint should pass `--model-id`.
pub const DEFAULT_MODEL_ID: &str = "crucible-120m";

/// Reject an id that would be confusing or unsafe to echo back.
pub fn validate_model_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("model id must not be empty".into());
    }
    if id.contains('/') || id.contains('\\') {
        return Err(format!(
            "model id {id:?} looks like a path; ids are published to clients and \
             must not describe the server's filesystem"
        ));
    }
    Ok(())
}

// --- errors -----------------------------------------------------------------

/// OpenAI's error envelope.
///
/// All four inner fields are required by the schema, so they are always
/// present; `param` and `code` are null when they do not apply. Rust error
/// text never reaches the client verbatim -- messages here are written for
/// somebody debugging a client, not for somebody reading a backtrace.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ApiErrorBody,
}

impl ApiError {
    fn new(
        status: StatusCode,
        kind: &'static str,
        message: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            status,
            body: ApiErrorBody {
                message: message.into(),
                kind,
                param: param.map(str::to_string),
                code: code.map(str::to_string),
            },
        }
    }

    /// A request that is malformed or asks for something impossible.
    pub fn invalid(message: impl Into<String>, param: Option<&str>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
            param,
            Some("invalid_value"),
        )
    }

    /// A required field is absent.
    pub fn missing(param: &str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Missing required parameter: '{param}'."),
            Some(param),
            Some("missing_required_parameter"),
        )
    }

    /// A parameter Crucible does not implement.
    ///
    /// Deliberately 400 rather than a silent default. The alternative --
    /// accepting `top_p` and sampling with top-k anyway -- produces output that
    /// does not match what the client asked for and gives it no way to find
    /// out.
    pub fn unsupported(param: &str, detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Unsupported parameter: '{param}'. {}", detail.into()),
            Some(param),
            Some("unsupported_parameter"),
        )
    }

    pub fn model_not_found(requested: &str, served: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!(
                "The model '{requested}' does not exist. This server serves one \
                 model: '{served}'."
            ),
            Some("model"),
            Some("model_not_found"),
        )
    }

    pub fn context_length(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
            Some("max_tokens"),
            Some("context_length_exceeded"),
        )
    }

    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            message,
            None,
            Some("rate_limit_exceeded"),
        )
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            message,
            None,
            Some("service_unavailable"),
        )
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            message,
            None,
            None,
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorEnvelope { error: self.body }),
        )
            .into_response()
    }
}

// --- identifiers ------------------------------------------------------------

static SEQ: AtomicU64 = AtomicU64::new(0);

/// `chatcmpl-...` / `cmpl-...`, unique within this process.
///
/// Derived from a counter mixed with process start time, not from a scheduler
/// slot or a request's address: slots are reused by `swap_remove` between
/// steps, so anything keyed on one would eventually hand two different
/// conversations the same id. Uniqueness only has to hold for one process --
/// these are correlation handles for a local server, not distributed ids.
pub fn new_id(prefix: &str) -> String {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let base = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // 96 bits of hex, the same shape upstream ids have.
    format!("{prefix}-{:016x}{:08x}", base ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15), n as u32)
}

pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// --- models -----------------------------------------------------------------

/// `GET /v1/models`.
///
/// Exactly one entry, because this process serves exactly one checkpoint.
/// Listing anything else -- aliases, sizes the server does not have, names that
/// look like OpenAI models -- would make a client's model picker offer things
/// that 404 on first use.
pub(crate) async fn models_list(State(st): State<crate::server::AppState>) -> Response {
    Json(types::ModelList {
        object: "list",
        data: vec![types::ModelObject::new(&st.model_id, st.model_created)],
    })
    .into_response()
}

/// `GET /v1/models/{model}`.
pub(crate) async fn models_get(
    State(st): State<crate::server::AppState>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Response {
    if model != *st.model_id {
        return ApiError::model_not_found(&model, &st.model_id).into_response();
    }
    Json(types::ModelObject::new(&st.model_id, st.model_created)).into_response()
}

// --- shared request checks --------------------------------------------------

/// Check the model field against the one model this process serves.
///
/// Absent is accepted even though the upstream schema marks it required: the
/// native service has always been single-model and rejecting an omitted field
/// would break `curl` users for no safety gain. A *different* model is a 404,
/// never a silent substitution -- a client that asked for gpt-4 and received
/// 120M-parameter output would have no way to tell.
pub fn check_model(requested: Option<&str>, served: &str) -> Result<(), ApiError> {
    match requested {
        None => Ok(()),
        Some(m) if m == served => Ok(()),
        Some(m) => Err(ApiError::model_not_found(m, served)),
    }
}

/// Reject a parameter that is present and not at its no-op value.
pub fn reject_if_set<T>(
    value: Option<T>,
    is_noop: impl Fn(&T) -> bool,
    param: &str,
    detail: &str,
) -> Result<(), ApiError> {
    match value {
        Some(v) if !is_noop(&v) => Err(ApiError::unsupported(param, detail)),
        _ => Ok(()),
    }
}

/// True when a JSON value is absent in the "not asked for" sense: null, an
/// empty array, or an empty object.
pub fn json_is_empty(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Object(o) => o.is_empty(),
        serde_json::Value::String(s) => s.is_empty() || s == "none",
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_shaped_like_the_upstream_ones() {
        let a = new_id("chatcmpl");
        let b = new_id("chatcmpl");
        assert_ne!(a, b);
        assert!(a.starts_with("chatcmpl-"), "{a}");
        assert_eq!(a.len(), "chatcmpl-".len() + 24);
        assert!(a["chatcmpl-".len()..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn many_ids_do_not_collide() {
        let ids: std::collections::HashSet<String> =
            (0..10_000).map(|_| new_id("cmpl")).collect();
        assert_eq!(ids.len(), 10_000);
    }

    #[test]
    fn model_checking_accepts_absent_and_exact_but_not_another_model() {
        assert!(check_model(None, "crucible-120m").is_ok());
        assert!(check_model(Some("crucible-120m"), "crucible-120m").is_ok());
        let e = check_model(Some("gpt-4o"), "crucible-120m").unwrap_err();
        assert_eq!(e.status, StatusCode::NOT_FOUND);
        assert_eq!(e.body.code.as_deref(), Some("model_not_found"));
        // The message must name what is actually served, or a client that
        // guessed wrong has nothing to go on.
        assert!(e.body.message.contains("crucible-120m"), "{}", e.body.message);
    }

    #[test]
    fn model_ids_may_not_be_paths() {
        assert!(validate_model_id("crucible-120m").is_ok());
        assert!(validate_model_id("").is_err());
        assert!(validate_model_id("/home/user/export/120m").is_err());
        assert!(validate_model_id("models\\120m").is_err());
    }

    #[test]
    fn no_op_parameter_values_are_accepted_and_others_are_not() {
        assert!(reject_if_set(Some(1.0f64), |v| *v == 1.0, "top_p", "").is_ok());
        assert!(reject_if_set(Some(0.2f64), |v| *v == 1.0, "top_p", "").is_err());
        assert!(reject_if_set(None::<f64>, |v| *v == 1.0, "top_p", "").is_ok());
    }

    #[test]
    fn error_bodies_carry_every_field_the_schema_requires() {
        let e = ApiError::unsupported("top_p", "Crucible samples with top-k.");
        let json = serde_json::to_value(ApiErrorEnvelope { error: e.body }).unwrap();
        let err = &json["error"];
        for field in ["message", "type", "param", "code"] {
            assert!(err.get(field).is_some(), "missing {field} in {json}");
        }
        assert_eq!(err["type"], "invalid_request_error");
        assert_eq!(err["param"], "top_p");
    }
}
