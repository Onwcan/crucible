//! Wire types for the OpenAI-compatible endpoints.
//!
//! Deliberately separate from `protocol.rs`, which is Crucible's own wire
//! format. The two have different owners: the native protocol can change when
//! the engine changes, while these types are pinned to somebody else's
//! published schema and may only change when that schema does. Mixing them
//! would mean every OpenAI field addition litters the native protocol, and
//! every native change risks breaking third-party clients.
//!
//! Schemas here follow the official OpenAI OpenAPI specification, version
//! 2.3.0. Where a field is deprecated upstream but still sent by real clients
//! (`max_tokens`, `seed`) it is accepted; where a field would change generation
//! behaviour that Crucible does not implement, it is captured so the handler
//! can reject it rather than ignore it.

use serde::{Deserialize, Serialize};

// --- shared -----------------------------------------------------------------

/// Token accounting, from the tokenizer rather than from string lengths.
///
/// The `*_tokens_details` sub-objects the upstream schema allows are omitted
/// rather than zero-filled: Crucible has no cached prompts, no reasoning tokens
/// and no audio, and reporting zeros would assert those features exist and
/// happened not to be used.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// `stream_options`, of which only `include_usage` is meaningful here.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

// --- models -----------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

impl ModelObject {
    pub fn new(id: &str, created: i64) -> Self {
        Self {
            id: id.to_string(),
            object: "model",
            created,
            // Not "openai". This is a locally trained model served by a local
            // engine, and saying otherwise in a field named `owned_by` would be
            // the one lie the whole endpoint exists to avoid.
            owned_by: "crucible",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// --- chat completions -------------------------------------------------------

/// One message's content: a bare string, or the structured content-part array.
///
/// Untagged, because the upstream schema is a `oneOf` on shape rather than on a
/// discriminator field.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    /// `null` is legal upstream for assistant messages with tool calls.
    Null,
}

/// A structured content part. Only `text` carries anything Crucible can use;
/// every other type is captured so it can be rejected by name rather than
/// silently flattened into the prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub name: Option<String>,
    /// Present only on assistant messages that called tools. Captured to
    /// reject, never to interpret.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,

    // --- supported ---
    /// Deprecated upstream in favour of `max_completion_tokens`, still sent by
    /// most clients and by older SDK versions, so both are accepted.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Crucible extension, not an OpenAI field. Documented as such; clients
    /// that never send it are unaffected.
    #[serde(default)]
    pub top_k: Option<usize>,

    // --- captured in order to be refused ---
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub functions: Option<serde_json::Value>,
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub modalities: Option<serde_json::Value>,
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_effort: Option<serde_json::Value>,
    // Unlisted fields (user, store, metadata, service_tier, ...) are accepted
    // and ignored: they are bookkeeping upstream and change no output here.
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
    /// Required by the schema, and genuinely null: Crucible has no refusal
    /// classifier, so there is never a refusal string to report.
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    /// Present only when the client asked for usage. Upstream sends `null` on
    /// every chunk but the last in that case, and omits the field entirely
    /// otherwise, which is what the two levels of Option encode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}

// --- legacy completions -----------------------------------------------------

/// The `prompt` field, which upstream allows as a string, an array of strings,
/// or token id arrays.
///
/// Only the single-string form maps onto one Crucible request. The array forms
/// are a batch of independent completions in one call, which is a different
/// response shape, so they are parsed and then refused rather than silently
/// serving only the first element.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    Text(String),
    Many(Vec<serde_json::Value>),
    Other(serde_json::Value),
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<PromptField>,

    // --- supported ---
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Crucible extension, as on the chat endpoint.
    #[serde(default)]
    pub top_k: Option<usize>,

    // --- captured in order to be refused ---
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub best_of: Option<u32>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<&'static str>,
}

/// Used for streamed and non-streamed responses alike: unlike chat, the legacy
/// completions schema gives both the same shape and the same `object` value.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}
