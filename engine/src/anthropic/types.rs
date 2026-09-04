//! Wire types for the Anthropic-compatible endpoints.
//!
//! Verified against the official `anthropic` Python SDK, version 1.3.0, whose
//! types are generated from Anthropic's own OpenAPI spec. Where this file and
//! that SDK disagree, this file is wrong.
//!
//! Separate from `openai/types.rs` on purpose, and not built on top of it. The
//! two protocols shape the same operation differently -- Anthropic puts the
//! system prompt at the top level, requires `max_tokens`, returns content as a
//! block list, and streams typed events rather than chunks -- and expressing
//! one in terms of the other would make every future divergence a refactor.
//! What they genuinely share, conversation-to-prompt serialization, lives in
//! `chat_template` and is shared there instead.
//!
//! # The sampling parameters are gone
//!
//! `temperature`, `top_p` and `top_k` appear nowhere in the current SDK's
//! Messages types -- not in `message_create_params.py`, not anywhere under
//! `types/`. They are captured below only so they can be *refused* with an
//! explanation, because silently ignoring a field a caller set is how a server
//! ends up lying about its own output.

use serde::{Deserialize, Serialize};

// --- request ----------------------------------------------------------------

/// `system`: a bare string, or a list of text blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemField {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

/// One inbound content block. Only `text` carries anything usable; every other
/// type is captured so it can be refused by name rather than flattened.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockIn {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// A message's content: a bare string or a block list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlockIn>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct InMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Required by the schema, unlike every other surface here.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub messages: Option<Vec<InMessage>>,
    #[serde(default)]
    pub system: Option<SystemField>,
    #[serde(default)]
    pub stream: Option<bool>,

    // --- accepted and inert ---
    /// `{user_id}`. An opaque identifier with no effect on generation upstream
    /// either, so it is accepted and ignored rather than refused.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// `auto` | `standard_only`. This server has one tier and honours both
    /// trivially.
    #[serde(default)]
    pub service_tier: Option<String>,

    // --- captured in order to be refused ---
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    #[serde(default)]
    pub output_config: Option<serde_json::Value>,
    #[serde(default)]
    pub container: Option<serde_json::Value>,
    #[serde(default)]
    pub cache_control: Option<serde_json::Value>,
    /// Not in the current SDK's Messages types at all. Refused with a pointer
    /// at the namespaced extensions below, so a caller who wants sampling is
    /// told how to ask for it rather than quietly given greedy output.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,

    // --- Crucible extensions, unmistakably not Anthropic fields ---
    /// Positive value enables sampling; absent or zero is greedy.
    #[serde(default)]
    pub crucible_temperature: Option<f32>,
    #[serde(default)]
    pub crucible_top_k: Option<usize>,
    #[serde(default)]
    pub crucible_seed: Option<u64>,
}

/// `POST /v1/messages/count_tokens`. The same conversation fields, no budget
/// and no stream.
#[derive(Debug, Clone, Deserialize)]
pub struct CountTokensRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<InMessage>>,
    #[serde(default)]
    pub system: Option<SystemField>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
}

// --- response ---------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            kind: "text",
            text: text.into(),
        }
    }
}

/// `input_tokens` and `output_tokens` are the only required members.
///
/// The cache and server-tool counters the schema also allows are omitted rather
/// than zero-filled: this server has no prompt cache and no server tools, and
/// `cache_read_input_tokens: 0` would assert that it has both and neither
/// happened to be used.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<TextBlock>,
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: usize,
}

// --- streaming --------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: MessageResponse,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub index: u32,
    pub content_block: TextBlock,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
}

/// The `message_delta` usage object carries cumulative output tokens; only
/// `output_tokens` is required, and `input_tokens` is echoed because the SDK
/// folds this event into its final accumulated message.
#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub delta: MessageDeltaBody,
    pub usage: MessageDeltaUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

// --- models -----------------------------------------------------------------

/// Anthropic's model object, which is *not* OpenAI's: `type` rather than
/// `object`, an RFC 3339 `created_at` rather than a unix `created`, a
/// `display_name`, and no `owned_by`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub display_name: String,
    pub created_at: String,
    pub max_tokens: usize,
    pub max_input_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelPage {
    pub data: Vec<ModelInfo>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}
