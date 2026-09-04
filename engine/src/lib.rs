//! Inference engine for models trained by the Python half of this repository.
//!
//! Built in stages, each validated before the next begins:
//!   1. checkpoint loading                (this stage)
//!   2. CPU forward pass, numerics pinned against PyTorch
//!   3. KV cache
//!   4. GPU kernels, then paged attention and continuous batching
//!
//! The CPU path is not a throwaway. It stays as the reference implementation
//! that GPU kernels are validated against, which is the only practical way to
//! tell a wrong kernel from a merely slow one.

pub mod cache;
// Conversation -> prompt text. Protocol-neutral, and shared by every
// compatibility adapter so that equivalent conversations cannot diverge.
pub mod chat_template;
pub mod config;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_model;
pub mod model;
pub mod paged;
// Wire types, shared by the service and its clients. No feature gate: the point
// is that a client can depend on the protocol without depending on the engine.
pub mod protocol;
// Token selection, shared by the CLI and the batched runtime so the two cannot
// drift apart.
pub mod sampling;
pub mod ops;
pub mod quant;
// Compatibility adapters. Gated with the service, since each is an adapter over
// it and has no meaning without one. Siblings, not layers: neither is built on
// the other, and what they share lives in `chat_template`.
#[cfg(feature = "cuda")]
pub mod anthropic;
#[cfg(feature = "cuda")]
pub mod openai;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
pub mod server;
#[cfg(feature = "tui")]
pub mod tui;
pub mod tokenizer;
pub mod weights;

pub use cache::KvCache;
pub use config::Config;
pub use model::Model;
pub use paged::{PagePool, SequencePages, PAGE_TOKENS};
pub use quant::QuantTensor;
pub use tokenizer::Tokenizer;
pub use weights::{Tensor, Weights};
