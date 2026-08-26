//! Model architecture, parsed from the `config.json` written by `export.py`.
//!
//! The engine refuses to guess. Head counts, norm type and positional encoding
//! all change the compute graph, and inferring them from tensor shapes alone
//! would silently produce a model that runs and emits plausible-looking
//! nonsense. Anything unrecognised is an error, not a default.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub block_size: usize,

    pub attention: String,
    pub pos_encoding: String,
    pub activation: String,
    pub norm: String,
    pub norm_placement: String,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub bias: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub val_loss: Option<f32>,
    #[serde(default)]
    pub trained_steps: Option<usize>,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configurations this engine cannot execute correctly, loudly and
    /// at load time rather than as degraded output later.
    fn validate(&self) -> Result<()> {
        if self.n_embd % self.n_head != 0 {
            bail!(
                "n_embd ({}) is not divisible by n_head ({})",
                self.n_embd,
                self.n_head
            );
        }
        if self.n_head % self.n_kv_head != 0 {
            bail!(
                "n_head ({}) is not divisible by n_kv_head ({})",
                self.n_head,
                self.n_kv_head
            );
        }
        for (field, value, supported) in [
            ("norm", &self.norm, &["rmsnorm", "layernorm"][..]),
            ("activation", &self.activation, &["swiglu", "gelu"][..]),
            ("norm_placement", &self.norm_placement, &["pre", "post"][..]),
            (
                "pos_encoding",
                &self.pos_encoding,
                &["rope", "alibi", "learned", "none"][..],
            ),
        ] {
            if !supported.contains(&value.as_str()) {
                bail!("unsupported {field}: {value:?} (expected one of {supported:?})");
            }
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Query heads per KV head. 1 means MHA, n_head means MQA.
    pub fn n_rep(&self) -> usize {
        self.n_head / self.n_kv_head
    }

    /// Bytes of KV cache for one sequence at full context, in f32.
    pub fn kv_cache_bytes(&self, seq_len: usize) -> usize {
        2 * self.n_layer * self.n_kv_head * self.head_dim() * seq_len * 4
    }
}
