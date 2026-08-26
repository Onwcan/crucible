//! Transformer forward pass on CPU.
//!
//! Token-major: each token flows through every layer before the next token
//! starts. That ordering is what makes incremental decoding possible -- by the
//! time position p reaches layer L, every earlier position has already written
//! its layer-L keys and values into the cache, so attention can read them
//! instead of recomputing them.
//!
//! Layout note: PyTorch stores `nn.Linear` weights as `[out_features,
//! in_features]`, so every projection here is a plain row-major mat-vec with no
//! transpose.

use anyhow::{bail, Result};

use crate::cache::KvCache;
use crate::config::Config;
use crate::ops::{self, RopeTable};
use crate::weights::{Tensor, Weights};

const NORM_EPS: f32 = 1e-6;

struct Layer {
    attn_norm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    mlp_norm: Tensor,
    gate_proj: Option<Tensor>, // SwiGLU only
    up_proj: Tensor,
    down_proj: Tensor,
}

/// Scratch buffers, allocated once and reused across every token and layer.
///
/// Decoding one token touches these dozens of times; allocating per call would
/// dominate the profile of an otherwise memory-bound workload.
struct Scratch {
    normed: Vec<f32>,
    q: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mlp_out: Vec<f32>,
    scores: Vec<f32>,
}

pub struct Model {
    pub cfg: Config,
    tok_emb: Tensor,
    pos_emb: Option<Tensor>,
    layers: Vec<Layer>,
    final_norm: Tensor,
    rope: Option<RopeTable>,
}

impl Model {
    pub fn load(cfg: Config, w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blocks.{i}");
            layers.push(Layer {
                attn_norm: w.get(&format!("{p}.attn_norm.weight"))?,
                q_proj: w.get(&format!("{p}.attn.q_proj.weight"))?,
                k_proj: w.get(&format!("{p}.attn.k_proj.weight"))?,
                v_proj: w.get(&format!("{p}.attn.v_proj.weight"))?,
                o_proj: w.get(&format!("{p}.attn.o_proj.weight"))?,
                mlp_norm: w.get(&format!("{p}.mlp_norm.weight"))?,
                gate_proj: match cfg.activation.as_str() {
                    "swiglu" => Some(w.get(&format!("{p}.mlp.gate_proj.weight"))?),
                    _ => None,
                },
                up_proj: w.get(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: w.get(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }

        let rope = match cfg.pos_encoding.as_str() {
            "rope" => Some(RopeTable::new(cfg.head_dim(), cfg.block_size, cfg.rope_theta)),
            _ => None,
        };
        let pos_emb = match cfg.pos_encoding.as_str() {
            "learned" => Some(w.get("pos_emb.weight")?),
            _ => None,
        };
        if cfg.pos_encoding == "alibi" {
            bail!("alibi is not implemented in the CPU path yet");
        }

        Ok(Self {
            tok_emb: w.get("tok_emb.weight")?,
            pos_emb,
            layers,
            final_norm: w.get("final_norm.weight")?,
            rope,
            cfg,
        })
    }

    pub fn new_cache(&self, capacity: usize) -> KvCache {
        KvCache::new(&self.cfg, capacity.min(self.cfg.block_size))
    }

    fn scratch(&self) -> Scratch {
        let d = self.cfg.n_embd;
        let hidden = self.layers[0].up_proj.rows();
        Scratch {
            normed: vec![0.0; d],
            q: vec![0.0; d],
            attn: vec![0.0; d],
            proj: vec![0.0; d],
            gate: vec![0.0; hidden],
            up: vec![0.0; hidden],
            mlp_out: vec![0.0; d],
            scores: vec![0.0; self.cfg.block_size],
        }
    }

    fn norm(&self, x: &[f32], weight: &Tensor, out: &mut [f32]) {
        match self.cfg.norm.as_str() {
            "layernorm" => ops::layernorm(x, &weight.data, None, NORM_EPS, out),
            _ => ops::rmsnorm(x, &weight.data, NORM_EPS, out),
        }
    }

    /// Full forward over a sequence, with no cache retained.
    ///
    /// Kept because it is the simplest thing to compare against PyTorch; the
    /// cached path must agree with it exactly.
    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<f32>> {
        let mut cache = self.new_cache(tokens.len().max(1));
        self.forward_cached(tokens, &mut cache)
    }

    /// Append `tokens` to the cache and return logits for the final position.
    ///
    /// Serves both roles: prefill passes the whole prompt, decode passes a
    /// single token. Positions are numbered from the cache's current length, so
    /// rotary embeddings and causal masking stay correct across calls.
    pub fn forward_cached(&self, tokens: &[usize], cache: &mut KvCache) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let n_rep = cfg.n_rep();
        let scale = 1.0 / (hd as f32).sqrt();

        if tokens.is_empty() {
            bail!("no tokens to process");
        }
        if cache.n_layer() != cfg.n_layer {
            bail!("cache was built for a different model");
        }

        let start = cache.len();
        if start + tokens.len() > cache.capacity() {
            bail!(
                "sequence of {} exceeds cache capacity {}",
                start + tokens.len(),
                cache.capacity()
            );
        }

        let mut s = self.scratch();
        let mut x = vec![0.0f32; d];

        for (i, &token) in tokens.iter().enumerate() {
            let pos = start + i;
            if token >= cfg.vocab_size {
                bail!("token id {token} outside vocabulary {}", cfg.vocab_size);
            }

            x.copy_from_slice(self.tok_emb.row(token));
            if let Some(pe) = &self.pos_emb {
                for (xi, pi) in x.iter_mut().zip(pe.row(pos)) {
                    *xi += pi;
                }
            }

            for (l, layer) in self.layers.iter().enumerate() {
                // ---- attention ------------------------------------------
                let src: &[f32] = if cfg.norm_placement == "pre" {
                    self.norm(&x, &layer.attn_norm, &mut s.normed);
                    &s.normed
                } else {
                    &x
                };

                {
                    let (k_slot, v_slot) = cache.slot_mut(l, pos);
                    ops::matvec(&layer.k_proj.data, n_kv * hd, d, src, k_slot);
                    ops::matvec(&layer.v_proj.data, n_kv * hd, d, src, v_slot);
                    if let Some(rope) = &self.rope {
                        for h in 0..n_kv {
                            rope.apply(&mut k_slot[h * hd..(h + 1) * hd], pos);
                        }
                    }
                }

                ops::matvec(&layer.q_proj.data, d, d, src, &mut s.q);
                if let Some(rope) = &self.rope {
                    for h in 0..n_head {
                        rope.apply(&mut s.q[h * hd..(h + 1) * hd], pos);
                    }
                }

                for h in 0..n_head {
                    // Query head h reads KV head h / n_rep, mirroring
                    // repeat_interleave(n_rep, dim=1), which maps KV head j to
                    // query heads [j*n_rep, (j+1)*n_rep). Reversing this yields
                    // a model that runs and produces plausible nonsense.
                    let kv_h = h / n_rep;
                    let qh = &s.q[h * hd..(h + 1) * hd];

                    // Causal: positions 0..=pos, which is exactly what the
                    // cache holds now that this position has been written.
                    for (j, score) in s.scores[..=pos].iter_mut().enumerate() {
                        let kh = cache.key(l, j, kv_h, hd);
                        let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                        *score = dot * scale;
                    }
                    ops::softmax(&mut s.scores[..=pos]);

                    let dst = &mut s.attn[h * hd..(h + 1) * hd];
                    dst.fill(0.0);
                    for (j, &weight) in s.scores[..=pos].iter().enumerate() {
                        let vh = cache.value(l, j, kv_h, hd);
                        for (o, vi) in dst.iter_mut().zip(vh) {
                            *o += weight * vi;
                        }
                    }
                }

                ops::matvec(&layer.o_proj.data, d, d, &s.attn, &mut s.proj);
                for (xi, pi) in x.iter_mut().zip(s.proj.iter()) {
                    *xi += pi;
                }
                if cfg.norm_placement == "post" {
                    let snapshot = x.clone();
                    self.norm(&snapshot, &layer.attn_norm, &mut x);
                }

                // ---- feed-forward ----------------------------------------
                let src: &[f32] = if cfg.norm_placement == "pre" {
                    self.norm(&x, &layer.mlp_norm, &mut s.normed);
                    &s.normed
                } else {
                    &x
                };

                let hidden = layer.up_proj.rows();
                match &layer.gate_proj {
                    Some(gate) => {
                        ops::matvec(&gate.data, hidden, d, src, &mut s.gate);
                        ops::matvec(&layer.up_proj.data, hidden, d, src, &mut s.up);
                        for (g, u) in s.gate.iter_mut().zip(s.up.iter()) {
                            *g = ops::silu(*g) * u;
                        }
                    }
                    None => {
                        ops::matvec(&layer.up_proj.data, hidden, d, src, &mut s.gate);
                        for g in s.gate.iter_mut() {
                            *g = ops::gelu(*g);
                        }
                    }
                }
                ops::matvec(&layer.down_proj.data, d, hidden, &s.gate, &mut s.mlp_out);

                for (xi, mi) in x.iter_mut().zip(s.mlp_out.iter()) {
                    *xi += mi;
                }
                if cfg.norm_placement == "post" {
                    let snapshot = x.clone();
                    self.norm(&snapshot, &layer.mlp_norm, &mut x);
                }
            }

            // Every layer has written this position, so it is safe to expose.
            cache.advance(1)?;
        }

        // ---- final norm and vocabulary projection --------------------------
        let mut final_out = vec![0.0f32; d];
        self.norm(&x, &self.final_norm, &mut final_out);

        // lm_head is tied to tok_emb, so the embedding table is the projection.
        let mut logits = vec![0.0f32; cfg.vocab_size];
        ops::matvec(&self.tok_emb.data, cfg.vocab_size, d, &final_out, &mut logits);
        Ok(logits)
    }
}
