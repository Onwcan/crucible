//! Transformer forward pass on the GPU.
//!
//! Weights and the KV cache live on the device for the model's whole lifetime.
//! Only two transfers happen per token: the token id in, and the logits out.
//! Anything else would put a PCIe round-trip inside the decode loop, and at
//! ~1 ms per token that would dominate everything the kernels do.
//!
//! Structurally identical to the CPU path in `model.rs` -- token-major, same
//! ordering, same GQA head mapping -- so the two can be compared directly.
//! `crate::gpu::validate` checks the kernels; `gpu-logits` checks the whole
//! model against PyTorch.

use anyhow::{bail, Result};
use cudarc::driver::CudaSlice;

use crate::config::Config;
use crate::gpu::Gpu;
use crate::ops::RopeTable;
use crate::weights::Weights;

const NORM_EPS: f32 = 1e-6;

struct GpuLayer {
    attn_norm: CudaSlice<f32>,
    q_proj: CudaSlice<f32>,
    k_proj: CudaSlice<f32>,
    v_proj: CudaSlice<f32>,
    o_proj: CudaSlice<f32>,
    mlp_norm: CudaSlice<f32>,
    gate_proj: Option<CudaSlice<f32>>,
    up_proj: CudaSlice<f32>,
    down_proj: CudaSlice<f32>,
}

/// Device-side scratch, allocated once. Decoding one token launches ~170
/// kernels; allocating inside that loop would be pure overhead.
struct Scratch {
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    q: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    mlp_out: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

pub struct GpuModel {
    pub gpu: Gpu,
    pub cfg: Config,
    tok_emb: CudaSlice<f32>,
    layers: Vec<GpuLayer>,
    final_norm: CudaSlice<f32>,
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
    scratch: Scratch,

    /// `[layer][position][n_kv_head * head_dim]`, matching the CPU cache.
    k_cache: CudaSlice<f32>,
    v_cache: CudaSlice<f32>,
    capacity: usize,
    cache_len: usize,
    hidden: usize,
}

impl GpuModel {
    pub fn load(cfg: Config, w: &Weights, capacity: usize) -> Result<Self> {
        if cfg.pos_encoding != "rope" {
            bail!("the GPU path currently implements rope only, not {}", cfg.pos_encoding);
        }
        if cfg.norm != "rmsnorm" {
            bail!("the GPU path currently implements rmsnorm only, not {}", cfg.norm);
        }
        if cfg.norm_placement != "pre" {
            bail!("the GPU path currently implements pre-norm only");
        }

        let gpu = Gpu::new(0)?;
        let capacity = capacity.min(cfg.block_size);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blocks.{i}");
            layers.push(GpuLayer {
                attn_norm: gpu.to_device(&w.get(&format!("{p}.attn_norm.weight"))?.data)?,
                q_proj: gpu.to_device(&w.get(&format!("{p}.attn.q_proj.weight"))?.data)?,
                k_proj: gpu.to_device(&w.get(&format!("{p}.attn.k_proj.weight"))?.data)?,
                v_proj: gpu.to_device(&w.get(&format!("{p}.attn.v_proj.weight"))?.data)?,
                o_proj: gpu.to_device(&w.get(&format!("{p}.attn.o_proj.weight"))?.data)?,
                mlp_norm: gpu.to_device(&w.get(&format!("{p}.mlp_norm.weight"))?.data)?,
                gate_proj: match cfg.activation.as_str() {
                    "swiglu" => Some(gpu.to_device(&w.get(&format!("{p}.mlp.gate_proj.weight"))?.data)?),
                    _ => None,
                },
                up_proj: gpu.to_device(&w.get(&format!("{p}.mlp.up_proj.weight"))?.data)?,
                down_proj: gpu.to_device(&w.get(&format!("{p}.mlp.down_proj.weight"))?.data)?,
            });
        }

        let hidden = w.get("blocks.0.mlp.up_proj.weight")?.shape[0];
        let table = RopeTable::new(cfg.head_dim(), cfg.block_size, cfg.rope_theta);
        let d = cfg.n_embd;
        let kv_dim = cfg.n_kv_head * cfg.head_dim();

        Ok(Self {
            tok_emb: gpu.to_device(&w.get("tok_emb.weight")?.data)?,
            final_norm: gpu.to_device(&w.get("final_norm.weight")?.data)?,
            rope_cos: gpu.to_device(&table.cos)?,
            rope_sin: gpu.to_device(&table.sin)?,
            scratch: Scratch {
                x: gpu.alloc(d)?,
                normed: gpu.alloc(d)?,
                q: gpu.alloc(d)?,
                attn: gpu.alloc(d)?,
                proj: gpu.alloc(d)?,
                gate: gpu.alloc(hidden)?,
                up: gpu.alloc(hidden)?,
                mlp_out: gpu.alloc(d)?,
                logits: gpu.alloc(cfg.vocab_size)?,
            },
            k_cache: gpu.alloc(cfg.n_layer * capacity * kv_dim)?,
            v_cache: gpu.alloc(cfg.n_layer * capacity * kv_dim)?,
            layers,
            capacity,
            cache_len: 0,
            hidden,
            cfg,
            gpu,
        })
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn reset(&mut self) {
        self.cache_len = 0;
    }

    /// Bytes of device memory held by weights and cache.
    pub fn device_bytes(&self) -> usize {
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        let cache = 2 * self.cfg.n_layer * self.capacity * kv_dim * 4;
        let weights = self.cfg.vocab_size * self.cfg.n_embd * 4
            + self.cfg.n_layer
                * (3 * self.cfg.n_embd * self.cfg.n_embd + 3 * self.hidden * self.cfg.n_embd)
                * 4;
        weights + cache
    }

    /// Append tokens to the cache and return logits for the final position.
    ///
    /// Kernels are queued without synchronising between them: the stream
    /// executes in order, so correctness holds, and the host does not stall on
    /// each of the ~170 launches a token requires. The single sync happens when
    /// logits are copied back.
    pub fn forward(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;

        if tokens.is_empty() {
            bail!("no tokens to process");
        }
        if self.cache_len + tokens.len() > self.capacity {
            bail!(
                "sequence of {} exceeds cache capacity {}",
                self.cache_len + tokens.len(),
                self.capacity
            );
        }

        for &token in tokens {
            if token >= cfg.vocab_size {
                bail!("token id {token} outside vocabulary {}", cfg.vocab_size);
            }
            let pos = self.cache_len;
            let s = &mut self.scratch;

            self.gpu.embed(&self.tok_emb, &mut s.x, token, d)?;

            for (l, layer) in self.layers.iter().enumerate() {
                // Offsets of this layer's cache region and this position's slot.
                let layer_base = l * self.capacity * kv_dim;
                let slot = layer_base + pos * kv_dim;

                self.gpu.rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;

                // K and V are written straight into the cache: passing an
                // output offset avoids a separate copy kernel per layer.
                self.gpu.gemv_at(&layer.k_proj, &s.normed, &mut self.k_cache, kv_dim, d, slot)?;
                self.gpu.gemv_at(&layer.v_proj, &s.normed, &mut self.v_cache, kv_dim, d, slot)?;
                self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin, n_kv, hd, pos, slot)?;

                self.gpu.gemv_at(&layer.q_proj, &s.normed, &mut s.q, d, d, 0)?;
                self.gpu.rope_at(&mut s.q, &self.rope_cos, &self.rope_sin, n_head, hd, pos, 0)?;

                self.gpu.attention_decode(
                    &s.q,
                    &self.k_cache,
                    &self.v_cache,
                    &mut s.attn,
                    n_head,
                    n_kv,
                    hd,
                    pos + 1,
                    kv_dim,
                    layer_base,
                )?;

                self.gpu.gemv_at(&layer.o_proj, &s.attn, &mut s.proj, d, d, 0)?;
                self.gpu.add_inplace(&mut s.x, &s.proj, d)?;

                self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
                match &layer.gate_proj {
                    Some(gate) => {
                        self.gpu.gemv_at(gate, &s.normed, &mut s.gate, self.hidden, d, 0)?;
                        self.gpu.gemv_at(&layer.up_proj, &s.normed, &mut s.up, self.hidden, d, 0)?;
                        self.gpu.silu_mul(&mut s.gate, &s.up, self.hidden)?;
                    }
                    None => bail!("the GPU path currently implements swiglu only"),
                }
                self.gpu.gemv_at(&layer.down_proj, &s.gate, &mut s.mlp_out, d, self.hidden, 0)?;
                self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
            }

            self.cache_len += 1;
        }

        let s = &mut self.scratch;
        self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
        // lm_head is tied to tok_emb, so the embedding table is the projection.
        self.gpu.gemv_at(&self.tok_emb, &s.normed, &mut s.logits, cfg.vocab_size, d, 0)?;

        self.gpu.to_host(&self.scratch.logits)
    }
}
