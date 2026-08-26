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
use crate::quant::QuantTensor;
use crate::weights::Weights;

const NORM_EPS: f32 = 1e-6;

/// Weight precision for the large projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    /// int8 weights with per-row scales; activations stay f32.
    Int8,
}

impl Precision {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" | "fp32" => Some(Self::F32),
            "int8" | "i8" => Some(Self::Int8),
            _ => None,
        }
    }
}

/// A projection matrix, in whichever precision the model was loaded at.
///
/// Norms and other small tensors stay f32 unconditionally: they are a rounding
/// error in the byte budget, and quantising them would cost accuracy for
/// nothing measurable.
enum Proj {
    F32(CudaSlice<f32>),
    Int8 { data: CudaSlice<i8>, scales: CudaSlice<f32> },
}

impl Proj {
    fn bytes(&self, rows: usize, cols: usize) -> usize {
        match self {
            Proj::F32(_) => rows * cols * 4,
            Proj::Int8 { .. } => rows * cols + rows * 4,
        }
    }
}

struct GpuLayer {
    attn_norm: CudaSlice<f32>,
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    mlp_norm: CudaSlice<f32>,
    gate_proj: Option<Proj>,
    up_proj: Proj,
    down_proj: Proj,
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
    pub precision: Precision,
    tok_emb: Proj,
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
        Self::load_with(cfg, w, capacity, Precision::F32)
    }

    pub fn load_with(
        cfg: Config,
        w: &Weights,
        capacity: usize,
        precision: Precision,
    ) -> Result<Self> {
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

        // Upload one projection at the requested precision.
        let upload = |name: &str| -> Result<Proj> {
            let t = w.get(name)?;
            Ok(match precision {
                Precision::F32 => Proj::F32(gpu.to_device(&t.data)?),
                Precision::Int8 => {
                    let q = QuantTensor::from_tensor(&t);
                    Proj::Int8 {
                        data: gpu.to_device_i8(&q.data)?,
                        scales: gpu.to_device(&q.scales)?,
                    }
                }
            })
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blocks.{i}");
            layers.push(GpuLayer {
                attn_norm: gpu.to_device(&w.get(&format!("{p}.attn_norm.weight"))?.data)?,
                q_proj: upload(&format!("{p}.attn.q_proj.weight"))?,
                k_proj: upload(&format!("{p}.attn.k_proj.weight"))?,
                v_proj: upload(&format!("{p}.attn.v_proj.weight"))?,
                o_proj: upload(&format!("{p}.attn.o_proj.weight"))?,
                mlp_norm: gpu.to_device(&w.get(&format!("{p}.mlp_norm.weight"))?.data)?,
                gate_proj: match cfg.activation.as_str() {
                    "swiglu" => Some(upload(&format!("{p}.mlp.gate_proj.weight"))?),
                    _ => None,
                },
                up_proj: upload(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: upload(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }

        let hidden = w.get("blocks.0.mlp.up_proj.weight")?.shape[0];
        let table = RopeTable::new(cfg.head_dim(), cfg.block_size, cfg.rope_theta);
        let d = cfg.n_embd;
        let kv_dim = cfg.n_kv_head * cfg.head_dim();

        Ok(Self {
            tok_emb: upload("tok_emb.weight")?,
            precision,
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

    /// y[offset..] = W · x, dispatching on how the weights were stored.
    fn project(
        gpu: &Gpu,
        w: &Proj,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        y_offset: usize,
    ) -> Result<()> {
        match w {
            Proj::F32(data) => gpu.gemv_at(data, x, y, rows, cols, y_offset),
            Proj::Int8 { data, scales } => {
                gpu.gemv_i8_at(data, scales, x, y, rows, cols, y_offset)
            }
        }
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn reset(&mut self) {
        self.cache_len = 0;
    }

    /// Bytes of device memory held by weights and cache.
    pub fn device_bytes(&self) -> usize {
        self.weight_bytes() + self.cache_bytes()
    }

    pub fn cache_bytes(&self) -> usize {
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        2 * self.cfg.n_layer * self.capacity * kv_dim * 4
    }

    /// Weight bytes actually resident, counted from how each tensor is stored
    /// rather than assumed from the parameter count -- which is the whole point
    /// of quantising, and would be invisible if this were hardcoded to f32.
    pub fn weight_bytes(&self) -> usize {
        let (d, hidden) = (self.cfg.n_embd, self.hidden);
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        let mut total = self.tok_emb.bytes(self.cfg.vocab_size, d);
        for l in &self.layers {
            total += l.q_proj.bytes(d, d)
                + l.k_proj.bytes(kv_dim, d)
                + l.v_proj.bytes(kv_dim, d)
                + l.o_proj.bytes(d, d)
                + l.up_proj.bytes(hidden, d)
                + l.down_proj.bytes(d, hidden)
                + l.gate_proj.as_ref().map_or(0, |g| g.bytes(hidden, d))
                + 2 * d * 4; // the two norms stay f32
        }
        total
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

            match &self.tok_emb {
                Proj::F32(t) => self.gpu.embed(t, &mut s.x, token, d)?,
                Proj::Int8 { data, scales } => {
                    self.gpu.embed_i8(data, scales, &mut s.x, token, d)?
                }
            }

            for (l, layer) in self.layers.iter().enumerate() {
                // Offsets of this layer's cache region and this position's slot.
                let layer_base = l * self.capacity * kv_dim;
                let slot = layer_base + pos * kv_dim;

                self.gpu.rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;

                // K and V are written straight into the cache: passing an
                // output offset avoids a separate copy kernel per layer.
                Self::project(&self.gpu, &layer.k_proj, &s.normed, &mut self.k_cache, kv_dim, d, slot)?;
                Self::project(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_cache, kv_dim, d, slot)?;
                self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin, n_kv, hd, pos, slot)?;

                Self::project(&self.gpu, &layer.q_proj, &s.normed, &mut s.q, d, d, 0)?;
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

                Self::project(&self.gpu, &layer.o_proj, &s.attn, &mut s.proj, d, d, 0)?;
                self.gpu.add_inplace(&mut s.x, &s.proj, d)?;

                self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
                match &layer.gate_proj {
                    Some(gate) => {
                        Self::project(&self.gpu, gate, &s.normed, &mut s.gate, self.hidden, d, 0)?;
                        Self::project(&self.gpu, &layer.up_proj, &s.normed, &mut s.up, self.hidden, d, 0)?;
                        self.gpu.silu_mul(&mut s.gate, &s.up, self.hidden)?;
                    }
                    None => bail!("the GPU path currently implements swiglu only"),
                }
                Self::project(&self.gpu, &layer.down_proj, &s.gate, &mut s.mlp_out, d, self.hidden, 0)?;
                self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
            }

            self.cache_len += 1;
        }

        let s = &mut self.scratch;
        self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
        // lm_head is tied to tok_emb, so the embedding table is the projection.
        Self::project(&self.gpu, &self.tok_emb, &s.normed, &mut s.logits, cfg.vocab_size, d, 0)?;

        self.gpu.to_host(&self.scratch.logits)
    }
}
