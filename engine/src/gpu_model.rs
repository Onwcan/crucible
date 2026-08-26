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
use cudarc::driver::{CudaGraph, CudaSlice};

use crate::config::Config;
use crate::gpu::{Gpu, PARAM_COUNT, PARAM_POS, PARAM_SEQ, PARAM_SLOT, PARAM_TOKEN, PARAM_ZERO};
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

/// One stage of a profiled decode step.
pub struct Stage {
    pub name: String,
    /// Timed blocks per token, which is how much sync overhead it absorbs.
    pub calls: usize,
    /// Measured time including the sync at the end of each block.
    pub raw: f64,
    /// With sync overhead removed. This is the number worth acting on.
    pub adjusted: f64,
}

pub struct ProfileReport {
    pub stages: Vec<Stage>,
    /// Estimated per-block launch + sync overhead, taken from the cheapest
    /// kernel-launching stage.
    pub sync_cost: f64,
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

    /// Per-step scalars the kernels read from device memory. A captured graph
    /// freezes kernel arguments, so these cannot be passed by value.
    params: CudaSlice<i32>,
    host_params: Vec<i32>,

    /// Captured decode graph, if graph mode is enabled and warm.
    graph: Option<CudaGraph>,
    use_graph: bool,
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
            params: gpu.to_device_i32(&vec![0i32; PARAM_COUNT])?,
            host_params: vec![0i32; PARAM_COUNT],
            graph: None,
            use_graph: false,
            layers,
            capacity,
            cache_len: 0,
            hidden,
            cfg,
            gpu,
        })
    }

    /// y[offset..] = W · x, dispatching on how the weights were stored.
    #[allow(clippy::too_many_arguments)]
    fn project_dyn(
        gpu: &Gpu,
        w: &Proj,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        params: &CudaSlice<i32>,
        y_base: usize,
        y_idx: usize,
    ) -> Result<()> {
        match w {
            Proj::F32(data) => gpu.gemv_at(data, x, y, rows, cols, params, y_base, y_idx),
            Proj::Int8 { data, scales } => {
                gpu.gemv_i8_at(data, scales, x, y, rows, cols, params, y_base, y_idx)
            }
        }
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn reset(&mut self) {
        self.cache_len = 0;
    }

    /// Enable CUDA graph capture for single-token decode.
    ///
    /// The graph is captured lazily, on the first single-token step after at
    /// least one position exists: capture requires the exact launch sequence a
    /// replay will repeat, and that sequence is only stable once the cache is
    /// non-empty.
    pub fn enable_graph(&mut self, on: bool) {
        self.use_graph = on;
        if !on {
            self.graph = None;
        }
    }

    pub fn graph_active(&self) -> bool {
        self.graph.is_some()
    }

    /// Write this step's scalars into the buffer the kernels read.
    fn set_params(&mut self, token: usize, pos: usize) -> Result<()> {
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        self.host_params[PARAM_ZERO] = 0;
        self.host_params[PARAM_TOKEN] = token as i32;
        self.host_params[PARAM_POS] = pos as i32;
        self.host_params[PARAM_SEQ] = (pos + 1) as i32;
        self.host_params[PARAM_SLOT] = (pos * kv_dim) as i32;
        let host = self.host_params.clone();
        self.gpu.write_i32(&mut self.params, &host)
    }

    /// Queue every kernel for one token. Shared by the eager path and by graph
    /// capture, so a replay executes exactly what the eager path would.
    fn queue_token(&mut self) -> Result<()> {
        let cfg = &self.cfg;
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;

        {
            let s = &mut self.scratch;
            match &self.tok_emb {
                Proj::F32(t) => self.gpu.embed(t, &mut s.x, &self.params, d)?,
                Proj::Int8 { data, scales } => {
                    self.gpu.embed_i8(data, scales, &mut s.x, &self.params, d)?
                }
            }
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let layer_base = l * self.capacity * kv_dim;
            let s = &mut self.scratch;

            self.gpu.rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;

            // K and V land directly in the cache; the slot offset comes from
            // the parameter buffer so the graph stays valid as pos advances.
            Self::project_dyn(&self.gpu, &layer.k_proj, &s.normed, &mut self.k_cache,
                              kv_dim, d, &self.params, layer_base, PARAM_SLOT)?;
            Self::project_dyn(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_cache,
                              kv_dim, d, &self.params, layer_base, PARAM_SLOT)?;
            self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin,
                             n_kv, hd, &self.params, layer_base, PARAM_SLOT)?;

            Self::project_dyn(&self.gpu, &layer.q_proj, &s.normed, &mut s.q,
                              d, d, &self.params, 0, PARAM_ZERO)?;
            self.gpu.rope_at(&mut s.q, &self.rope_cos, &self.rope_sin,
                             n_head, hd, &self.params, 0, PARAM_ZERO)?;

            self.gpu.attention_decode(&s.q, &self.k_cache, &self.v_cache, &mut s.attn,
                                      n_head, n_kv, hd, &self.params,
                                      self.capacity, kv_dim, layer_base)?;

            Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.proj,
                              d, d, &self.params, 0, PARAM_ZERO)?;
            self.gpu.add_inplace(&mut s.x, &s.proj, d)?;

            self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
            match &layer.gate_proj {
                Some(gate) => {
                    Self::project_dyn(&self.gpu, gate, &s.normed, &mut s.gate,
                                      self.hidden, d, &self.params, 0, PARAM_ZERO)?;
                    Self::project_dyn(&self.gpu, &layer.up_proj, &s.normed, &mut s.up,
                                      self.hidden, d, &self.params, 0, PARAM_ZERO)?;
                    self.gpu.silu_mul(&mut s.gate, &s.up, self.hidden)?;
                }
                None => bail!("the GPU path currently implements swiglu only"),
            }
            Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.mlp_out,
                              d, self.hidden, &self.params, 0, PARAM_ZERO)?;
            self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
        }

        let s = &mut self.scratch;
        self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
        Self::project_dyn(&self.gpu, &self.tok_emb, &s.normed, &mut s.logits,
                          cfg.vocab_size, d, &self.params, 0, PARAM_ZERO)?;
        Ok(())
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

    /// Time each stage of one decode step.
    ///
    /// Nsight Systems reports no CUDA kernel data under WSL2 virtualisation, so
    /// attribution is done here instead: the stream is synchronised between
    /// stages and each is timed on the host.
    ///
    /// The syncs serialise work that normally overlaps, so absolute totals come
    /// out higher than real decode. The proportions are what this is for --
    /// which stage to attack, not how fast the engine is.
    pub fn profile_step(&mut self, token: usize, iters: usize) -> Result<ProfileReport> {
        let cfg = self.cfg.clone();
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;

        // Each timed block ends with a stream sync, and a sync is not free. The
        // stages called most often would otherwise absorb the most overhead and
        // look expensive purely for being frequent -- which is how a
        // 768-element rmsnorm first appeared to cost more than a 50304x768
        // matmul.
        //
        // The overhead is estimated from the cheapest stage rather than by
        // timing syncs on an idle stream. That probe gave 70.8 us, which cannot
        // be right: 111 blocks would then cost 7.9 ms against a 3.9 ms measured
        // total. Syncing an idle stream is simply not the same operation as
        // syncing after queued work. `embed` copies one 768-element row and is
        // the least work any block does, so its per-call time is a defensible
        // floor for launch + sync.
        // Timed blocks per token, used to remove that overhead.
        let n_layer = cfg.n_layer;
        let calls: Vec<(String, usize)> = vec![
            ("embed".into(), 1),
            ("rmsnorm".into(), 2 * n_layer),
            ("qkv_proj".into(), n_layer),
            ("rope".into(), n_layer),
            ("attention".into(), n_layer),
            ("o_proj".into(), n_layer),
            ("mlp".into(), n_layer),
            ("residual".into(), 2 * n_layer),
            ("lm_head".into(), 1),
            ("logits_copy".into(), 1),
        ];
        let mut totals: Vec<(String, f64)> =
            calls.iter().map(|(n, _)| (n.clone(), 0.0)).collect();

        for _ in 0..iters {
            let pos = self.cache_len;
            if pos + 1 > self.capacity {
                bail!("cache full during profiling");
            }
            self.set_params(token, pos)?;
            self.gpu.sync()?;

            macro_rules! timed {
                ($slot:expr, $body:block) => {{
                    let t0 = std::time::Instant::now();
                    $body
                    self.gpu.sync()?;
                    totals[$slot].1 += t0.elapsed().as_secs_f64();
                }};
            }

            timed!(0, {
                let s = &mut self.scratch;
                match &self.tok_emb {
                    Proj::F32(t) => self.gpu.embed(t, &mut s.x, &self.params, d)?,
                    Proj::Int8 { data, scales } => {
                        self.gpu.embed_i8(data, scales, &mut s.x, &self.params, d)?
                    }
                }
            });

            for (l, layer) in self.layers.iter().enumerate() {
                let layer_base = l * self.capacity * kv_dim;
                let s = &mut self.scratch;

                timed!(1, {
                    self.gpu.rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;
                });
                timed!(2, {
                    Self::project_dyn(&self.gpu, &layer.k_proj, &s.normed, &mut self.k_cache,
                                      kv_dim, d, &self.params, layer_base, PARAM_SLOT)?;
                    Self::project_dyn(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_cache,
                                      kv_dim, d, &self.params, layer_base, PARAM_SLOT)?;
                    Self::project_dyn(&self.gpu, &layer.q_proj, &s.normed, &mut s.q,
                                      d, d, &self.params, 0, PARAM_ZERO)?;
                });
                timed!(3, {
                    self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin,
                                     n_kv, hd, &self.params, layer_base, PARAM_SLOT)?;
                    self.gpu.rope_at(&mut s.q, &self.rope_cos, &self.rope_sin,
                                     n_head, hd, &self.params, 0, PARAM_ZERO)?;
                });
                timed!(4, {
                    self.gpu.attention_decode(&s.q, &self.k_cache, &self.v_cache, &mut s.attn,
                                              n_head, n_kv, hd, &self.params,
                                              self.capacity, kv_dim, layer_base)?;
                });
                timed!(5, {
                    Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.proj,
                                      d, d, &self.params, 0, PARAM_ZERO)?;
                });
                timed!(7, {
                    self.gpu.add_inplace(&mut s.x, &s.proj, d)?;
                });
                timed!(1, {
                    self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
                });
                timed!(6, {
                    match &layer.gate_proj {
                        Some(gate) => {
                            Self::project_dyn(&self.gpu, gate, &s.normed, &mut s.gate,
                                              self.hidden, d, &self.params, 0, PARAM_ZERO)?;
                            Self::project_dyn(&self.gpu, &layer.up_proj, &s.normed, &mut s.up,
                                              self.hidden, d, &self.params, 0, PARAM_ZERO)?;
                            self.gpu.silu_mul(&mut s.gate, &s.up, self.hidden)?;
                        }
                        None => bail!("swiglu only"),
                    }
                    Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.mlp_out,
                                      d, self.hidden, &self.params, 0, PARAM_ZERO)?;
                });
                timed!(7, {
                    self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
                });
            }

            timed!(8, {
                let s = &mut self.scratch;
                self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
                Self::project_dyn(&self.gpu, &self.tok_emb, &s.normed, &mut s.logits,
                                  cfg.vocab_size, d, &self.params, 0, PARAM_ZERO)?;
            });

            let t0 = std::time::Instant::now();
            let _ = self.gpu.to_host(&self.scratch.logits)?;
            totals[9].1 += t0.elapsed().as_secs_f64();

            self.cache_len += 1;
        }

        let per_call: Vec<f64> = totals
            .iter()
            .enumerate()
            .map(|(i, (_, raw))| raw / iters as f64 / calls[i].1 as f64)
            .collect();

        // Cheapest kernel-launching stage sets the floor. logits_copy is a
        // blocking transfer rather than a launch, so it is excluded.
        let sync_cost = totals
            .iter()
            .enumerate()
            .filter(|(_, (name, _))| name != "logits_copy")
            .map(|(i, _)| per_call[i])
            .fold(f64::INFINITY, f64::min);

        let mut stages = Vec::new();
        for (i, (name, raw)) in totals.into_iter().enumerate() {
            let n = calls[i].1;
            let overhead = if name == "logits_copy" { 0.0 } else { sync_cost };
            stages.push(Stage {
                name,
                calls: n,
                raw: raw / iters as f64,
                adjusted: ((per_call[i] - overhead) * n as f64).max(0.0),
            });
        }
        Ok(ProfileReport { stages, sync_cost })
    }

    /// Append tokens to the cache and return logits for the final position.
    ///
    /// Kernels are queued without synchronising between them: the stream
    /// executes in order, so correctness holds, and the host does not stall on
    /// each launch. The single sync happens when logits are copied back.
    ///
    /// With graph mode enabled, a single-token step replays a captured graph
    /// instead of issuing ~170 launches. The graph is captured once, on the
    /// second single-token step, because capture needs the exact sequence a
    /// replay will repeat.
    pub fn forward(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
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
        for &t in tokens {
            if t >= self.cfg.vocab_size {
                bail!("token id {t} outside vocabulary {}", self.cfg.vocab_size);
            }
        }

        for &token in tokens {
            let pos = self.cache_len;
            self.set_params(token, pos)?;

            let single = tokens.len() == 1;
            match (&self.graph, self.use_graph && single && pos > 0) {
                // Warm graph: replay it.
                (Some(g), true) => self.gpu.graph_launch(g)?,
                // Graph wanted but not captured yet: capture this step.
                (None, true) => {
                    self.gpu.begin_capture()?;
                    let queued = self.queue_token();
                    // End capture unconditionally: leaving the stream in
                    // capture mode would break every subsequent launch.
                    let graph = self.gpu.end_capture();
                    queued?;
                    let graph = graph?;
                    self.gpu.graph_launch(&graph)?;
                    self.graph = Some(graph);
                }
                _ => self.queue_token()?,
            }

            self.cache_len += 1;
        }

        self.gpu.to_host(&self.scratch.logits)
    }
}
