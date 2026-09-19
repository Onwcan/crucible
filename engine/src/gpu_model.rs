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

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use cudarc::driver::{CudaEvent, CudaGraph, CudaSlice};

use crate::config::Config;
use crate::gpu::{
    attn_chunks, Gpu, PrefillAttentionVariant, Proj2, PARAM_COUNT, PARAM_POS, PARAM_PREFILL_POS,
    PARAM_SEQ, PARAM_SLOT, PARAM_TOKEN, PARAM_ZERO, TOPK_MAX,
};
use crate::ops::RopeTable;
use crate::paged::{PagePool, SequencePages, PAGE_TOKENS};
use crate::quant::QuantTensor;
use crate::weights::Weights;

const NORM_EPS: f32 = 1e-6;

/// Opt-in CUDA-event attribution for packed prefill. These are device-stream
/// intervals, including event instrumentation and any submission gaps, rather
/// than a replacement for uninstrumented end-to-end benchmark timings.
pub struct PackedPrefillStage {
    pub name: &'static str,
    /// Timed GPU operations per pass, aggregated over all transformer layers.
    /// This is an operation count, not a request count or a kernel trace.
    pub calls: usize,
    /// Mean summed event intervals for this stage in one complete pass.
    pub milliseconds: f64,
}

pub struct PackedPrefillProfile {
    pub stages: Vec<PackedPrefillStage>,
    /// Mean start-to-end event interval, including event instrumentation and
    /// host submission gaps, but excluding metadata H2D and output readback.
    pub device_milliseconds: f64,
}

/// Created only by the explicit profiling API, before its timed passes. Serving
/// instantiates the queue helpers with PROFILE=false, removing all event code.
struct PackedPrefillEvents {
    events: Vec<CudaEvent>,
    labels: Vec<&'static str>,
}

impl PackedPrefillEvents {
    fn new(gpu: &Gpu, boundaries: usize) -> Result<Self> {
        let mut events = Vec::with_capacity(boundaries + 1);
        for _ in 0..=boundaries {
            events.push(
                gpu.ctx
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(|e| anyhow::anyhow!("CUDA profiling event: {e:?}"))?,
            );
        }
        Ok(Self {
            events,
            labels: Vec::with_capacity(boundaries),
        })
    }

    fn start(&mut self, gpu: &Gpu) -> Result<()> {
        self.labels.clear();
        self.events[0]
            .record(&gpu.stream)
            .map_err(|e| anyhow::anyhow!("CUDA profiling event: {e:?}"))
    }

    fn mark(&mut self, gpu: &Gpu, name: &'static str) -> Result<()> {
        let next = self.labels.len() + 1;
        let event = self
            .events
            .get(next)
            .ok_or_else(|| anyhow::anyhow!("packed profiling event capacity exceeded"))?;
        event
            .record(&gpu.stream)
            .map_err(|e| anyhow::anyhow!("CUDA profiling event: {e:?}"))?;
        self.labels.push(name);
        Ok(())
    }

    /// Called only after the entire pass has completed. cudarc synchronizes
    /// each queried event too, but these are already complete: no stage-wise
    /// synchronization is inserted into the model's execution.
    fn accumulate(&self, report: &mut PackedPrefillProfile, iters: usize) -> Result<()> {
        for (index, &name) in self.labels.iter().enumerate() {
            let elapsed = self.events[index]
                .elapsed_ms(&self.events[index + 1])
                .map_err(|e| anyhow::anyhow!("CUDA profiling elapsed time: {e:?}"))?;
            let stage = match report.stages.iter().position(|s| s.name == name) {
                Some(index) => &mut report.stages[index],
                None => {
                    report.stages.push(PackedPrefillStage {
                        name,
                        calls: 0,
                        milliseconds: 0.0,
                    });
                    report.stages.last_mut().expect("just inserted")
                }
            };
            stage.calls += 1;
            stage.milliseconds += elapsed as f64 / iters as f64;
        }
        report.device_milliseconds += self.events[0]
            .elapsed_ms(&self.events[self.labels.len()])
            .map_err(|e| anyhow::anyhow!("CUDA profiling elapsed time: {e:?}"))?
            as f64
            / iters as f64;
        Ok(())
    }
}

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
    Int8 {
        data: CudaSlice<i8>,
        scales: CudaSlice<f32>,
    },
}

impl Proj {
    fn view(&self) -> Proj2<'_> {
        match self {
            Proj::F32(d) => Proj2::F32(d),
            Proj::Int8 { data, scales } => Proj2::Int8(data, scales),
        }
    }

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

    /// Per-chunk partial softmax state for split attention: one local max, one
    /// local sum, and one unnormalised value vector per (head, chunk).
    partial_o: CudaSlice<f32>,
    partial_m: CudaSlice<f32>,
    partial_l: CudaSlice<f32>,
}

/// Buffers for processing a whole prompt at once.
///
/// Held separately from the decode scratch because they scale with prompt
/// length rather than being single vectors, and because decode never touches
/// them. At a 1024-token capacity this is ~30 MB against 471 MB of weights and
/// cache -- cheap for removing two orders of magnitude of launch overhead.
///
/// Note there is no [T, vocab] buffer: only the final position's logits are
/// ever needed, so the vocabulary projection stays a GEMV over one row. A full
/// [1024, 50304] result would be 206 MB on its own.
struct PrefillScratch {
    tokens: CudaSlice<i32>,
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    // Retained rotated K only for the explicit hybrid attention experiment.
    current_k: Option<CudaSlice<f32>>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    last: CudaSlice<f32>,
}

/// A transient view of one request's contiguous prompt chunk. Request identity
/// remains in the scheduler; descriptor indices live for this GPU call only.
pub struct PackedPrefillRequest<'a> {
    pub tokens: &'a [usize],
    pub page_table: &'a [i32],
    pub pos_offset: usize,
    pub want_logits: bool,
}

/// Persistent packed metadata. Tensor scratch is shared with reference prefill;
/// completed rows reuse decode scratch after that iteration's decode has ended.
struct PackedPrefillScratch {
    owners: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    segments: CudaSlice<i32>,
    host_segments: Vec<i32>,
    requests: usize,
    max_chunk: usize,
    max_history: usize,
    final_rows: CudaSlice<i32>,
    host_tokens: Vec<i32>,
    host_owners: Vec<i32>,
    host_positions: Vec<i32>,
    host_final_rows: Vec<i32>,
    page_owner: Vec<i32>,
    selection_kind: Vec<u8>,
    prepared_shape: Option<(usize, usize)>,
}

/// Prefill graphs kept at once.
///
/// Chunked prefill needs a handful: every full chunk shares one key and only
/// each prompt's tail differs. Monolithic prefill needs one per distinct prompt
/// length, which is unbounded in principle, so the cache stops growing here and
/// anything past it runs eager -- correct, just without the saving.
const MAX_PREFILL_GRAPHS: usize = 64;

/// Buffers for one decode step across several requests.
///
/// Shaped `[max_batch][...]`, allocated once by `enable_paging`. Unlike the
/// prefill scratch this one does carry a full `[batch, vocab]` logits buffer:
/// every request needs its own row, and at batch 16 that is 3.2 MB rather than
/// the 206 MB a full prompt would have cost.
struct BatchScratch {
    tokens: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    /// One token id per request, produced on the device.
    ///
    /// Persistent so it can be written from inside a captured graph and read
    /// back afterwards.
    argmax_ids: CudaSlice<i32>,

    /// Candidates to extract for each slot; 0 means "greedy, skip this row".
    ///
    /// Uploaded with the rest of the per-step metadata. A device buffer rather
    /// than a kernel argument precisely so the sampling composition of a batch
    /// can change between graph replays: a graph bakes in launch arguments, but
    /// not the contents of the buffers they point at.
    row_k: CudaSlice<i32>,
    /// `[max_batch, TOPK_MAX]` candidate logits and ids, canonically ordered.
    ///
    /// Written from inside the graph, like `argmax_ids`, so a sampled step's
    /// transfer is these two small blocks rather than a full logits row per
    /// sampled request.
    cand_vals: CudaSlice<f32>,
    cand_ids: CudaSlice<i32>,
}

/// What one decode step brought back from the device.
///
/// Three shapes, because one batch can want all three at once: every row gets
/// an argmax id, rows sampling within the kernel's candidate capacity get
/// candidates, and rows asking for more candidates than it can hold get their
/// full logits row.
pub struct DecodeSelection {
    /// Argmax token id for every row. Always populated.
    pub ids: Vec<usize>,
    /// `[n, TOPK_MAX]` candidate values; only requested rows are meaningful.
    pub cand_vals: Vec<f32>,
    /// `[n, TOPK_MAX]` candidate ids; `-1` past a row's k.
    pub cand_ids: Vec<i32>,
    /// `[full_rows.len(), vocab_size]` logits, in the order `full_rows` gave.
    pub full: Vec<f32>,
    /// Bytes this step actually copied device to host.
    pub d2h_bytes: usize,
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
    ///
    /// The contiguous cache. Still the default, and still the reference the
    /// paged path is validated against.
    k_cache: CudaSlice<f32>,
    v_cache: CudaSlice<f32>,

    /// Paged KV: `[n_pages][n_layer][PAGE_TOKENS][kv_dim]`.
    ///
    /// Allocated lazily by `enable_paging`, because a model used only for
    /// single-request decode should not pay for a second cache.
    k_pool: CudaSlice<f32>,
    v_pool: CudaSlice<f32>,
    pool: PagePool,
    /// Page tables for every batch slot, `[max_batch][table_stride]`.
    page_tables: CudaSlice<i32>,
    /// Logical length per batch slot. Zero means the slot is inactive, which
    /// the attention kernel checks before touching any page.
    seq_lens: CudaSlice<i32>,
    host_tables: Vec<i32>,
    host_lens: Vec<i32>,
    /// Last value uploaded to the batch scratch's `row_k`.
    ///
    /// Kept so an unchanged value is not re-uploaded. Every other per-step
    /// buffer genuinely changes each step; this one is constant across a run
    /// of same-shaped steps, and an all-greedy run would otherwise pay a
    /// host-to-device copy per step that the engine did not make before.
    host_row_k: Vec<i32>,
    table_stride: usize,
    max_batch: usize,
    /// The single-request sequence, when running paged through `forward`.
    seq: SequencePages,
    use_paged: bool,
    batch: Option<BatchScratch>,
    /// Force the tiled GEMM for every batched projection, for A/B measurement.
    force_decode_gemm: bool,

    /// Captured decode graphs, indexed by exact active count minus one.
    ///
    /// Keyed by the exact batch rather than bucketed, because `n` drives five
    /// separate things -- grid dimensions, the batch kernel argument,
    /// attention's grid.y, which GEMV BMAX instantiation runs, and the lm_head
    /// GEMV/GEMM dispatch -- and bucketing would have to round `n` up to graph
    /// capacity for all five. Rounding the last one up silently changes the
    /// measured dispatch policy; masking the others needs an active flag
    /// threaded through the KV scatter, since padded page-table slots read 0
    /// and an inactive row would write into physical page 0, corrupting
    /// whichever request owns it. Sixteen small graphs are cheaper than that
    /// proof.
    ///
    /// Two per batch size: one whose kernel sequence ends at the argmax, and
    /// one that also extracts sampling candidates. A single graph carrying the
    /// top-k launch unconditionally would work -- `row_k` already makes it a
    /// no-op for greedy rows -- but it measured a consistent 1% off batch-1
    /// greedy, which is a launch this engine used not to make. Two graphs cost
    /// a few milliseconds of capture and leave the greedy path executing
    /// exactly the sequence it executed before sampling existed.
    batch_graphs: Vec<Option<CudaGraph>>,
    use_batch_graph: bool,

    /// Captured prefill graphs, keyed by chunk length and whether the chunk
    /// produces logits.
    ///
    /// Length is in the key because it sets every grid dimension in the
    /// sequence -- there is no padding scheme here, for the same reason the
    /// decode graphs are per exact batch size: masking rows that must not write
    /// KV, must not shift attention and must not affect logits is a correctness
    /// proof, and exact lengths need no proof at all.
    ///
    /// `want_logits` is in the key because a non-final chunk deliberately skips
    /// the lm_head projection. Folding both into one graph would put that work
    /// back, which the previous milestone removed on purpose.
    ///
    /// Positions and pages stay dynamic. Exact attention also keys the bounded
    /// shared-score capacity (256/512/768/1024 for this model), since that changes
    /// launch topology. Full chunks reuse a graph within each capacity bucket;
    /// the existing 64-entry limit still bounds the whole cache.
    prefill_graphs: HashMap<(usize, bool, usize), CudaGraph>,
    /// Keys whose capture failed. Retrying every call would pay the failure
    /// cost forever; eager execution is correct, so it is the fallback.
    prefill_graph_failed: HashSet<(usize, bool, usize)>,
    use_prefill_graph: bool,
    prefill_graphs_captured: usize,
    prefill_graph_capture_secs: f64,
    prefill_graph_replays: usize,
    /// Wall time spent capturing, and how many shapes were captured.
    graph_capture_secs: f64,
    graphs_captured: usize,
    /// Take token ids from the device instead of copying full logits back.
    use_device_argmax: bool,
    /// Extract sampling candidates on the device instead of copying full logits
    /// back for sampled rows.
    ///
    /// `CRUCIBLE_DEVICE_TOPK=0` turns it off, which routes every sampled row
    /// through the full-logit path. That path stays as the reference the device
    /// kernel is A/B'd against, so this switch is a measurement tool, not a
    /// deprecated branch waiting to be deleted.
    use_device_topk: bool,
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

    prefill_scratch: PrefillScratch,
    prefill_attention: PrefillAttentionVariant,
    // Exact attention's bounded shared-score allocation is graph topology.
    // Actual positions and page IDs remain dynamic descriptor contents.
    prefill_score_capacity: usize,
    packed_prefill: Option<PackedPrefillScratch>,

    /// Split-position attention, versus one block per head.
    ///
    /// Off by default: it was implemented to fix what profiling identified as
    /// the largest stage, it is numerically exact, and it did not help.
    ///
    ///   decode tok/s, int8 + graph, median of three
    ///                  256 tokens   900 tokens
    ///   single              1484         1424
    ///   split               1305         1389
    ///
    /// Splitting gives the grid n_head*n_chunks blocks instead of n_head, but
    /// costs a second kernel dispatch per layer -- 12 more per token -- and at
    /// this size that costs about what the extra parallelism saves. It is
    /// clearly worse at short context and a wash at long.
    ///
    /// Kept because the trade should invert with more heads, a larger head_dim,
    /// or a context well beyond 1024, where attention work grows but the extra
    /// dispatch does not. Set CRUCIBLE_ATTN=split to measure it.
    split_attention: bool,

    /// Batched prefill. On by default; CRUCIBLE_PREFILL=serial forces the
    /// token-at-a-time path, which is what the two are compared against.
    use_batched_prefill: bool,
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
            bail!(
                "the GPU path currently implements rope only, not {}",
                cfg.pos_encoding
            );
        }
        if cfg.norm != "rmsnorm" {
            bail!(
                "the GPU path currently implements rmsnorm only, not {}",
                cfg.norm
            );
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

        let prefill_attention: PrefillAttentionVariant = std::env::var("CRUCIBLE_PREFILL_ATTN")
            .unwrap_or_else(|_| "exact-q4-k64".to_owned())
            .parse()?;
        let hybrid = prefill_attention.uses_hybrid();

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
                partial_o: gpu.alloc(cfg.n_head * attn_chunks(capacity) * cfg.head_dim())?,
                partial_m: gpu.alloc(cfg.n_head * attn_chunks(capacity))?,
                partial_l: gpu.alloc(cfg.n_head * attn_chunks(capacity))?,
            },
            prefill_scratch: PrefillScratch {
                tokens: gpu.to_device_i32(&vec![0i32; capacity])?,
                x: gpu.alloc(capacity * d)?,
                normed: gpu.alloc(capacity * d)?,
                q: gpu.alloc(capacity * d)?,
                kv: gpu.alloc(capacity * kv_dim)?,
                current_k: if hybrid {
                    Some(gpu.alloc(capacity * kv_dim)?)
                } else {
                    None
                },
                attn: gpu.alloc(capacity * d)?,
                proj: gpu.alloc(capacity * d)?,
                gate: gpu.alloc(capacity * hidden)?,
                up: gpu.alloc(capacity * hidden)?,
                last: gpu.alloc(d)?,
            },
            packed_prefill: None,
            prefill_attention,
            prefill_score_capacity: 0,
            k_cache: gpu.alloc(cfg.n_layer * capacity * kv_dim)?,
            v_cache: gpu.alloc(cfg.n_layer * capacity * kv_dim)?,
            // Paging starts switched off and unallocated; `enable_paging`
            // sizes the pool for the workload that actually needs it.
            k_pool: gpu.alloc(1)?,
            v_pool: gpu.alloc(1)?,
            pool: PagePool::new(0, cfg.n_layer, kv_dim),
            page_tables: gpu.to_device_i32(&[0i32])?,
            seq_lens: gpu.to_device_i32(&[0i32])?,
            host_tables: vec![0i32; 1],
            host_row_k: vec![0i32; 1],
            host_lens: vec![0i32; 1],
            table_stride: 1,
            max_batch: 0,
            seq: SequencePages::new(),
            use_paged: false,
            batch: None,
            force_decode_gemm: std::env::var("CRUCIBLE_DECODE_GEMM").is_ok(),
            batch_graphs: Vec::new(),
            // On by default; CRUCIBLE_BATCH_GRAPH=0 keeps the eager path for
            // A/B measurement and debugging.
            use_batch_graph: std::env::var("CRUCIBLE_BATCH_GRAPH").as_deref() != Ok("0"),
            prefill_graphs: HashMap::new(),
            prefill_graph_failed: HashSet::new(),
            // On by default; CRUCIBLE_PREFILL_GRAPH=0 keeps the eager path,
            // which is the A/B control every prefill comparison is paired
            // against.
            use_prefill_graph: std::env::var("CRUCIBLE_PREFILL_GRAPH").as_deref() != Ok("0"),
            prefill_graphs_captured: 0,
            prefill_graph_capture_secs: 0.0,
            prefill_graph_replays: 0,
            graph_capture_secs: 0.0,
            graphs_captured: 0,
            use_device_argmax: std::env::var("CRUCIBLE_DEVICE_ARGMAX").as_deref() != Ok("0"),
            use_device_topk: std::env::var("CRUCIBLE_DEVICE_TOPK").as_deref() != Ok("0"),
            params: gpu.to_device_i32(&vec![0i32; PARAM_COUNT])?,
            host_params: vec![0i32; PARAM_COUNT],
            graph: None,
            use_graph: false,
            split_attention: std::env::var("CRUCIBLE_ATTN").as_deref() == Ok("split"),
            use_batched_prefill: std::env::var("CRUCIBLE_PREFILL").as_deref() != Ok("serial"),
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
        accumulate: bool,
    ) -> Result<()> {
        match w {
            Proj::F32(data) => {
                gpu.gemv_at(data, x, y, rows, cols, params, y_base, y_idx, accumulate)
            }
            Proj::Int8 { data, scales } => gpu.gemv_i8_at(
                data, scales, x, y, rows, cols, params, y_base, y_idx, accumulate,
            ),
        }
    }

    /// Allocate the page pool and switch `forward` onto the paged path.
    ///
    /// `n_pages` is total capacity across all requests; `max_batch` is how many
    /// requests may be resident at once. Sizing is explicit rather than derived
    /// because it is the memory budget, and a runtime that quietly grows its
    /// own cache is a runtime that fails at an unpredictable moment.
    pub fn enable_paging(&mut self, n_pages: usize, max_batch: usize) -> Result<()> {
        if max_batch == 0 || max_batch > Gpu::GEMV_BATCH_MAX {
            bail!("max_batch must be in 1..={}", Gpu::GEMV_BATCH_MAX);
        }
        let batch_rows = max_batch
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("max_batch exceeds representable GEMV capacity"))?;
        // PAGE_TOKENS is duplicated as a compile-time constant in the kernels
        // so translation is a shift rather than a division. If the two ever
        // disagree every paged read silently lands in the wrong page.
        if PAGE_TOKENS != 16 {
            bail!("kernels hard-code PAGE_TOKENS=16 but paged.rs says {PAGE_TOKENS}");
        }
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        let page_floats = self.cfg.n_layer * PAGE_TOKENS * kv_dim;

        // One table entry per page a single sequence could ever need.
        self.table_stride = self.capacity.div_ceil(PAGE_TOKENS);
        self.max_batch = max_batch;
        self.pool = PagePool::new(n_pages, self.cfg.n_layer, kv_dim);
        self.k_pool = self.gpu.alloc(n_pages * page_floats)?;
        self.v_pool = self.gpu.alloc(n_pages * page_floats)?;
        self.host_tables = vec![0i32; max_batch * self.table_stride];
        self.host_lens = vec![0i32; max_batch];
        self.host_row_k = vec![0i32; max_batch];
        self.page_tables = self.gpu.to_device_i32(&self.host_tables.clone())?;
        self.seq_lens = self.gpu.to_device_i32(&self.host_lens.clone())?;
        self.seq = SequencePages::new();
        let d = self.cfg.n_embd;
        // The batched GEMV reads every row of its 1/2/4/8/16 instantiation,
        // including inactive accumulators. Keep their inputs in bounds even
        // when the configured request limit is not a power of two.
        self.batch = Some(BatchScratch {
            tokens: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            positions: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            x: self.gpu.alloc(batch_rows * d)?,
            normed: self.gpu.alloc(batch_rows * d)?,
            q: self.gpu.alloc(max_batch * d)?,
            kv: self.gpu.alloc(max_batch * kv_dim)?,
            attn: self.gpu.alloc(batch_rows * d)?,
            proj: self.gpu.alloc(max_batch * d)?,
            gate: self.gpu.alloc(batch_rows * self.hidden)?,
            up: self.gpu.alloc(max_batch * self.hidden)?,
            logits: self.gpu.alloc(max_batch * self.cfg.vocab_size)?,
            argmax_ids: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            row_k: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            cand_vals: self.gpu.alloc(max_batch * TOPK_MAX)?,
            cand_ids: self.gpu.to_device_i32(&vec![-1i32; max_batch * TOPK_MAX])?,
        });
        self.packed_prefill = Some(PackedPrefillScratch {
            owners: self.gpu.to_device_i32(&vec![0; self.capacity])?,
            positions: self.gpu.to_device_i32(&vec![0; self.capacity])?,
            segments: self.gpu.to_device_i32(&vec![0; 4 * max_batch])?,
            host_segments: vec![0; 4 * max_batch],
            requests: 0,
            max_chunk: 0,
            max_history: 0,
            final_rows: self.gpu.to_device_i32(&vec![0; max_batch])?,
            host_tokens: vec![0; self.capacity],
            host_owners: vec![0; self.capacity],
            host_positions: vec![0; self.capacity],
            host_final_rows: vec![0; max_batch],
            page_owner: vec![-1; n_pages],
            selection_kind: vec![0; max_batch],
            prepared_shape: None,
        });
        // Buffer addresses just changed, so every captured graph is stale.
        self.invalidate_batch_graphs();
        self.invalidate_prefill_graphs();
        self.batch_graphs = (0..2 * max_batch).map(|_| None).collect();
        self.use_paged = true;
        // A captured contiguous-path graph would replay contiguous kernels.
        self.graph = None;
        Ok(())
    }

    pub fn use_paged(&self) -> bool {
        self.use_paged
    }

    /// Switch back to the contiguous cache, keeping the pool allocated.
    pub fn set_paged(&mut self, on: bool) {
        if on != self.use_paged {
            self.graph = None;
        }
        self.use_paged = on;
    }

    pub fn page_pool(&self) -> &PagePool {
        &self.pool
    }

    /// The page allocator, so a scheduler can grow and release sequences it
    /// owns. The pool lives here because the device memory it hands out does.
    pub fn page_pool_mut(&mut self) -> &mut PagePool {
        // A caller may release/reassign any descriptor's pages through this
        // borrow. A benchmark must prepare again before replaying them.
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        &mut self.pool
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn prefill_token_capacity(&self) -> usize {
        // Attention uses grid.y for actual rows (CUDA's limit is 65535),
        // while row-wise kernels use signed32 element counts. Internal GEMM
        // tiles may pad their loads, but no semantic row is padded.
        let kv_dim = self.cfg.n_kv_head.saturating_mul(self.cfg.head_dim());
        let widest = self.cfg.n_embd.max(self.hidden).max(kv_dim).max(1);
        self.capacity.min(65_535).min(i32::MAX as usize / widest)
    }

    pub fn prefill_request_capacity(&self) -> usize {
        self.max_batch.min(Gpu::GEMV_BATCH_MAX)
    }

    /// Force the tiled GEMM for batched projections instead of GEMV.
    ///
    /// Changes which kernels a step launches, so any captured graph is stale.
    pub fn set_force_decode_gemm(&mut self, on: bool) {
        if on != self.force_decode_gemm {
            self.invalidate_batch_graphs();
        }
        self.force_decode_gemm = on;
    }

    /// Enable or disable graph replay for the batched decode path.
    ///
    /// Disabling does not drop the captured graphs: replay and eager execution
    /// queue the same kernels, so a graph stays valid while unused. Keeping
    /// them makes an interleaved eager-vs-graph benchmark measure replay rather
    /// than repeated capture.
    pub fn set_batch_graph(&mut self, on: bool) {
        self.use_batch_graph = on;
    }

    pub fn batch_graph_enabled(&self) -> bool {
        self.use_batch_graph
    }

    /// Whether the scheduler should take token ids from the device.
    ///
    /// Off routes it back through full logits plus a host scan, which is the
    /// A/B control. Validation and evaluation keep using the full-logit path
    /// regardless -- this only changes what the scheduler asks for.
    pub fn set_device_argmax(&mut self, on: bool) {
        self.use_device_argmax = on;
    }

    pub fn device_argmax(&self) -> bool {
        self.use_device_argmax
    }

    /// Whether sampled rows may take their candidates from the device.
    ///
    /// Off routes them through full logits and a host top-k, which is the A/B
    /// control. Deliberately does *not* invalidate captured graphs: the top-k
    /// launch is in the graph either way, and `row_k` -- a device buffer -- is
    /// what decides whether it does anything. That keeps the switch free to
    /// flip between interleaved benchmark trials, at the cost of leaving one
    /// predicated block exit per row in the control measurement.
    pub fn set_device_topk(&mut self, on: bool) {
        self.use_device_topk = on;
    }

    pub fn device_topk(&self) -> bool {
        self.use_device_topk
    }

    /// Candidates the device path can return for one row.
    pub fn topk_capacity(&self) -> usize {
        TOPK_MAX
    }

    /// Replay prefill from captured graphs instead of issuing every launch.
    pub fn set_prefill_graph(&mut self, on: bool) {
        self.use_prefill_graph = on;
        if !on {
            self.prefill_graphs.clear();
            self.prefill_graph_failed.clear();
        }
    }

    pub fn prefill_graph(&self) -> bool {
        self.use_prefill_graph
    }

    /// Diagnostic attention A/B control. Changing code or scratch addresses
    /// invalidates singleton captures; no request identity enters graph keys.
    pub fn set_prefill_attention(&mut self, variant: PrefillAttentionVariant) -> Result<()> {
        variant.validate()?;
        self.gpu.sync()?;
        self.invalidate_prefill_graphs();
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        if variant.uses_hybrid() && self.prefill_scratch.current_k.is_none() {
            self.prefill_scratch.current_k = Some(
                self.gpu
                    .alloc(self.capacity * self.cfg.n_kv_head * self.cfg.head_dim())?,
            );
        }
        self.prefill_attention = variant;
        Ok(())
    }

    pub fn prefill_attention(&self) -> PrefillAttentionVariant {
        self.prefill_attention
    }

    fn tiled_prefill_attention(&self) -> bool {
        self.use_paged
            && !self.prefill_attention.is_reference()
            && self.cfg.head_dim() == 64
            && self.cfg.n_head == 4 * self.cfg.n_kv_head
    }

    /// Graphs captured, replays served, and the wall time capture cost.
    pub fn prefill_graph_stats(&self) -> (usize, usize, f64) {
        (
            self.prefill_graphs_captured,
            self.prefill_graph_replays,
            self.prefill_graph_capture_secs,
        )
    }

    /// Device execution time of one prefill, with no submission cost at all.
    ///
    /// Replays the captured graph `iters` times back to back and synchronises
    /// once, so what comes back is what the GPU spends on the kernel sequence:
    /// no per-launch driver work, no metadata upload, no logits copy. Against
    /// the eager wall time it says how much of a prefill is submission and how
    /// much is execution -- which is the difference between a launch-overhead
    /// problem and a kernel problem.
    pub fn time_prefill_replay(
        &mut self,
        len: usize,
        want_logits: bool,
        iters: usize,
    ) -> Result<f64> {
        let key = (len, want_logits, self.prefill_score_capacity);
        if !self.prefill_graphs.contains_key(&key) {
            bail!("no captured prefill graph for {len} tokens; run one first");
        }
        let g = self.prefill_graphs.get(&key).expect("checked above");
        for _ in 0..3 {
            self.gpu.graph_launch(g)?;
        }
        self.gpu.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            self.gpu.graph_launch(g)?;
        }
        self.gpu.sync()?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }

    /// Drop every captured prefill graph.
    ///
    /// A graph holds the device addresses it was captured with, so anything
    /// that can move a buffer has to come through here. Nothing in a prefill
    /// graph is request-specific: token ids, page tables, sequence lengths and
    /// the chunk offset are all contents of buffers whose addresses outlive any
    /// one request, which is why one graph serves every request of that shape.
    pub fn invalidate_prefill_graphs(&mut self) {
        self.prefill_graphs.clear();
        self.prefill_graph_failed.clear();
        self.prefill_graphs_captured = 0;
        self.prefill_graph_capture_secs = 0.0;
    }

    /// Drop every captured decode graph.
    ///
    /// Called whenever something a graph baked in could have changed: buffer
    /// addresses (`enable_paging`), or which kernels run (`force_decode_gemm`).
    /// Dropping the `CudaGraph` releases the exec object, so this is also the
    /// only place graphs are freed.
    pub fn invalidate_batch_graphs(&mut self) {
        for g in self.batch_graphs.iter_mut() {
            *g = None;
        }
        self.graphs_captured = 0;
        self.graph_capture_secs = 0.0;
    }

    /// Batch sizes with a captured graph resident.
    pub fn graphs_captured(&self) -> usize {
        self.graphs_captured
    }

    /// Total time spent capturing graphs, which a steady-state throughput
    /// number must exclude.
    pub fn graph_capture_secs(&self) -> f64 {
        self.graph_capture_secs
    }

    /// Time pure graph replay for an already-captured shape.
    ///
    /// Replays back to back with a single sync at the end, so the result is the
    /// GPU's execution time for the kernel sequence with no metadata upload, no
    /// device-to-host copy and no host work. This is the floor a full step can
    /// approach, and it is a better reference than the profiler's "adjusted"
    /// figure: that one syncs between stages, which suppresses the overlap a
    /// replay gets for free and so over-estimates kernel time.
    pub fn time_graph_replay(&mut self, n: usize, iters: usize) -> Result<f64> {
        if n == 0 || 2 * n > self.batch_graphs.len() {
            bail!("no graph slot for batch {n}");
        }
        let Some(g) = &self.batch_graphs[Self::graph_slot(n, false)] else {
            bail!("no captured graph for batch {n}; run a step at that size first");
        };
        // Warm, then time.
        for _ in 0..5 {
            self.gpu.graph_launch(g)?;
        }
        self.gpu.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            self.gpu.graph_launch(g)?;
        }
        self.gpu.sync()?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }

    /// Time just the device-to-host copy each path performs per step.
    ///
    /// Isolates the transfer from everything else, so the saving can be stated
    /// as a measurement rather than inferred from byte counts.
    pub fn time_d2h(&mut self, n: usize, device_argmax: bool, iters: usize) -> Result<f64> {
        let vocab = self.cfg.vocab_size;
        for _ in 0..5 {
            if device_argmax {
                self.gpu
                    .to_host_i32_n(&self.batch.as_ref().unwrap().argmax_ids, n)?;
            } else {
                self.gpu
                    .to_host_n(&self.batch.as_ref().unwrap().logits, n * vocab)?;
            }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            if device_argmax {
                self.gpu
                    .to_host_i32_n(&self.batch.as_ref().unwrap().argmax_ids, n)?;
            } else {
                self.gpu
                    .to_host_n(&self.batch.as_ref().unwrap().logits, n * vocab)?;
            }
        }
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }

    /// Kernels launched by one batched decode step, for reporting.
    pub fn batch_step_kernels(&self) -> usize {
        // embed + per layer (2 rmsnorm, 3 proj, 2 rope, 2 cache_store,
        // attention, o_proj, gate, up, swiglu, down) + final rmsnorm + lm_head
        1 + self.cfg.n_layer * 15 + 2
    }

    /// Keep default int8 decode arithmetic independent of active batch size.
    /// The former lm_head crossover above eight requests changed f32 GEMV
    /// activations to half-rounded WMMA inputs, changing generated tokens when
    /// unrelated requests joined or retired. All supported int8 batches now
    /// retain GEMV; force_gemm remains an explicit diagnostic comparison.
    fn project_batch(
        gpu: &Gpu,
        w: &Proj,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        batch: usize,
        accumulate: bool,
        force_gemm: bool,
    ) -> Result<()> {
        match w {
            Proj::Int8 { data, scales } if !force_gemm && cols % 4 == 0 => {
                gpu.gemv_batch_i8(data, scales, x, y, rows, cols, batch, accumulate)
            }
            // f32 weights, non-vectorizable widths and explicitly forced
            // comparisons keep their existing GEMM arithmetic at every batch.
            _ => gpu.gemm(&w.view(), x, y, batch, rows, cols, accumulate),
        }
    }

    /// Validate this step's metadata and upload it.
    ///
    /// Must run outside graph capture: these are host-to-device copies from
    /// temporary buffers, and capturing one would freeze a host pointer that is
    /// gone by the next replay.
    ///
    /// `row_k` names how many candidates the top-k kernel should extract for
    /// each slot, zero meaning "skip this row". It is uploaded every step, not
    /// only when something samples: a stale value would make the kernel extract
    /// candidates for whichever request last occupied the slot.
    fn upload_batch(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
        row_k: &[i32],
    ) -> Result<()> {
        let n = tokens.len();
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        if !self.use_paged {
            bail!("batched decode requires paging; call enable_paging first");
        }
        if n > self.max_batch {
            bail!("batch of {n} exceeds max_batch {}", self.max_batch);
        }
        if positions.len() != n || lens.len() != n {
            bail!("batch metadata length mismatch");
        }
        if tables.len() != self.max_batch * self.table_stride {
            bail!(
                "page table buffer must be max_batch * table_stride = {}",
                self.max_batch * self.table_stride
            );
        }
        for &t in tokens {
            if t >= self.cfg.vocab_size {
                bail!("token id {t} outside vocabulary {}", self.cfg.vocab_size);
            }
        }

        // Slots past `n` keep a length of 0, which the attention kernel treats
        // as inactive.
        let mut host_tok = vec![0i32; self.max_batch];
        let mut host_pos = vec![0i32; self.max_batch];
        let mut host_len = vec![0i32; self.max_batch];
        for i in 0..n {
            host_tok[i] = tokens[i] as i32;
            host_pos[i] = positions[i] as i32;
            host_len[i] = lens[i];
        }
        let mut host_k = vec![0i32; self.max_batch];
        for (i, &k) in row_k.iter().enumerate().take(n) {
            host_k[i] = k;
        }
        self.host_tables.copy_from_slice(tables);
        let ht = self.host_tables.clone();
        self.gpu.write_i32(&mut self.page_tables, &ht)?;
        self.gpu.write_i32(&mut self.seq_lens, &host_len)?;
        let b = self.batch.as_mut().expect("paging allocates batch scratch");
        self.gpu.write_i32(&mut b.tokens, &host_tok)?;
        self.gpu.write_i32(&mut b.positions, &host_pos)?;
        if host_k != self.host_row_k {
            self.gpu.write_i32(&mut b.row_k, &host_k)?;
            self.host_row_k = host_k;
        }
        Ok(())
    }

    /// One decode step for `n` requests at once.
    ///
    /// Every per-request quantity arrives as an array: the token to embed, the
    /// position to rotate at, the page table to write into and attend over, and
    /// the sequence length that bounds the attention loop. Nothing is padded to
    /// the longest sequence -- a 7-position request costs a 7-position
    /// attention loop even when batched with a 511-position one.
    ///
    /// Returns `[n][vocab_size]` logits, one row per request, in the order
    /// given.
    pub fn decode_batch(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.upload_batch(tokens, positions, tables, lens, &[])?;
        self.run_decode_batch(n, false)?;
        let rows = n * self.cfg.vocab_size;
        return self
            .gpu
            .to_host_n(&self.batch.as_ref().unwrap().logits, rows);
    }

    /// One decode step returning only the argmax token id per request.
    ///
    /// The scheduler's path. Identical compute to `decode_batch` -- same graph,
    /// same kernels -- differing only in what crosses PCIe afterwards: `n * 4`
    /// bytes instead of `n * vocab_size * 4`.
    pub fn decode_batch_tokens(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
    ) -> Result<Vec<usize>> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.upload_batch(tokens, positions, tables, lens, &[])?;
        self.run_decode_batch(n, false)?;
        let ids = self
            .gpu
            .to_host_i32_n(&self.batch.as_ref().unwrap().argmax_ids, n)?;
        Ok(ids.into_iter().map(|v| v as usize).collect())
    }

    /// One decode step returning argmax ids for every request plus full logits
    /// for the rows named in `rows`.
    ///
    /// The reference path sampling is validated against, and the fallback for a
    /// request whose top-k exceeds the device kernel's capacity. Rows come back
    /// individually rather than as one `[batch, vocab]` block because sampled
    /// rows are usually a minority: with one sampled request in a batch of
    /// sixteen, copying the block would move 3.2 MB to use 200 KB of it. Each
    /// row is contiguous, so a per-row copy is a plain range and needs no
    /// gather kernel.
    ///
    /// Returns `(ids, rows_concatenated)` where the second value holds
    /// `rows.len() * vocab_size` floats in the order given.
    pub fn decode_batch_select(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
        rows: &[usize],
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let n = tokens.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        for &r in rows {
            if r >= n {
                bail!("row {r} outside a batch of {n}");
            }
        }
        let sel = self.decode_batch_mixed(tokens, positions, tables, lens, &[], rows)?;
        Ok((sel.ids, sel.full))
    }

    /// One decode step returning whatever each row's sampling policy needs.
    ///
    /// `topk_rows` names `(row, k)` pairs to extract candidates for on the
    /// device; `full_rows` names rows whose whole logits row must come back,
    /// which is where a request wanting more candidates than `TOPK_MAX` goes.
    /// Every other row is greedy and costs its four argmax bytes.
    ///
    /// The forward pass is identical either way -- same kernels, one batched
    /// pass. What differs is one kernel at the end and what crosses PCIe after
    /// it, which is the entire point: the candidate blocks are two copies of
    /// `n * TOPK_MAX` elements regardless of how many rows sampled, where the
    /// full-logit path is one 201 KB copy per sampled row.
    pub fn decode_batch_mixed(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
    ) -> Result<DecodeSelection> {
        let n = tokens.len();
        if n == 0 {
            return Ok(DecodeSelection {
                ids: Vec::new(),
                cand_vals: Vec::new(),
                cand_ids: Vec::new(),
                full: Vec::new(),
                d2h_bytes: 0,
            });
        }
        for &r in full_rows {
            if r >= n {
                bail!("row {r} outside a batch of {n}");
            }
        }
        let mut row_k = vec![0i32; n];
        for &(r, k) in topk_rows {
            if r >= n {
                bail!("row {r} outside a batch of {n}");
            }
            if k == 0 || k > TOPK_MAX {
                bail!("top-k of {k} outside the device kernel's capacity 1..={TOPK_MAX}");
            }
            row_k[r] = k as i32;
        }

        self.upload_batch(tokens, positions, tables, lens, &row_k)?;
        self.run_decode_batch(n, !topk_rows.is_empty())?;
        self.read_batch_selection(n, !topk_rows.is_empty(), full_rows)
    }

    /// Decode and completed prefill rows share the same per-row D2H routing.
    fn read_batch_selection(
        &self,
        n: usize,
        topk: bool,
        full_rows: &[usize],
    ) -> Result<DecodeSelection> {
        let vocab = self.cfg.vocab_size;
        let b = self.batch.as_ref().expect("paging allocates batch scratch");
        let ids: Vec<usize> = self
            .gpu
            .to_host_i32_n(&b.argmax_ids, n)?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mut d2h = n * std::mem::size_of::<i32>();

        // One copy each for values and ids, covering every active row rather
        // than only the sampled ones. Copying the block is cheaper than issuing
        // a separate transfer per sampled row: at batch 16 the whole thing is
        // 16 KB, and each extra transfer costs more in launch and
        // synchronisation than the bytes it saves.
        let (cand_vals, cand_ids) = if !topk {
            (Vec::new(), Vec::new())
        } else {
            let vals = self.gpu.to_host_n(&b.cand_vals, n * TOPK_MAX)?;
            let cids = self.gpu.to_host_i32_n(&b.cand_ids, n * TOPK_MAX)?;
            d2h += n * TOPK_MAX * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());
            (vals, cids)
        };

        let mut full = Vec::with_capacity(full_rows.len() * vocab);
        for &r in full_rows {
            full.extend(self.gpu.to_host_range(&b.logits, r * vocab, vocab)?);
        }
        d2h += full_rows.len() * vocab * std::mem::size_of::<f32>();

        Ok(DecodeSelection {
            ids,
            cand_vals,
            cand_ids,
            full,
            d2h_bytes: d2h,
        })
    }

    /// Bytes copied device-to-host by one step on each path.
    pub fn d2h_bytes(&self, n: usize, device_argmax: bool) -> usize {
        if device_argmax {
            n * std::mem::size_of::<i32>()
        } else {
            n * self.cfg.vocab_size * std::mem::size_of::<f32>()
        }
    }

    /// Where batch size `n` keeps its graph, with and without top-k.
    ///
    /// Which rows sample and with what k lives in `row_k`, a device buffer, so
    /// it can change every step. Only *whether any row samples at all* selects
    /// a graph, and that is one bit.
    fn graph_slot(n: usize, topk: bool) -> usize {
        (n - 1) * 2 + topk as usize
    }

    /// Capture-or-replay the decode graph for `n` active requests.
    fn run_decode_batch(&mut self, n: usize, topk: bool) -> Result<()> {
        // Capture-or-replay. Everything before this is host-to-device
        // metadata, which must stay outside the graph: memcpy_htod from a
        // temporary Vec would be captured as a node holding a dangling host
        // pointer.
        let slot = Self::graph_slot(n, topk);
        if self.use_batch_graph && slot < self.batch_graphs.len() {
            if self.batch_graphs[slot].is_none() {
                let t0 = std::time::Instant::now();
                self.gpu.begin_capture()?;
                let queued = self.queue_decode_batch(n, topk);
                // End capture unconditionally: leaving the stream in capture
                // mode would break every later launch.
                let graph = self.gpu.end_capture();
                queued?;
                self.batch_graphs[slot] = Some(graph?);
                self.graph_capture_secs += t0.elapsed().as_secs_f64();
                self.graphs_captured += 1;
            }
            let g = self.batch_graphs[slot]
                .as_ref()
                .expect("just captured above");
            self.gpu.graph_launch(g)?;
        } else {
            self.queue_decode_batch(n, topk)?;
        }
        Ok(())
    }

    /// Queue every kernel of one batched decode step.
    ///
    /// Shared by the eager path and by graph capture, so a replay executes
    /// exactly what eager execution would. Reads all per-request state from
    /// device buffers whose addresses never change, which is what makes the
    /// captured graph valid across steps: only the *contents* of those buffers
    /// differ between replays.
    ///
    /// Nothing here depends on which request occupies a slot, only on slot
    /// position, so the scheduler reordering slots with `swap_remove` cannot
    /// invalidate a graph.
    fn queue_decode_batch(&mut self, n: usize, topk: bool) -> Result<()> {
        let cfg = self.cfg.clone();
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;

        {
            let b = self.batch.as_mut().expect("batch scratch");
            self.gpu
                .embed_batch(&self.tok_emb.view(), &b.tokens, &mut b.x, n, d)?;
        }

        let force_gemm = self.force_decode_gemm;
        for (l, layer) in self.layers.iter().enumerate() {
            let b = self.batch.as_mut().expect("batch scratch");

            self.gpu
                .rmsnorm_batch(&b.x, &layer.attn_norm, &mut b.normed, n, d, NORM_EPS)?;

            // K/V go through a dense [n, kv_dim] block and are then scattered,
            // one row per request, into that request's own page.
            Self::project_batch(
                &self.gpu,
                &layer.k_proj,
                &b.normed,
                &mut b.kv,
                kv_dim,
                d,
                n,
                false,
                force_gemm,
            )?;
            self.gpu.rope_rows(
                &mut b.kv,
                &self.rope_cos,
                &self.rope_sin,
                &b.positions,
                n,
                n_kv,
                hd,
                kv_dim,
            )?;
            self.gpu.cache_store_rows_paged(
                &b.kv,
                &mut self.k_pool,
                &self.page_tables,
                &b.positions,
                n,
                kv_dim,
                self.table_stride,
                cfg.n_layer,
                l,
            )?;

            Self::project_batch(
                &self.gpu,
                &layer.v_proj,
                &b.normed,
                &mut b.kv,
                kv_dim,
                d,
                n,
                false,
                force_gemm,
            )?;
            self.gpu.cache_store_rows_paged(
                &b.kv,
                &mut self.v_pool,
                &self.page_tables,
                &b.positions,
                n,
                kv_dim,
                self.table_stride,
                cfg.n_layer,
                l,
            )?;

            Self::project_batch(
                &self.gpu,
                &layer.q_proj,
                &b.normed,
                &mut b.q,
                d,
                d,
                n,
                false,
                force_gemm,
            )?;
            self.gpu.rope_rows(
                &mut b.q,
                &self.rope_cos,
                &self.rope_sin,
                &b.positions,
                n,
                n_head,
                hd,
                d,
            )?;

            self.gpu.attention_decode_paged(
                &b.q,
                &self.k_pool,
                &self.v_pool,
                &mut b.attn,
                &self.page_tables,
                &self.seq_lens,
                n,
                n_head,
                n_kv,
                hd,
                self.table_stride,
                cfg.n_layer,
                l,
                kv_dim,
                self.capacity,
            )?;

            // Residual folded into the projection, the same fusion the
            // single-request decode path uses: one kernel instead of two, and
            // no [batch, d] intermediate.
            Self::project_batch(
                &self.gpu,
                &layer.o_proj,
                &b.attn,
                &mut b.x,
                d,
                d,
                n,
                true,
                force_gemm,
            )?;

            self.gpu
                .rmsnorm_batch(&b.x, &layer.mlp_norm, &mut b.normed, n, d, NORM_EPS)?;
            match &layer.gate_proj {
                Some(gate) => {
                    Self::project_batch(
                        &self.gpu,
                        gate,
                        &b.normed,
                        &mut b.gate,
                        self.hidden,
                        d,
                        n,
                        false,
                        force_gemm,
                    )?;
                    Self::project_batch(
                        &self.gpu,
                        &layer.up_proj,
                        &b.normed,
                        &mut b.up,
                        self.hidden,
                        d,
                        n,
                        false,
                        force_gemm,
                    )?;
                    self.gpu.swiglu_batch(&mut b.gate, &b.up, n * self.hidden)?;
                }
                None => bail!("the GPU path currently implements swiglu only"),
            }
            Self::project_batch(
                &self.gpu,
                &layer.down_proj,
                &b.gate,
                &mut b.x,
                d,
                self.hidden,
                n,
                true,
                force_gemm,
            )?;
        }

        let b = self.batch.as_mut().expect("batch scratch");
        self.gpu
            .rmsnorm_batch(&b.x, &self.final_norm, &mut b.normed, n, d, NORM_EPS)?;
        Self::project_batch(
            &self.gpu,
            &self.tok_emb,
            &b.normed,
            &mut b.logits,
            cfg.vocab_size,
            d,
            n,
            false,
            force_gemm,
        )?;
        // Inside the graph, so the token ids are ready the moment replay ends
        // and the step's only transfer is n * 4 bytes. Runs unconditionally:
        // the full-logit path ignores the result, and a single graph per batch
        // size then serves both paths.
        self.gpu
            .argmax_rows(&b.logits, &mut b.argmax_ids, n, cfg.vocab_size)?;
        // Candidate extraction only when some row wants it, so an all-greedy
        // step executes exactly the sequence it did before sampling existed.
        // *Which* rows want it, and with what k, still comes from `row_k` -- a
        // device buffer written per step -- so a batch whose sampling
        // composition changes between steps needs no recapture. Only the
        // presence of the launch is baked in, and that is what the two graphs
        // per batch size are for.
        if topk {
            self.gpu.topk_rows(
                &b.logits,
                &b.row_k,
                &mut b.cand_vals,
                &mut b.cand_ids,
                n,
                cfg.vocab_size,
            )?;
        }
        Ok(())
    }

    pub fn table_stride(&self) -> usize {
        self.table_stride
    }

    /// Pages held by the single-request sequence.
    pub fn seq_pages(&self) -> usize {
        self.seq.n_pages()
    }

    /// Allocated-but-unoccupied slots in the single-request sequence.
    pub fn seq_wasted_slots(&self) -> usize {
        self.seq.wasted_slots()
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn reset(&mut self) {
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        self.cache_len = 0;
        if self.use_paged {
            // Releasing on reset is what makes a slot reusable. Leaking here
            // would look like a slow capacity loss rather than a bug.
            let _ = self.seq.release(&mut self.pool);
        }
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
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        let kv_dim = self.cfg.n_kv_head * self.cfg.head_dim();
        self.host_params[PARAM_ZERO] = 0;
        self.host_params[PARAM_TOKEN] = token as i32;
        self.host_params[PARAM_POS] = pos as i32;
        self.host_params[PARAM_SEQ] = (pos + 1) as i32;
        // Paged: the destination splits into a per-launch layer constant
        // (folded into layer_base below) and this one per-step scalar, which is
        // why decode needs no new projection kernel and the captured graph
        // stays valid as the sequence grows.
        self.host_params[PARAM_SLOT] = if self.use_paged {
            let (page, slot) = self.seq.translate(pos)?;
            ((page as usize * self.cfg.n_layer * PAGE_TOKENS + slot) * kv_dim) as i32
        } else {
            (pos * kv_dim) as i32
        };
        let host = self.host_params.clone();
        self.gpu.write_i32(&mut self.params, &host)?;

        if self.use_paged {
            // Buffer addresses never change, only their contents, so these
            // uploads sit outside graph capture and do not invalidate a replay.
            let table = self.seq.table_padded(self.table_stride);
            self.host_tables[..self.table_stride].copy_from_slice(&table);
            self.host_lens[0] = (pos + 1) as i32;
            let (t, l) = (self.host_tables.clone(), self.host_lens.clone());
            self.gpu.write_i32(&mut self.page_tables, &t)?;
            self.gpu.write_i32(&mut self.seq_lens, &l)?;
        }
        Ok(())
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
            // Paged pages hold PAGE_TOKENS positions for every layer, so the
            // layer stride shrinks from the whole context to one page.
            let layer_base = if self.use_paged {
                l * PAGE_TOKENS * kv_dim
            } else {
                l * self.capacity * kv_dim
            };
            let s = &mut self.scratch;

            self.gpu
                .rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;

            // K and V land directly in the cache; the slot offset comes from
            // the parameter buffer so the graph stays valid as pos advances.
            if self.use_paged {
                Self::project_dyn(
                    &self.gpu,
                    &layer.k_proj,
                    &s.normed,
                    &mut self.k_pool,
                    kv_dim,
                    d,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                    false,
                )?;
                Self::project_dyn(
                    &self.gpu,
                    &layer.v_proj,
                    &s.normed,
                    &mut self.v_pool,
                    kv_dim,
                    d,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                    false,
                )?;
                self.gpu.rope_at(
                    &mut self.k_pool,
                    &self.rope_cos,
                    &self.rope_sin,
                    n_kv,
                    hd,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                )?;
            } else {
                Self::project_dyn(
                    &self.gpu,
                    &layer.k_proj,
                    &s.normed,
                    &mut self.k_cache,
                    kv_dim,
                    d,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                    false,
                )?;
                Self::project_dyn(
                    &self.gpu,
                    &layer.v_proj,
                    &s.normed,
                    &mut self.v_cache,
                    kv_dim,
                    d,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                    false,
                )?;
                self.gpu.rope_at(
                    &mut self.k_cache,
                    &self.rope_cos,
                    &self.rope_sin,
                    n_kv,
                    hd,
                    &self.params,
                    layer_base,
                    PARAM_SLOT,
                )?;
            }

            Self::project_dyn(
                &self.gpu,
                &layer.q_proj,
                &s.normed,
                &mut s.q,
                d,
                d,
                &self.params,
                0,
                PARAM_ZERO,
                false,
            )?;
            self.gpu.rope_at(
                &mut s.q,
                &self.rope_cos,
                &self.rope_sin,
                n_head,
                hd,
                &self.params,
                0,
                PARAM_ZERO,
            )?;

            if self.use_paged {
                self.gpu.attention_decode_paged(
                    &s.q,
                    &self.k_pool,
                    &self.v_pool,
                    &mut s.attn,
                    &self.page_tables,
                    &self.seq_lens,
                    1,
                    n_head,
                    n_kv,
                    hd,
                    self.table_stride,
                    cfg.n_layer,
                    l,
                    kv_dim,
                    self.capacity,
                )?;
            } else if self.split_attention {
                self.gpu.attention_split(
                    &s.q,
                    &self.k_cache,
                    &self.v_cache,
                    &mut s.partial_o,
                    &mut s.partial_m,
                    &mut s.partial_l,
                    &mut s.attn,
                    n_head,
                    n_kv,
                    hd,
                    &self.params,
                    self.capacity,
                    kv_dim,
                    layer_base,
                )?;
            } else {
                self.gpu.attention_decode(
                    &s.q,
                    &self.k_cache,
                    &self.v_cache,
                    &mut s.attn,
                    n_head,
                    n_kv,
                    hd,
                    &self.params,
                    self.capacity,
                    kv_dim,
                    layer_base,
                )?;
            }

            // Fused residual: the projection accumulates straight into the
            // stream, removing a kernel. Only the warp-per-row path supports
            // it, so f32 keeps the separate add.
            if self.precision == Precision::Int8 {
                Self::project_dyn(
                    &self.gpu,
                    &layer.o_proj,
                    &s.attn,
                    &mut s.x,
                    d,
                    d,
                    &self.params,
                    0,
                    PARAM_ZERO,
                    true,
                )?;
            } else {
                Self::project_dyn(
                    &self.gpu,
                    &layer.o_proj,
                    &s.attn,
                    &mut s.proj,
                    d,
                    d,
                    &self.params,
                    0,
                    PARAM_ZERO,
                    false,
                )?;
                self.gpu.add_inplace(&mut s.x, &s.proj, d)?;
            }

            self.gpu
                .rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
            match &layer.gate_proj {
                // One kernel instead of three: both projections and the
                // elementwise product, with no hidden-sized intermediates.
                Some(gate) => self.gpu.mlp_swiglu(
                    &gate.view(),
                    &layer.up_proj.view(),
                    &s.normed,
                    &mut s.gate,
                    self.hidden,
                    d,
                )?,
                None => bail!("the GPU path currently implements swiglu only"),
            }
            if self.precision == Precision::Int8 {
                Self::project_dyn(
                    &self.gpu,
                    &layer.down_proj,
                    &s.gate,
                    &mut s.x,
                    d,
                    self.hidden,
                    &self.params,
                    0,
                    PARAM_ZERO,
                    true,
                )?;
            } else {
                Self::project_dyn(
                    &self.gpu,
                    &layer.down_proj,
                    &s.gate,
                    &mut s.mlp_out,
                    d,
                    self.hidden,
                    &self.params,
                    0,
                    PARAM_ZERO,
                    false,
                )?;
                self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
            }
        }

        let s = &mut self.scratch;
        self.gpu
            .rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
        Self::project_dyn(
            &self.gpu,
            &self.tok_emb,
            &s.normed,
            &mut s.logits,
            cfg.vocab_size,
            d,
            &self.params,
            0,
            PARAM_ZERO,
            false,
        )?;
        Ok(())
    }

    /// Stage breakdown of one batched decode step.
    ///
    /// Not the single-request profiler with a batch argument. That one groups
    /// three projections into "qkv_proj" and folds the residual into the
    /// projection kernel, neither of which matches this path; and its per-call
    /// arithmetic assumes every stage in a group launches the same shape. Here
    /// the whole question is which projection shape dominates, so they are
    /// timed separately.
    ///
    /// Sync-overhead handling is inherited unchanged, because the failure it
    /// prevents is the same: each timed block ends with a stream sync, so a
    /// stage called 36 times absorbs 36 syncs and looks expensive for being
    /// frequent. The cost is estimated from the cheapest kernel-launching
    /// stage rather than by timing an idle-stream sync, which measured 70.8 us
    /// here and is arithmetically impossible.
    pub fn profile_batch(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
        tables: &[i32],
        lens: &[i32],
        iters: usize,
    ) -> Result<ProfileReport> {
        let n = tokens.len();
        if !self.use_paged {
            bail!("profile_batch requires paging");
        }
        let cfg = self.cfg.clone();
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;
        let nl = cfg.n_layer;

        let calls: Vec<(String, usize)> = vec![
            ("embed".into(), 1),
            ("rmsnorm".into(), 2 * nl + 1),
            ("qkv_proj".into(), 3 * nl),
            ("rope".into(), 2 * nl),
            ("cache_store".into(), 2 * nl),
            ("attention".into(), nl),
            ("o_proj".into(), nl),
            ("gate_up_proj".into(), 2 * nl),
            ("swiglu".into(), nl),
            ("down_proj".into(), nl),
            ("residual".into(), 0),
            ("lm_head".into(), 1),
            ("logits_copy".into(), 1),
        ];
        let mut totals: Vec<(String, f64)> = calls.iter().map(|(n, _)| (n.clone(), 0.0)).collect();

        let mut host_tok = vec![0i32; self.max_batch];
        let mut host_pos = vec![0i32; self.max_batch];
        let mut host_len = vec![0i32; self.max_batch];
        for i in 0..n {
            host_tok[i] = tokens[i] as i32;
            host_pos[i] = positions[i] as i32;
            host_len[i] = lens[i];
        }
        self.host_tables.copy_from_slice(tables);
        let ht = self.host_tables.clone();
        self.gpu.write_i32(&mut self.page_tables, &ht)?;
        self.gpu.write_i32(&mut self.seq_lens, &host_len)?;
        {
            let b = self.batch.as_mut().expect("batch scratch");
            self.gpu.write_i32(&mut b.tokens, &host_tok)?;
            self.gpu.write_i32(&mut b.positions, &host_pos)?;
        }

        let force_gemm = self.force_decode_gemm;
        for _ in 0..iters {
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
                let b = self.batch.as_mut().expect("batch scratch");
                self.gpu
                    .embed_batch(&self.tok_emb.view(), &b.tokens, &mut b.x, n, d)?;
            });

            for (l, layer) in self.layers.iter().enumerate() {
                let b = self.batch.as_mut().expect("batch scratch");
                timed!(1, {
                    self.gpu.rmsnorm_batch(
                        &b.x,
                        &layer.attn_norm,
                        &mut b.normed,
                        n,
                        d,
                        NORM_EPS,
                    )?;
                });
                timed!(2, {
                    Self::project_batch(
                        &self.gpu,
                        &layer.k_proj,
                        &b.normed,
                        &mut b.kv,
                        kv_dim,
                        d,
                        n,
                        false,
                        force_gemm,
                    )?;
                });
                timed!(3, {
                    self.gpu.rope_rows(
                        &mut b.kv,
                        &self.rope_cos,
                        &self.rope_sin,
                        &b.positions,
                        n,
                        n_kv,
                        hd,
                        kv_dim,
                    )?;
                });
                timed!(4, {
                    self.gpu.cache_store_rows_paged(
                        &b.kv,
                        &mut self.k_pool,
                        &self.page_tables,
                        &b.positions,
                        n,
                        kv_dim,
                        self.table_stride,
                        cfg.n_layer,
                        l,
                    )?;
                });
                timed!(2, {
                    Self::project_batch(
                        &self.gpu,
                        &layer.v_proj,
                        &b.normed,
                        &mut b.kv,
                        kv_dim,
                        d,
                        n,
                        false,
                        force_gemm,
                    )?;
                });
                timed!(4, {
                    self.gpu.cache_store_rows_paged(
                        &b.kv,
                        &mut self.v_pool,
                        &self.page_tables,
                        &b.positions,
                        n,
                        kv_dim,
                        self.table_stride,
                        cfg.n_layer,
                        l,
                    )?;
                });
                timed!(2, {
                    Self::project_batch(
                        &self.gpu,
                        &layer.q_proj,
                        &b.normed,
                        &mut b.q,
                        d,
                        d,
                        n,
                        false,
                        force_gemm,
                    )?;
                });
                timed!(3, {
                    self.gpu.rope_rows(
                        &mut b.q,
                        &self.rope_cos,
                        &self.rope_sin,
                        &b.positions,
                        n,
                        n_head,
                        hd,
                        d,
                    )?;
                });
                timed!(5, {
                    self.gpu.attention_decode_paged(
                        &b.q,
                        &self.k_pool,
                        &self.v_pool,
                        &mut b.attn,
                        &self.page_tables,
                        &self.seq_lens,
                        n,
                        n_head,
                        n_kv,
                        hd,
                        self.table_stride,
                        cfg.n_layer,
                        l,
                        kv_dim,
                        self.capacity,
                    )?;
                });
                // Residual fused into the projection, as decode_batch does.
                timed!(6, {
                    Self::project_batch(
                        &self.gpu,
                        &layer.o_proj,
                        &b.attn,
                        &mut b.x,
                        d,
                        d,
                        n,
                        true,
                        force_gemm,
                    )?;
                });
                timed!(1, {
                    self.gpu
                        .rmsnorm_batch(&b.x, &layer.mlp_norm, &mut b.normed, n, d, NORM_EPS)?;
                });
                match &layer.gate_proj {
                    Some(gate) => {
                        timed!(7, {
                            Self::project_batch(
                                &self.gpu,
                                gate,
                                &b.normed,
                                &mut b.gate,
                                self.hidden,
                                d,
                                n,
                                false,
                                force_gemm,
                            )?;
                        });
                        timed!(7, {
                            Self::project_batch(
                                &self.gpu,
                                &layer.up_proj,
                                &b.normed,
                                &mut b.up,
                                self.hidden,
                                d,
                                n,
                                false,
                                force_gemm,
                            )?;
                        });
                        timed!(8, {
                            self.gpu.swiglu_batch(&mut b.gate, &b.up, n * self.hidden)?;
                        });
                    }
                    None => bail!("the GPU path currently implements swiglu only"),
                }
                timed!(9, {
                    Self::project_batch(
                        &self.gpu,
                        &layer.down_proj,
                        &b.gate,
                        &mut b.x,
                        d,
                        self.hidden,
                        n,
                        true,
                        force_gemm,
                    )?;
                });
            }

            let b = self.batch.as_mut().expect("batch scratch");
            timed!(1, {
                self.gpu
                    .rmsnorm_batch(&b.x, &self.final_norm, &mut b.normed, n, d, NORM_EPS)?;
            });
            timed!(11, {
                Self::project_batch(
                    &self.gpu,
                    &self.tok_emb,
                    &b.normed,
                    &mut b.logits,
                    cfg.vocab_size,
                    d,
                    n,
                    false,
                    force_gemm,
                )?;
            });
            timed!(12, {
                let _ = self
                    .gpu
                    .to_host_n(&self.batch.as_ref().unwrap().logits, n * cfg.vocab_size)?;
            });
        }

        let per_call: Vec<f64> = totals
            .iter()
            .enumerate()
            .map(|(i, (_, raw))| {
                let c = calls[i].1;
                if c == 0 {
                    0.0
                } else {
                    raw / iters as f64 / c as f64
                }
            })
            .collect();

        let sync_cost = totals
            .iter()
            .enumerate()
            .filter(|(i, (name, _))| name != "logits_copy" && calls[*i].1 > 0)
            .map(|(i, _)| per_call[i])
            .fold(f64::INFINITY, f64::min);

        let mut stages = Vec::new();
        for (i, (name, raw)) in totals.into_iter().enumerate() {
            let c = calls[i].1;
            let overhead = if name == "logits_copy" {
                0.0
            } else {
                sync_cost
            };
            stages.push(Stage {
                name,
                calls: c,
                raw: raw / iters as f64,
                adjusted: ((per_call[i] - overhead) * c as f64).max(0.0),
            });
        }
        Ok(ProfileReport { stages, sync_cost })
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
    ///
    /// This mirrors `queue_token` stage for stage, including the fused paths.
    /// The two are separate functions because timing needs syncs between
    /// stages and decode must not have them, which means they can drift: an
    /// earlier version profiled the unfused MLP after decode had been fused,
    /// and reported a `residual` stage that no longer existed. Any change to
    /// `queue_token` belongs here too.
    pub fn profile_step(&mut self, token: usize, iters: usize) -> Result<ProfileReport> {
        let int8 = self.precision == Precision::Int8;
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
            ("residual".into(), if int8 { 0 } else { 2 * n_layer }),
            ("lm_head".into(), 1),
            ("logits_copy".into(), 1),
        ];
        let mut totals: Vec<(String, f64)> = calls.iter().map(|(n, _)| (n.clone(), 0.0)).collect();

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
                    self.gpu
                        .rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;
                });
                timed!(2, {
                    Self::project_dyn(
                        &self.gpu,
                        &layer.k_proj,
                        &s.normed,
                        &mut self.k_cache,
                        kv_dim,
                        d,
                        &self.params,
                        layer_base,
                        PARAM_SLOT,
                        false,
                    )?;
                    Self::project_dyn(
                        &self.gpu,
                        &layer.v_proj,
                        &s.normed,
                        &mut self.v_cache,
                        kv_dim,
                        d,
                        &self.params,
                        layer_base,
                        PARAM_SLOT,
                        false,
                    )?;
                    Self::project_dyn(
                        &self.gpu,
                        &layer.q_proj,
                        &s.normed,
                        &mut s.q,
                        d,
                        d,
                        &self.params,
                        0,
                        PARAM_ZERO,
                        false,
                    )?;
                });
                timed!(3, {
                    self.gpu.rope_at(
                        &mut self.k_cache,
                        &self.rope_cos,
                        &self.rope_sin,
                        n_kv,
                        hd,
                        &self.params,
                        layer_base,
                        PARAM_SLOT,
                    )?;
                    self.gpu.rope_at(
                        &mut s.q,
                        &self.rope_cos,
                        &self.rope_sin,
                        n_head,
                        hd,
                        &self.params,
                        0,
                        PARAM_ZERO,
                    )?;
                });
                timed!(4, {
                    self.gpu.attention_split(
                        &s.q,
                        &self.k_cache,
                        &self.v_cache,
                        &mut s.partial_o,
                        &mut s.partial_m,
                        &mut s.partial_l,
                        &mut s.attn,
                        n_head,
                        n_kv,
                        hd,
                        &self.params,
                        self.capacity,
                        kv_dim,
                        layer_base,
                    )?;
                });
                timed!(5, {
                    if int8 {
                        // Residual folded into the projection.
                        Self::project_dyn(
                            &self.gpu,
                            &layer.o_proj,
                            &s.attn,
                            &mut s.x,
                            d,
                            d,
                            &self.params,
                            0,
                            PARAM_ZERO,
                            true,
                        )?;
                    } else {
                        Self::project_dyn(
                            &self.gpu,
                            &layer.o_proj,
                            &s.attn,
                            &mut s.proj,
                            d,
                            d,
                            &self.params,
                            0,
                            PARAM_ZERO,
                            false,
                        )?;
                    }
                });
                if !int8 {
                    timed!(7, {
                        self.gpu.add_inplace(&mut s.x, &s.proj, d)?;
                    });
                }
                timed!(1, {
                    self.gpu
                        .rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
                });
                timed!(6, {
                    match &layer.gate_proj {
                        Some(gate) => self.gpu.mlp_swiglu(
                            &gate.view(),
                            &layer.up_proj.view(),
                            &s.normed,
                            &mut s.gate,
                            self.hidden,
                            d,
                        )?,
                        None => bail!("swiglu only"),
                    }
                    if int8 {
                        Self::project_dyn(
                            &self.gpu,
                            &layer.down_proj,
                            &s.gate,
                            &mut s.x,
                            d,
                            self.hidden,
                            &self.params,
                            0,
                            PARAM_ZERO,
                            true,
                        )?;
                    } else {
                        Self::project_dyn(
                            &self.gpu,
                            &layer.down_proj,
                            &s.gate,
                            &mut s.mlp_out,
                            d,
                            self.hidden,
                            &self.params,
                            0,
                            PARAM_ZERO,
                            false,
                        )?;
                    }
                });
                if !int8 {
                    timed!(7, {
                        self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
                    });
                }
            }

            timed!(8, {
                let s = &mut self.scratch;
                self.gpu
                    .rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
                Self::project_dyn(
                    &self.gpu,
                    &self.tok_emb,
                    &s.normed,
                    &mut s.logits,
                    cfg.vocab_size,
                    d,
                    &self.params,
                    0,
                    PARAM_ZERO,
                    false,
                )?;
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
            let overhead = if name == "logits_copy" {
                0.0
            } else {
                sync_cost
            };
            stages.push(Stage {
                name,
                calls: n,
                raw: raw / iters as f64,
                adjusted: ((per_call[i] - overhead) * n as f64).max(0.0),
            });
        }
        Ok(ProfileReport { stages, sync_cost })
    }

    /// Process a whole prompt in one pass and return the final position's logits.
    ///
    /// Decode and prefill are different problems. Decode has one token in
    /// flight, so every matmul is a matrix-vector product and the work is
    /// bandwidth-bound. Prefill has no sequential dependency between prompt
    /// tokens, so the same weights serve every row: matrix-matrix, compute-
    /// bound, and vastly more efficient per token.
    ///
    /// Running a prompt through the decode path costs ~150 launches per token --
    /// about 77,000 for 512 tokens, measured at 888 tok/s against llama.cpp's
    /// 92,071. This path costs ~14 launches per layer for the entire prompt
    /// regardless of its length.
    fn prefill(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        let pos_offset = self.cache_len;
        if self.use_paged {
            self.seq.grow(&mut self.pool, tokens.len())?;
            let table = self.seq.table_padded(self.table_stride);
            self.upload_slot0(&table, pos_offset + tokens.len())?;
        }
        let out = self.prefill_body(tokens, pos_offset)?;
        self.cache_len += tokens.len();
        Ok(out)
    }

    /// Upload one page table and length into batch slot 0.
    ///
    /// Slot 0 is what the single-request paged path and prefill both use; a
    /// batched decode step overwrites every slot anyway.
    fn upload_slot0(&mut self, table: &[i32], len: usize) -> Result<()> {
        if let Some(p) = self.packed_prefill.as_mut() {
            p.prepared_shape = None;
        }
        self.host_tables[..self.table_stride].copy_from_slice(&table[..self.table_stride]);
        self.host_lens[0] = len as i32;
        let (ht, hl) = (self.host_tables.clone(), self.host_lens.clone());
        self.gpu.write_i32(&mut self.page_tables, &ht)?;
        self.gpu.write_i32(&mut self.seq_lens, &hl)?;
        Ok(())
    }

    /// Prefill a prompt into pages the caller owns.
    ///
    /// Used to admit a new request: the scheduler holds that request's
    /// `SequencePages`, so the model must not touch its own. Returns the final
    /// position's logits, which is the request's first generated token.
    pub fn prefill_request(
        &mut self,
        tokens: &[usize],
        table: &[i32],
        pos_offset: usize,
    ) -> Result<Vec<f32>> {
        if !self.use_paged {
            bail!("prefill_request requires paging; call enable_paging first");
        }
        self.upload_slot0(table, pos_offset + tokens.len())?;
        self.prefill_body(tokens, pos_offset)
    }

    /// One chunk of a prompt, written into pages the caller owns.
    ///
    /// Identical compute to `prefill_request` -- same kernels, same paged
    /// writes -- except that a chunk which is not the last one skips the
    /// lm_head projection, since only the final position's logits become a
    /// token. `pos_offset` is how many of this request's tokens are already
    /// cached, which the RoPE tables, the cache stores and the attention kernel
    /// all consume: attention for chunk row `r` runs over `0 ..= pos_offset + r`,
    /// so a chunk sees the whole prefix and its own causal history, and nothing
    /// is recomputed.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[usize],
        table: &[i32],
        pos_offset: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        if !self.use_paged {
            bail!("prefill_chunk requires paging; call enable_paging first");
        }
        self.upload_slot0(table, pos_offset + tokens.len())?;
        self.prefill_body_opt(tokens, pos_offset, want_logits)
    }

    /// Consume several independent chunks in one transformer execution. Every
    /// projection sees M=sum(chunk lengths), while RoPE/KV/attention route by
    /// the real row's owner and absolute position. Selection rows enumerate
    /// only `want_logits` descriptors, in descriptor order.
    ///
    /// All bounds and page aliases are checked before the first device write.
    /// The inference thread owns this call through completion: cancellation is
    /// observed at the next scheduler boundary, never inside this sequence.
    pub fn prefill_packed(
        &mut self,
        chunks: &[PackedPrefillRequest<'_>],
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
    ) -> Result<DecodeSelection> {
        let (rows, finals) = self.upload_packed_prefill(chunks, topk_rows, full_rows)?;
        self.queue_packed_prefill(rows, finals, !topk_rows.is_empty())?;
        if finals == 0 {
            // A bounded batch has completed before pages can be cancelled and
            // recycled. This also keeps non-final execution time honest.
            self.gpu.sync()?;
            return Ok(DecodeSelection {
                ids: Vec::new(),
                cand_vals: Vec::new(),
                cand_ids: Vec::new(),
                full: Vec::new(),
                d2h_bytes: 0,
            });
        }
        self.read_batch_selection(finals, !topk_rows.is_empty(), full_rows)
    }

    /// Measured singleton dispatch: reuse the exact reference graph while
    /// retaining compact first-token selection. The logits transfer below is
    /// device-to-device into existing scratch, never a full host readback.
    pub fn prefill_single_mixed(
        &mut self,
        chunk: &PackedPrefillRequest<'_>,
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
    ) -> Result<DecodeSelection> {
        let (rows, finals) =
            self.prepare_packed_prefill(std::slice::from_ref(chunk), topk_rows, full_rows)?;
        let p = self
            .packed_prefill
            .as_ref()
            .expect("paging allocates metadata");
        self.gpu
            .write_i32(&mut self.prefill_scratch.tokens, &p.host_tokens[..rows])?;
        self.gpu
            .write_i32(&mut self.page_tables, &self.host_tables)?;
        self.host_params[PARAM_PREFILL_POS] = chunk.pos_offset as i32;
        self.gpu.write_i32(&mut self.params, &self.host_params)?;
        // Keep decode's cached row_k contents truthful even for a non-final
        // slice; a later decode may reuse exactly this sampling composition.
        self.gpu.write_i32(
            &mut self.batch.as_mut().expect("paging").row_k,
            &self.host_row_k,
        )?;
        self.run_prefill(rows, chunk.want_logits)?;
        if finals == 0 {
            self.gpu.sync()?;
            return Ok(DecodeSelection {
                ids: Vec::new(),
                cand_vals: Vec::new(),
                cand_ids: Vec::new(),
                full: Vec::new(),
                d2h_bytes: 0,
            });
        }
        let b = self
            .batch
            .as_mut()
            .expect("paging allocates selection scratch");
        let vocab = self.cfg.vocab_size;
        self.gpu
            .copy_rows(&self.scratch.logits.slice(..vocab), &mut b.logits, vocab)?;
        self.gpu
            .argmax_rows(&b.logits, &mut b.argmax_ids, 1, vocab)?;
        if !topk_rows.is_empty() {
            self.gpu.topk_rows(
                &b.logits,
                &b.row_k,
                &mut b.cand_vals,
                &mut b.cand_ids,
                1,
                vocab,
            )?;
        }
        self.read_batch_selection(1, !topk_rows.is_empty(), full_rows)
    }

    fn prepare_packed_prefill(
        &mut self,
        chunks: &[PackedPrefillRequest<'_>],
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
    ) -> Result<(usize, usize)> {
        if !self.use_paged {
            bail!("packed prefill requires paging");
        }
        if chunks.is_empty() || chunks.len() > self.prefill_request_capacity() {
            bail!(
                "packed prefill needs 1..={} requests, got {}",
                self.prefill_request_capacity(),
                chunks.len()
            );
        }
        let token_capacity = self.prefill_token_capacity();
        if self.table_stride > i32::MAX as usize {
            bail!("packed page-table stride exceeds kernel index range");
        }
        let p = self
            .packed_prefill
            .as_mut()
            .expect("paging allocates packed metadata");
        p.prepared_shape = None;
        p.requests = 0;
        p.max_chunk = 0;
        p.max_history = 0;
        p.page_owner.fill(-1);
        p.selection_kind.fill(0);
        self.host_tables.fill(0);
        let mut rows = 0usize;
        let mut finals = 0usize;
        for (owner, chunk) in chunks.iter().enumerate() {
            let end = chunk
                .pos_offset
                .checked_add(chunk.tokens.len())
                .ok_or_else(|| anyhow::anyhow!("packed position overflow"))?;
            let packed_end = rows
                .checked_add(chunk.tokens.len())
                .ok_or_else(|| anyhow::anyhow!("packed row count overflow"))?;
            if chunk.tokens.is_empty()
                || end > self.capacity
                || end > i32::MAX as usize
                || packed_end > token_capacity
            {
                bail!("packed chunk/batch exceeds {token_capacity} token capacity or is empty");
            }
            if chunk.page_table.len() != self.table_stride {
                bail!(
                    "packed page table must contain {} entries",
                    self.table_stride
                );
            }
            for &page in &chunk.page_table[..end.div_ceil(PAGE_TOKENS)] {
                if page < 0 || page as usize >= p.page_owner.len() {
                    bail!("packed page {page} outside the pool");
                }
                if p.page_owner[page as usize] != -1 {
                    bail!("packed page {page} is aliased by multiple logical pages");
                }
                p.page_owner[page as usize] = owner as i32;
            }
            for (local, &token) in chunk.tokens.iter().enumerate() {
                if token >= self.cfg.vocab_size || token > i32::MAX as usize {
                    bail!(
                        "packed token {token} outside vocabulary {}",
                        self.cfg.vocab_size
                    );
                }
                p.host_tokens[rows + local] = token as i32;
                p.host_owners[rows + local] = owner as i32;
                p.host_positions[rows + local] = (chunk.pos_offset + local) as i32;
            }
            let start = owner * self.table_stride;
            self.host_tables[start..start + self.table_stride].copy_from_slice(chunk.page_table);
            p.host_segments[4 * owner..4 * owner + 4].copy_from_slice(&[
                rows as i32,
                chunk.tokens.len() as i32,
                chunk.pos_offset as i32,
                owner as i32,
            ]);
            p.max_chunk = p.max_chunk.max(chunk.tokens.len());
            p.max_history = p.max_history.max(end);
            if chunk.want_logits {
                p.host_final_rows[finals] = (packed_end - 1) as i32;
                finals += 1;
            }
            rows = packed_end;
        }
        if finals
            .checked_mul(self.cfg.vocab_size)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            bail!("packed final logits exceed kernel index range");
        }
        for &(row, k) in topk_rows {
            if row >= finals || k == 0 || k > TOPK_MAX {
                bail!("invalid packed top-k row {row}, k={k}, final rows={finals}");
            }
            if p.selection_kind[row] != 0 {
                bail!("duplicate packed selection row {row}");
            }
            p.selection_kind[row] = 1;
        }
        for &row in full_rows {
            if row >= finals || p.selection_kind[row] != 0 {
                bail!("invalid or duplicate packed full-logit row {row}");
            }
            p.selection_kind[row] = 2;
        }
        self.host_row_k.fill(0);
        for &(row, k) in topk_rows {
            self.host_row_k[row] = k as i32;
        }
        p.requests = chunks.len();
        self.prefill_score_capacity = if matches!(
            self.prefill_attention,
            PrefillAttentionVariant::Exact { .. }
        ) {
            (p.max_history.div_ceil(256) * 256).min(self.table_stride * PAGE_TOKENS)
        } else {
            0
        };
        Ok((rows, finals))
    }

    fn upload_packed_prefill(
        &mut self,
        chunks: &[PackedPrefillRequest<'_>],
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
    ) -> Result<(usize, usize)> {
        let (rows, finals) = self.prepare_packed_prefill(chunks, topk_rows, full_rows)?;
        let p = self
            .packed_prefill
            .as_mut()
            .expect("paging allocates metadata");
        // Every consumed element is overwritten, including row_k zeros. Stale
        // capacity beyond these exact launch counts is never semantic input.
        self.gpu
            .write_i32(&mut self.prefill_scratch.tokens, &p.host_tokens[..rows])?;
        self.gpu.write_i32(&mut p.owners, &p.host_owners[..rows])?;
        self.gpu
            .write_i32(&mut p.positions, &p.host_positions[..rows])?;
        if !self.prefill_attention.is_reference() {
            self.gpu
                .write_i32(&mut p.segments, &p.host_segments[..4 * p.requests])?;
        }
        self.gpu
            .write_i32(&mut self.page_tables, &self.host_tables)?;
        if finals != 0 {
            self.gpu
                .write_i32(&mut p.final_rows, &p.host_final_rows[..finals])?;
        }
        self.gpu.write_i32(
            &mut self.batch.as_mut().expect("paging").row_k,
            &self.host_row_k,
        )?;
        p.prepared_shape = Some((rows, finals));
        Ok((rows, finals))
    }

    fn queue_packed_prefill(&mut self, rows: usize, finals: usize, topk: bool) -> Result<()> {
        self.queue_packed_prefill_impl::<false>(rows, finals, topk, None)
    }

    fn queue_packed_prefill_impl<const PROFILE: bool>(
        &mut self,
        rows: usize,
        finals: usize,
        topk: bool,
        mut events: Option<&mut PackedPrefillEvents>,
    ) -> Result<()> {
        self.queue_prefill_transformer_impl::<PROFILE>(rows, true, events.as_deref_mut())?;
        if finals == 0 {
            return Ok(());
        }
        macro_rules! mark {
            ($name:literal) => {
                if PROFILE {
                    events
                        .as_deref_mut()
                        .expect("profiling requires events")
                        .mark(&self.gpu, $name)?;
                }
            };
        }
        let b = self
            .batch
            .as_mut()
            .expect("paging allocates final-row scratch");
        let p = self
            .packed_prefill
            .as_ref()
            .expect("paging allocates metadata");
        let d = self.cfg.n_embd;
        self.gpu.gather_prefill_rows(
            &self.prefill_scratch.x,
            &mut b.x,
            &p.final_rows,
            finals,
            d,
        )?;
        mark!("final_gather");
        self.gpu
            .rmsnorm_batch(&b.x, &self.final_norm, &mut b.normed, finals, d, NORM_EPS)?;
        mark!("final_norm");
        self.gpu.project_final_rows(
            &self.tok_emb.view(),
            &b.normed,
            &mut b.logits,
            self.cfg.vocab_size,
            d,
            finals,
        )?;
        mark!("final_head");
        self.gpu
            .argmax_rows(&b.logits, &mut b.argmax_ids, finals, self.cfg.vocab_size)?;
        mark!("argmax");
        if topk {
            self.gpu.topk_rows(
                &b.logits,
                &b.row_k,
                &mut b.cand_vals,
                &mut b.cand_ids,
                finals,
                self.cfg.vocab_size,
            )?;
            mark!("topk");
        }
        Ok(())
    }

    /// Benchmark-only stage attribution using the exact packed transformer and
    /// selection queue. Metadata upload and output readback are excluded. CUDA
    /// events are allocated once before warmup, reused across `iters` complete
    /// passes, and dropped on return. No event, timer, or allocation is added
    /// to serving. Event insertion can perturb short stages; compare these
    /// proportions with the uninstrumented wall/replay timings before acting.
    pub fn profile_packed_prefill(
        &mut self,
        chunks: &[PackedPrefillRequest<'_>],
        topk_rows: &[(usize, usize)],
        full_rows: &[usize],
        iters: usize,
    ) -> Result<PackedPrefillProfile> {
        if iters == 0 {
            bail!("packed profiling needs at least one iteration");
        }
        let (rows, finals) = self.upload_packed_prefill(chunks, topk_rows, full_rows)?;
        // One boundary per GPU operation: embed, 17 per layer, up to five final
        // operations (including optional top-k). The extra event is the start.
        let mut events = PackedPrefillEvents::new(&self.gpu, 6 + 17 * self.layers.len())?;
        // Warm the event records as well as the kernels, outside the samples.
        self.gpu.sync()?;
        events.start(&self.gpu)?;
        self.queue_packed_prefill_impl::<true>(
            rows,
            finals,
            !topk_rows.is_empty(),
            Some(&mut events),
        )?;
        self.gpu.sync()?;
        let mut report = PackedPrefillProfile {
            stages: Vec::new(),
            device_milliseconds: 0.0,
        };
        for _ in 0..iters {
            events.start(&self.gpu)?;
            self.queue_packed_prefill_impl::<true>(
                rows,
                finals,
                !topk_rows.is_empty(),
                Some(&mut events),
            )?;
            self.gpu.sync()?;
            events.accumulate(&mut report, iters)?;
        }
        for stage in &mut report.stages {
            stage.calls /= iters;
        }
        Ok(report)
    }

    /// Benchmark-only graph: metadata must already be prepared by one packed
    /// call. It is dropped on return and never populates a serving graph cache.
    /// Capture cost is excluded; this isolates device execution from submission.
    pub fn time_packed_replay(&mut self, rows: usize, finals: usize, iters: usize) -> Result<f64> {
        if rows == 0
            || rows > self.capacity
            || finals > self.prefill_request_capacity()
            || iters == 0
        {
            bail!("invalid packed replay benchmark shape");
        }
        if self.packed_prefill.as_ref().and_then(|p| p.prepared_shape) != Some((rows, finals)) {
            bail!("packed replay requires a just-prepared matching batch");
        }
        self.gpu.sync()?;
        self.gpu.begin_capture()?;
        let queued = self.queue_packed_prefill(rows, finals, false);
        let graph = self.gpu.end_capture();
        queued?;
        let graph = graph?;
        for _ in 0..3 {
            self.gpu.graph_launch(&graph)?;
        }
        self.gpu.sync()?;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            self.gpu.graph_launch(&graph)?;
        }
        self.gpu.sync()?;
        Ok(start.elapsed().as_secs_f64() / iters as f64)
    }

    /// The prefill compute itself. Assumes page tables are already uploaded
    /// when paged, and touches neither `self.seq` nor `self.cache_len`.
    ///
    /// `want_logits` exists for chunked prefill. Only the final chunk of a
    /// prompt produces the token the request starts from, so every earlier
    /// chunk would otherwise pay an lm_head projection over the whole
    /// vocabulary and a 201 KB device-to-host copy for a result nobody reads --
    /// and that copy is also the only synchronisation in the function, so
    /// skipping it lets the chunk queue behind the next one instead of
    /// stalling the thread.
    fn prefill_body(&mut self, tokens: &[usize], pos_offset: usize) -> Result<Vec<f32>> {
        self.prefill_body_opt(tokens, pos_offset, true)
    }

    fn prefill_body_opt(
        &mut self,
        tokens: &[usize],
        pos_offset: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let t = tokens.len();

        let ids: Vec<i32> = tokens.iter().map(|v| *v as i32).collect();
        self.gpu.write_i32(&mut self.prefill_scratch.tokens, &ids)?;
        // Where this chunk starts in its own prompt. A kernel argument would be
        // frozen by graph capture; the parameter buffer is read at replay.
        // Both writes are host-to-device copies from temporaries, so they stay
        // outside anything captured -- a captured memcpy would hold a host
        // pointer that is gone by the next replay.
        self.host_params[PARAM_PREFILL_POS] = pos_offset as i32;
        let hp = self.host_params.clone();
        self.gpu.write_i32(&mut self.params, &hp)?;
        self.run_prefill(t, want_logits)?;
        self.prefill_result(t, want_logits)
    }

    /// Queue or replay the shared reference topology without choosing a host
    /// result representation. Full-logit and compact singleton APIs share it.
    fn run_prefill(&mut self, t: usize, want_logits: bool) -> Result<()> {
        self.prefill_score_capacity = if self.tiled_prefill_attention()
            && matches!(
                self.prefill_attention,
                PrefillAttentionVariant::Exact { .. }
            ) {
            ((t + self.host_params[PARAM_PREFILL_POS] as usize).div_ceil(256) * 256)
                .min(self.table_stride * PAGE_TOKENS)
        } else {
            0
        };
        if self.tiled_prefill_attention() {
            let p = self
                .packed_prefill
                .as_mut()
                .expect("paging allocates descriptors");
            p.host_segments[..4].copy_from_slice(&[
                0,
                t as i32,
                self.host_params[PARAM_PREFILL_POS],
                0,
            ]);
            p.requests = 1;
            p.max_chunk = t;
            self.gpu.write_i32(&mut p.segments, &p.host_segments[..4])?;
        }
        // Capture-or-replay, on exactly the sequence eager execution issues.
        // Only the paged path is eligible: the contiguous fallback still takes
        // its offset by value, and it is the legacy single-sequence path.
        let key = (t, want_logits, self.prefill_score_capacity);
        if self.use_prefill_graph && self.use_paged {
            if !self.prefill_graphs.contains_key(&key)
                && !self.prefill_graph_failed.contains(&key)
                && self.prefill_graphs.len() < MAX_PREFILL_GRAPHS
            {
                let t0 = std::time::Instant::now();
                let began = self.gpu.begin_capture();
                if began.is_ok() {
                    let queued = self.queue_prefill(t, want_logits);
                    // End capture unconditionally: leaving the stream in
                    // capture mode would break every later launch.
                    let graph = self.gpu.end_capture();
                    match (queued, graph) {
                        (Ok(()), Ok(g)) => {
                            self.prefill_graphs.insert(key, g);
                            self.prefill_graphs_captured += 1;
                            self.prefill_graph_capture_secs += t0.elapsed().as_secs_f64();
                        }
                        _ => {
                            // Eager execution is correct, so a capture failure
                            // costs speed and nothing else. Remembered so the
                            // cost is paid once rather than every call.
                            self.prefill_graph_failed.insert(key);
                        }
                    }
                } else {
                    self.prefill_graph_failed.insert(key);
                }
            }
            if let Some(g) = self.prefill_graphs.get(&key) {
                self.gpu.graph_launch(g)?;
                self.prefill_graph_replays += 1;
                return Ok(());
            }
        }
        self.queue_prefill(t, want_logits)
    }

    /// Copy back the final position's logits, if this chunk produced any.
    ///
    /// The only synchronisation in a prefill, and the only thing that cannot be
    /// captured: the graph writes the logits, the host reads them afterwards.
    fn prefill_result(&mut self, _t: usize, want_logits: bool) -> Result<Vec<f32>> {
        if !want_logits {
            return Ok(Vec::new());
        }
        self.gpu.to_host(&self.scratch.logits)
    }

    /// Queue every kernel of one prefill chunk.
    ///
    /// Shared by eager execution and by graph capture, so a replay executes
    /// exactly what eager execution would. Everything that varies between calls
    /// -- token ids, page table, sequence length, chunk offset -- is read from
    /// device buffers whose addresses never change, which is what makes a
    /// captured graph valid for a different request of the same shape.
    fn queue_prefill_transformer(&mut self, t: usize, packed: bool) -> Result<()> {
        self.queue_prefill_transformer_impl::<false>(t, packed, None)
    }

    fn queue_prefill_transformer_impl<const PROFILE: bool>(
        &mut self,
        t: usize,
        packed: bool,
        mut events: Option<&mut PackedPrefillEvents>,
    ) -> Result<()> {
        macro_rules! mark {
            ($name:literal) => {
                if PROFILE {
                    events
                        .as_deref_mut()
                        .expect("profiling requires events")
                        .mark(&self.gpu, $name)?;
                }
            };
        }
        let cfg = &self.cfg;
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;
        let tiled_attention = self.tiled_prefill_attention();
        let hybrid_attention = tiled_attention && self.prefill_attention.uses_hybrid();

        {
            let p = &mut self.prefill_scratch;
            self.gpu
                .embed_batch(&self.tok_emb.view(), &p.tokens, &mut p.x, t, d)?;
        }
        mark!("embed");

        for (l, layer) in self.layers.iter().enumerate() {
            let layer_base = l * self.capacity * kv_dim;
            // The contiguous fallback still takes the offset by value; it is
            // never captured, so reading it from the host copy is safe.
            let pos_offset = self.host_params[PARAM_PREFILL_POS] as usize;
            let p = &mut self.prefill_scratch;

            self.gpu
                .rmsnorm_batch(&p.x, &layer.attn_norm, &mut p.normed, t, d, NORM_EPS)?;
            mark!("norm");

            // K and V go through a dense [T, kv_dim] buffer and are then placed
            // into the cache, so prefill and decode share one cache layout.
            let current_k = if hybrid_attention {
                p.current_k
                    .as_mut()
                    .expect("hybrid scratch allocated before inference")
            } else {
                &mut p.kv
            };
            self.gpu.gemm(
                &layer.k_proj.view(),
                &p.normed,
                current_k,
                t,
                kv_dim,
                d,
                false,
            )?;
            mark!("qkv_gemm");
            if packed {
                let meta = self.packed_prefill.as_ref().expect("packed metadata");
                self.gpu.rope_rows(
                    current_k,
                    &self.rope_cos,
                    &self.rope_sin,
                    &meta.positions,
                    t,
                    n_kv,
                    hd,
                    kv_dim,
                )?;
            } else {
                self.gpu.rope_batch(
                    current_k,
                    &self.rope_cos,
                    &self.rope_sin,
                    t,
                    n_kv,
                    hd,
                    kv_dim,
                    &self.params,
                )?;
            }
            mark!("rope");
            if packed {
                let meta = self.packed_prefill.as_ref().expect("packed metadata");
                self.gpu.cache_store_packed_paged(
                    current_k,
                    &mut self.k_pool,
                    &self.page_tables,
                    &meta.owners,
                    &meta.positions,
                    t,
                    kv_dim,
                    self.table_stride,
                    cfg.n_layer,
                    l,
                )?;
            } else if self.use_paged {
                self.gpu.cache_store_paged(
                    current_k,
                    &mut self.k_pool,
                    &self.page_tables,
                    t,
                    kv_dim,
                    cfg.n_layer,
                    l,
                    &self.params,
                )?;
            } else {
                self.gpu.cache_store(
                    current_k,
                    &mut self.k_cache,
                    t,
                    kv_dim,
                    layer_base,
                    pos_offset,
                )?;
            }
            mark!("kv_store");

            self.gpu.gemm(
                &layer.v_proj.view(),
                &p.normed,
                &mut p.kv,
                t,
                kv_dim,
                d,
                false,
            )?;
            mark!("qkv_gemm");
            if packed {
                let meta = self.packed_prefill.as_ref().expect("packed metadata");
                self.gpu.cache_store_packed_paged(
                    &p.kv,
                    &mut self.v_pool,
                    &self.page_tables,
                    &meta.owners,
                    &meta.positions,
                    t,
                    kv_dim,
                    self.table_stride,
                    cfg.n_layer,
                    l,
                )?;
            } else if self.use_paged {
                self.gpu.cache_store_paged(
                    &p.kv,
                    &mut self.v_pool,
                    &self.page_tables,
                    t,
                    kv_dim,
                    cfg.n_layer,
                    l,
                    &self.params,
                )?;
            } else {
                self.gpu.cache_store(
                    &p.kv,
                    &mut self.v_cache,
                    t,
                    kv_dim,
                    layer_base,
                    pos_offset,
                )?;
            }
            mark!("kv_store");

            self.gpu
                .gemm(&layer.q_proj.view(), &p.normed, &mut p.q, t, d, d, false)?;
            mark!("qkv_gemm");
            if packed {
                let meta = self.packed_prefill.as_ref().expect("packed metadata");
                self.gpu.rope_rows(
                    &mut p.q,
                    &self.rope_cos,
                    &self.rope_sin,
                    &meta.positions,
                    t,
                    n_head,
                    hd,
                    d,
                )?;
            } else {
                self.gpu.rope_batch(
                    &mut p.q,
                    &self.rope_cos,
                    &self.rope_sin,
                    t,
                    n_head,
                    hd,
                    d,
                    &self.params,
                )?;
            }
            mark!("rope");

            if tiled_attention {
                let meta = self.packed_prefill.as_ref().expect("attention descriptors");
                self.gpu.attention_prefill_tiled(
                    &p.q,
                    &self.k_pool,
                    &self.v_pool,
                    p.current_k.as_ref().unwrap_or(&p.kv),
                    &p.kv,
                    &mut p.attn,
                    &self.page_tables,
                    &meta.segments,
                    meta.requests,
                    meta.max_chunk,
                    self.table_stride,
                    n_head,
                    n_kv,
                    hd,
                    cfg.n_layer,
                    l,
                    kv_dim,
                    self.prefill_attention,
                    self.prefill_score_capacity,
                )?;
            } else if packed {
                let meta = self.packed_prefill.as_ref().expect("packed metadata");
                self.gpu.attention_prefill_packed(
                    &p.q,
                    &self.k_pool,
                    &self.v_pool,
                    &mut p.attn,
                    &self.page_tables,
                    &meta.owners,
                    &meta.positions,
                    t,
                    self.table_stride,
                    n_head,
                    n_kv,
                    hd,
                    cfg.n_layer,
                    l,
                    kv_dim,
                    self.capacity,
                )?;
            } else if self.use_paged {
                self.gpu.attention_prefill_paged(
                    &p.q,
                    &self.k_pool,
                    &self.v_pool,
                    &mut p.attn,
                    &self.page_tables,
                    t,
                    n_head,
                    n_kv,
                    hd,
                    cfg.n_layer,
                    l,
                    kv_dim,
                    self.capacity,
                    &self.params,
                )?;
            } else {
                self.gpu.attention_prefill(
                    &p.q,
                    &self.k_cache,
                    &self.v_cache,
                    &mut p.attn,
                    t,
                    n_head,
                    n_kv,
                    hd,
                    self.capacity,
                    kv_dim,
                    layer_base,
                    pos_offset,
                )?;
            }
            mark!("attention");

            self.gpu
                .gemm(&layer.o_proj.view(), &p.attn, &mut p.proj, t, d, d, false)?;
            mark!("o_gemm");
            self.gpu.add_inplace(&mut p.x, &p.proj, t * d)?;
            mark!("residual");

            self.gpu
                .rmsnorm_batch(&p.x, &layer.mlp_norm, &mut p.normed, t, d, NORM_EPS)?;
            mark!("norm");
            match &layer.gate_proj {
                Some(gate) => {
                    self.gpu.gemm(
                        &gate.view(),
                        &p.normed,
                        &mut p.gate,
                        t,
                        self.hidden,
                        d,
                        false,
                    )?;
                    mark!("ffn_gate_up_gemm");
                    self.gpu.gemm(
                        &layer.up_proj.view(),
                        &p.normed,
                        &mut p.up,
                        t,
                        self.hidden,
                        d,
                        false,
                    )?;
                    mark!("ffn_gate_up_gemm");
                    self.gpu.swiglu_batch(&mut p.gate, &p.up, t * self.hidden)?;
                    mark!("swiglu");
                }
                None => bail!("the GPU path currently implements swiglu only"),
            }
            self.gpu.gemm(
                &layer.down_proj.view(),
                &p.gate,
                &mut p.proj,
                t,
                d,
                self.hidden,
                false,
            )?;
            mark!("down_gemm");
            self.gpu.add_inplace(&mut p.x, &p.proj, t * d)?;
            mark!("residual");
        }

        Ok(())
    }

    fn queue_prefill(&mut self, t: usize, want_logits: bool) -> Result<()> {
        self.queue_prefill_transformer(t, false)?;
        let d = self.cfg.n_embd;
        // Only the last position's logits are needed, so this stays a GEMV over
        // one row rather than a [T, vocab] matrix.
        if !want_logits {
            return Ok(());
        }
        {
            let p = &self.prefill_scratch;
            let last_row = p.x.slice((t - 1) * d..t * d);
            self.gpu.copy_rows(&last_row, &mut self.scratch.normed, d)?;
        }
        self.gpu.rmsnorm(
            &self.scratch.normed,
            &self.final_norm,
            &mut self.scratch.x,
            d,
            NORM_EPS,
        )?;
        Self::project_dyn(
            &self.gpu,
            &self.tok_emb,
            &self.scratch.x,
            &mut self.scratch.logits,
            self.cfg.vocab_size,
            d,
            &self.params,
            0,
            PARAM_ZERO,
            false,
        )?;
        Ok(())
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

        // More than one token means a prompt: process it as a matrix rather
        // than looping the single-token path.
        if tokens.len() > 1 && self.use_batched_prefill {
            return self.prefill(tokens);
        }

        for &token in tokens {
            let pos = self.cache_len;
            // Pages are taken one position at a time, so a short request never
            // reserves a full context. Exhaustion surfaces here as an error
            // rather than as a silent overwrite of somebody else's page.
            if self.use_paged {
                self.seq.grow(&mut self.pool, 1)?;
            }
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
