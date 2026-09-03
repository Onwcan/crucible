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
use crate::gpu::{attn_chunks, Gpu, Proj2, TOPK_MAX, PARAM_COUNT, PARAM_POS, PARAM_SEQ, PARAM_SLOT, PARAM_TOKEN, PARAM_ZERO};
use crate::ops::RopeTable;
use crate::paged::{PagePool, SequencePages, PAGE_TOKENS};
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
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    last: CudaSlice<f32>,
}

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
                attn: gpu.alloc(capacity * d)?,
                proj: gpu.alloc(capacity * d)?,
                gate: gpu.alloc(capacity * hidden)?,
                up: gpu.alloc(capacity * hidden)?,
                last: gpu.alloc(d)?,
            },
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
        if max_batch == 0 {
            bail!("max_batch must be at least 1");
        }
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
        self.batch = Some(BatchScratch {
            tokens: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            positions: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            x: self.gpu.alloc(max_batch * d)?,
            normed: self.gpu.alloc(max_batch * d)?,
            q: self.gpu.alloc(max_batch * d)?,
            kv: self.gpu.alloc(max_batch * kv_dim)?,
            attn: self.gpu.alloc(max_batch * d)?,
            proj: self.gpu.alloc(max_batch * d)?,
            gate: self.gpu.alloc(max_batch * self.hidden)?,
            up: self.gpu.alloc(max_batch * self.hidden)?,
            logits: self.gpu.alloc(max_batch * self.cfg.vocab_size)?,
            argmax_ids: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            row_k: self.gpu.to_device_i32(&vec![0i32; max_batch])?,
            cand_vals: self.gpu.alloc(max_batch * TOPK_MAX)?,
            cand_ids: self.gpu.to_device_i32(&vec![-1i32; max_batch * TOPK_MAX])?,
        });
        // Buffer addresses just changed, so every captured graph is stale.
        self.invalidate_batch_graphs();
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
        &mut self.pool
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
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
                self.gpu.to_host_i32_n(&self.batch.as_ref().unwrap().argmax_ids, n)?;
            } else {
                self.gpu.to_host_n(&self.batch.as_ref().unwrap().logits, n * vocab)?;
            }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            if device_argmax {
                self.gpu.to_host_i32_n(&self.batch.as_ref().unwrap().argmax_ids, n)?;
            } else {
                self.gpu.to_host_n(&self.batch.as_ref().unwrap().logits, n * vocab)?;
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

    /// Whether a batched projection of this shape should use GEMV or the
    /// tiled GEMM.
    ///
    /// Measured, int8, 200 iterations, median of 3, 158 W (speedup = gemm/gemv):
    ///
    ///   shape                b1     b2     b4     b8    b16
    ///   q/o    768x768      4.69   4.02   4.49   3.74   2.96
    ///   k/v    192x768      3.68   4.19   3.73   2.97   3.09
    ///   gate/up 2048x768    3.53   3.50   3.92   3.01   2.41
    ///   down   768x2048     7.87  10.64   7.30   6.35   5.39
    ///   lm_head 50304x768   7.76   5.56   3.57   1.77   1.00
    ///
    /// So the rule is shape-sensitive, because the shapes genuinely differ. The
    /// per-layer projections have at most 2048 output rows, which gives the
    /// GEMM only 12-32 blocks at decode M -- it is occupancy-starved and GEMV
    /// wins across the whole range. The lm_head has 50304 rows, so the GEMM
    /// already has 786 blocks and is not starved; there GEMV wins only to batch
    /// 8 and reaches parity by 10.
    ///
    /// The lm_head crossover sits on the instantiation boundary rather than
    /// anywhere physical: a batch of 10 runs the BMAX=16 kernel and pays for six
    /// accumulators it discards. Finer instantiations would move it, which is a
    /// reason not to read the constant as fundamental.
    fn use_gemv(rows: usize, batch: usize) -> bool {
        /// Below this many output rows, GEMV won at every batch measured.
        const ROWS_ALWAYS: usize = 4096;
        /// Above it, only up to this batch. Measured on the lm_head, the one
        /// shape in this model with more rows than that.
        const BIG_ROWS_MAX_BATCH: usize = 8;
        rows <= ROWS_ALWAYS || batch <= BIG_ROWS_MAX_BATCH
    }

    /// One batched projection, dispatched by measured shape and batch size.
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
            Proj::Int8 { data, scales }
                if !force_gemm && Self::use_gemv(rows, batch) && cols % 4 == 0 =>
            {
                gpu.gemv_batch_i8(data, scales, x, y, rows, cols, batch, accumulate)
            }
            // f32 weights and the large-batch lm_head keep the GEMM. f32 is not
            // the production decode path and a second GEMV instantiation family
            // for it would be untested code.
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
        return self.gpu.to_host_n(&self.batch.as_ref().unwrap().logits, rows);
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
        let (cand_vals, cand_ids) = if topk_rows.is_empty() {
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

        Ok(DecodeSelection { ids, cand_vals, cand_ids, full, d2h_bytes: d2h })
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
            let g = self.batch_graphs[slot].as_ref().expect("just captured above");
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
            self.gpu.embed_batch(&self.tok_emb.view(), &b.tokens, &mut b.x, n, d)?;
        }

        let force_gemm = self.force_decode_gemm;
        for (l, layer) in self.layers.iter().enumerate() {
            let b = self.batch.as_mut().expect("batch scratch");

            self.gpu.rmsnorm_batch(&b.x, &layer.attn_norm, &mut b.normed, n, d, NORM_EPS)?;

            // K/V go through a dense [n, kv_dim] block and are then scattered,
            // one row per request, into that request's own page.
            Self::project_batch(&self.gpu, &layer.k_proj, &b.normed, &mut b.kv,
                                kv_dim, d, n, false, force_gemm)?;
            self.gpu.rope_rows(&mut b.kv, &self.rope_cos, &self.rope_sin,
                               &b.positions, n, n_kv, hd, kv_dim)?;
            self.gpu.cache_store_rows_paged(&b.kv, &mut self.k_pool, &self.page_tables,
                                            &b.positions, n, kv_dim, self.table_stride,
                                            cfg.n_layer, l)?;

            Self::project_batch(&self.gpu, &layer.v_proj, &b.normed, &mut b.kv,
                                kv_dim, d, n, false, force_gemm)?;
            self.gpu.cache_store_rows_paged(&b.kv, &mut self.v_pool, &self.page_tables,
                                            &b.positions, n, kv_dim, self.table_stride,
                                            cfg.n_layer, l)?;

            Self::project_batch(&self.gpu, &layer.q_proj, &b.normed, &mut b.q,
                                d, d, n, false, force_gemm)?;
            self.gpu.rope_rows(&mut b.q, &self.rope_cos, &self.rope_sin,
                               &b.positions, n, n_head, hd, d)?;

            self.gpu.attention_decode_paged(
                &b.q, &self.k_pool, &self.v_pool, &mut b.attn,
                &self.page_tables, &self.seq_lens,
                n, n_head, n_kv, hd, self.table_stride,
                cfg.n_layer, l, kv_dim, self.capacity,
            )?;

            // Residual folded into the projection, the same fusion the
            // single-request decode path uses: one kernel instead of two, and
            // no [batch, d] intermediate.
            Self::project_batch(&self.gpu, &layer.o_proj, &b.attn, &mut b.x,
                                d, d, n, true, force_gemm)?;

            self.gpu.rmsnorm_batch(&b.x, &layer.mlp_norm, &mut b.normed, n, d, NORM_EPS)?;
            match &layer.gate_proj {
                Some(gate) => {
                    Self::project_batch(&self.gpu, gate, &b.normed, &mut b.gate,
                                        self.hidden, d, n, false, force_gemm)?;
                    Self::project_batch(&self.gpu, &layer.up_proj, &b.normed, &mut b.up,
                                        self.hidden, d, n, false, force_gemm)?;
                    self.gpu.swiglu_batch(&mut b.gate, &b.up, n * self.hidden)?;
                }
                None => bail!("the GPU path currently implements swiglu only"),
            }
            Self::project_batch(&self.gpu, &layer.down_proj, &b.gate, &mut b.x,
                                d, self.hidden, n, true, force_gemm)?;
        }

        let b = self.batch.as_mut().expect("batch scratch");
        self.gpu.rmsnorm_batch(&b.x, &self.final_norm, &mut b.normed, n, d, NORM_EPS)?;
        Self::project_batch(&self.gpu, &self.tok_emb, &b.normed, &mut b.logits,
                            cfg.vocab_size, d, n, false, force_gemm)?;
        // Inside the graph, so the token ids are ready the moment replay ends
        // and the step's only transfer is n * 4 bytes. Runs unconditionally:
        // the full-logit path ignores the result, and a single graph per batch
        // size then serves both paths.
        self.gpu.argmax_rows(&b.logits, &mut b.argmax_ids, n, cfg.vocab_size)?;
        // Candidate extraction only when some row wants it, so an all-greedy
        // step executes exactly the sequence it did before sampling existed.
        // *Which* rows want it, and with what k, still comes from `row_k` -- a
        // device buffer written per step -- so a batch whose sampling
        // composition changes between steps needs no recapture. Only the
        // presence of the launch is baked in, and that is what the two graphs
        // per batch size are for.
        if topk {
            self.gpu.topk_rows(&b.logits, &b.row_k, &mut b.cand_vals, &mut b.cand_ids,
                               n, cfg.vocab_size)?;
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

            self.gpu.rmsnorm(&s.x, &layer.attn_norm, &mut s.normed, d, NORM_EPS)?;

            // K and V land directly in the cache; the slot offset comes from
            // the parameter buffer so the graph stays valid as pos advances.
            if self.use_paged {
                Self::project_dyn(&self.gpu, &layer.k_proj, &s.normed, &mut self.k_pool,
                                  kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                Self::project_dyn(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_pool,
                                  kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                self.gpu.rope_at(&mut self.k_pool, &self.rope_cos, &self.rope_sin,
                                 n_kv, hd, &self.params, layer_base, PARAM_SLOT)?;
            } else {
                Self::project_dyn(&self.gpu, &layer.k_proj, &s.normed, &mut self.k_cache,
                                  kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                Self::project_dyn(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_cache,
                                  kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin,
                                 n_kv, hd, &self.params, layer_base, PARAM_SLOT)?;
            }

            Self::project_dyn(&self.gpu, &layer.q_proj, &s.normed, &mut s.q,
                              d, d, &self.params, 0, PARAM_ZERO, false)?;
            self.gpu.rope_at(&mut s.q, &self.rope_cos, &self.rope_sin,
                             n_head, hd, &self.params, 0, PARAM_ZERO)?;

            if self.use_paged {
                self.gpu.attention_decode_paged(
                    &s.q, &self.k_pool, &self.v_pool, &mut s.attn,
                    &self.page_tables, &self.seq_lens,
                    1, n_head, n_kv, hd, self.table_stride,
                    cfg.n_layer, l, kv_dim, self.capacity,
                )?;
            } else if self.split_attention {
                self.gpu.attention_split(
                    &s.q, &self.k_cache, &self.v_cache,
                    &mut s.partial_o, &mut s.partial_m, &mut s.partial_l,
                    &mut s.attn, n_head, n_kv, hd, &self.params,
                    self.capacity, kv_dim, layer_base,
                )?;
            } else {
                self.gpu.attention_decode(
                    &s.q, &self.k_cache, &self.v_cache, &mut s.attn,
                    n_head, n_kv, hd, &self.params,
                    self.capacity, kv_dim, layer_base,
                )?;
            }

            // Fused residual: the projection accumulates straight into the
            // stream, removing a kernel. Only the warp-per-row path supports
            // it, so f32 keeps the separate add.
            if self.precision == Precision::Int8 {
                Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.x,
                                  d, d, &self.params, 0, PARAM_ZERO, true)?;
            } else {
                Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.proj,
                                  d, d, &self.params, 0, PARAM_ZERO, false)?;
                self.gpu.add_inplace(&mut s.x, &s.proj, d)?;
            }

            self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
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
                Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.x,
                                  d, self.hidden, &self.params, 0, PARAM_ZERO, true)?;
            } else {
                Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.mlp_out,
                                  d, self.hidden, &self.params, 0, PARAM_ZERO, false)?;
                self.gpu.add_inplace(&mut s.x, &s.mlp_out, d)?;
            }
        }

        let s = &mut self.scratch;
        self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
        Self::project_dyn(&self.gpu, &self.tok_emb, &s.normed, &mut s.logits,
                          cfg.vocab_size, d, &self.params, 0, PARAM_ZERO, false)?;
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
        let mut totals: Vec<(String, f64)> =
            calls.iter().map(|(n, _)| (n.clone(), 0.0)).collect();

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
                self.gpu.embed_batch(&self.tok_emb.view(), &b.tokens, &mut b.x, n, d)?;
            });

            for (l, layer) in self.layers.iter().enumerate() {
                let b = self.batch.as_mut().expect("batch scratch");
                timed!(1, {
                    self.gpu.rmsnorm_batch(&b.x, &layer.attn_norm, &mut b.normed, n, d, NORM_EPS)?;
                });
                timed!(2, {
                    Self::project_batch(&self.gpu, &layer.k_proj, &b.normed, &mut b.kv,
                                        kv_dim, d, n, false, force_gemm)?;
                });
                timed!(3, {
                    self.gpu.rope_rows(&mut b.kv, &self.rope_cos, &self.rope_sin,
                                       &b.positions, n, n_kv, hd, kv_dim)?;
                });
                timed!(4, {
                    self.gpu.cache_store_rows_paged(&b.kv, &mut self.k_pool, &self.page_tables,
                                                    &b.positions, n, kv_dim, self.table_stride,
                                                    cfg.n_layer, l)?;
                });
                timed!(2, {
                    Self::project_batch(&self.gpu, &layer.v_proj, &b.normed, &mut b.kv,
                                        kv_dim, d, n, false, force_gemm)?;
                });
                timed!(4, {
                    self.gpu.cache_store_rows_paged(&b.kv, &mut self.v_pool, &self.page_tables,
                                                    &b.positions, n, kv_dim, self.table_stride,
                                                    cfg.n_layer, l)?;
                });
                timed!(2, {
                    Self::project_batch(&self.gpu, &layer.q_proj, &b.normed, &mut b.q,
                                        d, d, n, false, force_gemm)?;
                });
                timed!(3, {
                    self.gpu.rope_rows(&mut b.q, &self.rope_cos, &self.rope_sin,
                                       &b.positions, n, n_head, hd, d)?;
                });
                timed!(5, {
                    self.gpu.attention_decode_paged(
                        &b.q, &self.k_pool, &self.v_pool, &mut b.attn,
                        &self.page_tables, &self.seq_lens,
                        n, n_head, n_kv, hd, self.table_stride,
                        cfg.n_layer, l, kv_dim, self.capacity,
                    )?;
                });
                // Residual fused into the projection, as decode_batch does.
                timed!(6, {
                    Self::project_batch(&self.gpu, &layer.o_proj, &b.attn, &mut b.x,
                                        d, d, n, true, force_gemm)?;
                });
                timed!(1, {
                    self.gpu.rmsnorm_batch(&b.x, &layer.mlp_norm, &mut b.normed, n, d, NORM_EPS)?;
                });
                match &layer.gate_proj {
                    Some(gate) => {
                        timed!(7, {
                            Self::project_batch(&self.gpu, gate, &b.normed, &mut b.gate,
                                                self.hidden, d, n, false, force_gemm)?;
                        });
                        timed!(7, {
                            Self::project_batch(&self.gpu, &layer.up_proj, &b.normed, &mut b.up,
                                                self.hidden, d, n, false, force_gemm)?;
                        });
                        timed!(8, {
                            self.gpu.swiglu_batch(&mut b.gate, &b.up, n * self.hidden)?;
                        });
                    }
                    None => bail!("the GPU path currently implements swiglu only"),
                }
                timed!(9, {
                    Self::project_batch(&self.gpu, &layer.down_proj, &b.gate, &mut b.x,
                                        d, self.hidden, n, true, force_gemm)?;
                });
            }

            let b = self.batch.as_mut().expect("batch scratch");
            timed!(1, {
                self.gpu.rmsnorm_batch(&b.x, &self.final_norm, &mut b.normed, n, d, NORM_EPS)?;
            });
            timed!(11, {
                Self::project_batch(&self.gpu, &self.tok_emb, &b.normed, &mut b.logits,
                                    cfg.vocab_size, d, n, false, force_gemm)?;
            });
            timed!(12, {
                let _ = self.gpu.to_host_n(&self.batch.as_ref().unwrap().logits,
                                           n * cfg.vocab_size)?;
            });
        }

        let per_call: Vec<f64> = totals
            .iter()
            .enumerate()
            .map(|(i, (_, raw))| {
                let c = calls[i].1;
                if c == 0 { 0.0 } else { raw / iters as f64 / c as f64 }
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
            let overhead = if name == "logits_copy" { 0.0 } else { sync_cost };
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
                                      kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                    Self::project_dyn(&self.gpu, &layer.v_proj, &s.normed, &mut self.v_cache,
                                      kv_dim, d, &self.params, layer_base, PARAM_SLOT, false)?;
                    Self::project_dyn(&self.gpu, &layer.q_proj, &s.normed, &mut s.q,
                                      d, d, &self.params, 0, PARAM_ZERO, false)?;
                });
                timed!(3, {
                    self.gpu.rope_at(&mut self.k_cache, &self.rope_cos, &self.rope_sin,
                                     n_kv, hd, &self.params, layer_base, PARAM_SLOT)?;
                    self.gpu.rope_at(&mut s.q, &self.rope_cos, &self.rope_sin,
                                     n_head, hd, &self.params, 0, PARAM_ZERO)?;
                });
                timed!(4, {
                    self.gpu.attention_split(
                        &s.q, &self.k_cache, &self.v_cache,
                        &mut s.partial_o, &mut s.partial_m, &mut s.partial_l,
                        &mut s.attn, n_head, n_kv, hd, &self.params,
                        self.capacity, kv_dim, layer_base,
                    )?;
                });
                timed!(5, {
                    if int8 {
                        // Residual folded into the projection.
                        Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.x,
                                          d, d, &self.params, 0, PARAM_ZERO, true)?;
                    } else {
                        Self::project_dyn(&self.gpu, &layer.o_proj, &s.attn, &mut s.proj,
                                          d, d, &self.params, 0, PARAM_ZERO, false)?;
                    }
                });
                if !int8 {
                    timed!(7, {
                        self.gpu.add_inplace(&mut s.x, &s.proj, d)?;
                    });
                }
                timed!(1, {
                    self.gpu.rmsnorm(&s.x, &layer.mlp_norm, &mut s.normed, d, NORM_EPS)?;
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
                        Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.x,
                                          d, self.hidden, &self.params, 0, PARAM_ZERO, true)?;
                    } else {
                        Self::project_dyn(&self.gpu, &layer.down_proj, &s.gate, &mut s.mlp_out,
                                          d, self.hidden, &self.params, 0, PARAM_ZERO, false)?;
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
                self.gpu.rmsnorm(&s.x, &self.final_norm, &mut s.normed, d, NORM_EPS)?;
                Self::project_dyn(&self.gpu, &self.tok_emb, &s.normed, &mut s.logits,
                                  cfg.vocab_size, d, &self.params, 0, PARAM_ZERO, false)?;
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
        self.host_tables[..self.table_stride]
            .copy_from_slice(&table[..self.table_stride]);
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

    /// The prefill compute itself. Assumes page tables are already uploaded
    /// when paged, and touches neither `self.seq` nor `self.cache_len`.
    fn prefill_body(&mut self, tokens: &[usize], pos_offset: usize) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let (d, hd, n_head, n_kv) = (cfg.n_embd, cfg.head_dim(), cfg.n_head, cfg.n_kv_head);
        let kv_dim = n_kv * hd;
        let t = tokens.len();

        let ids: Vec<i32> = tokens.iter().map(|v| *v as i32).collect();
        self.gpu.write_i32(&mut self.prefill_scratch.tokens, &ids)?;

        {
            let p = &mut self.prefill_scratch;
            self.gpu.embed_batch(&self.tok_emb.view(), &p.tokens, &mut p.x, t, d)?;
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let layer_base = l * self.capacity * kv_dim;
            let p = &mut self.prefill_scratch;

            self.gpu.rmsnorm_batch(&p.x, &layer.attn_norm, &mut p.normed, t, d, NORM_EPS)?;

            // K and V go through a dense [T, kv_dim] buffer and are then placed
            // into the cache, so prefill and decode share one cache layout.
            self.gpu.gemm(&layer.k_proj.view(), &p.normed, &mut p.kv, t, kv_dim, d, false)?;
            self.gpu.rope_batch(&mut p.kv, &self.rope_cos, &self.rope_sin,
                                t, n_kv, hd, kv_dim, pos_offset)?;
            if self.use_paged {
                self.gpu.cache_store_paged(&p.kv, &mut self.k_pool, &self.page_tables,
                                           t, kv_dim, cfg.n_layer, l, pos_offset)?;
            } else {
                self.gpu.cache_store(&p.kv, &mut self.k_cache, t, kv_dim, layer_base, pos_offset)?;
            }

            self.gpu.gemm(&layer.v_proj.view(), &p.normed, &mut p.kv, t, kv_dim, d, false)?;
            if self.use_paged {
                self.gpu.cache_store_paged(&p.kv, &mut self.v_pool, &self.page_tables,
                                           t, kv_dim, cfg.n_layer, l, pos_offset)?;
            } else {
                self.gpu.cache_store(&p.kv, &mut self.v_cache, t, kv_dim, layer_base, pos_offset)?;
            }

            self.gpu.gemm(&layer.q_proj.view(), &p.normed, &mut p.q, t, d, d, false)?;
            self.gpu.rope_batch(&mut p.q, &self.rope_cos, &self.rope_sin,
                                t, n_head, hd, d, pos_offset)?;

            if self.use_paged {
                self.gpu.attention_prefill_paged(&p.q, &self.k_pool, &self.v_pool,
                                                 &mut p.attn, &self.page_tables,
                                                 t, n_head, n_kv, hd,
                                                 cfg.n_layer, l, kv_dim,
                                                 self.capacity, pos_offset)?;
            } else {
                self.gpu.attention_prefill(&p.q, &self.k_cache, &self.v_cache, &mut p.attn,
                                           t, n_head, n_kv, hd, self.capacity,
                                           kv_dim, layer_base, pos_offset)?;
            }

            self.gpu.gemm(&layer.o_proj.view(), &p.attn, &mut p.proj, t, d, d, false)?;
            self.gpu.add_inplace(&mut p.x, &p.proj, t * d)?;

            self.gpu.rmsnorm_batch(&p.x, &layer.mlp_norm, &mut p.normed, t, d, NORM_EPS)?;
            match &layer.gate_proj {
                Some(gate) => {
                    self.gpu.gemm(&gate.view(), &p.normed, &mut p.gate, t, self.hidden, d, false)?;
                    self.gpu.gemm(&layer.up_proj.view(), &p.normed, &mut p.up,
                                  t, self.hidden, d, false)?;
                    self.gpu.swiglu_batch(&mut p.gate, &p.up, t * self.hidden)?;
                }
                None => bail!("the GPU path currently implements swiglu only"),
            }
            self.gpu.gemm(&layer.down_proj.view(), &p.gate, &mut p.proj,
                          t, d, self.hidden, false)?;
            self.gpu.add_inplace(&mut p.x, &p.proj, t * d)?;
        }

        // Only the last position's logits are needed, so this stays a GEMV over
        // one row rather than a [T, vocab] matrix.
        {
            let p = &self.prefill_scratch;
            let last_row = p.x.slice((t - 1) * d..t * d);
            self.gpu.copy_rows(&last_row, &mut self.scratch.normed, d)?;
        }
        self.gpu.rmsnorm(&self.scratch.normed.clone(), &self.final_norm,
                         &mut self.scratch.x, d, NORM_EPS)?;
        Self::project_dyn(&self.gpu, &self.tok_emb, &self.scratch.x,
                          &mut self.scratch.logits, cfg.vocab_size, d,
                          &self.params, 0, PARAM_ZERO, false)?;

        self.gpu.to_host(&self.scratch.logits)
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
