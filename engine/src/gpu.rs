//! CUDA backend.
//!
//! Kernels are compiled at runtime with NVRTC rather than offline with nvcc.
//! Two reasons: CUDA 13's headers conflict with glibc 2.43 on Ubuntu 26.04
//! (both declare `rsqrt`, with incompatible exception specifications, and nvcc
//! injects host headers even for `--ptx`), and NVRTC never touches host headers
//! so the conflict cannot arise. It also means the engine builds on machines
//! without the CUDA toolkit.
//!
//! Every kernel here has a scalar CPU twin in `ops.rs`. The CPU version is the
//! reference: when the two disagree, the GPU one is wrong. `validate()` checks
//! that claim rather than assuming it.

use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::driver::CudaGraph;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Kernel source, embedded so the binary is self-contained.
const KERNELS: &str = include_str!("../kernels/kernels.cu");

/// Blackwell. Compiling for an older architecture still runs, via PTX JIT, but
/// slower -- the exact failure this project checks for elsewhere.
const ARCH: &str = "compute_120";

/// Positions handled by one attention block.
///
/// Sets how many blocks the attention grid has: `n_head * ceil(capacity/CHUNK)`
/// instead of `n_head`. At capacity 1024 this turns 12 blocks into 96, which is
/// the point -- the single-block-per-head kernel left most of the GPU idle.
///
/// Smaller chunks mean more parallelism but more partials to combine, and more
/// shared memory per block. 128 was chosen by measurement, not derivation.
pub const ATTN_CHUNK: usize = 128;

/// Chunks needed to cover `capacity` positions.
///
/// Derived from capacity rather than the current sequence length, because a
/// captured CUDA graph freezes grid dimensions.
pub fn attn_chunks(capacity: usize) -> usize {
    capacity.div_ceil(ATTN_CHUNK)
}

/// A projection's device buffers, borrowed so a fused kernel can take two.
pub enum Proj2<'a> {
    F32(&'a CudaSlice<f32>),
    Int8(&'a CudaSlice<i8>, &'a CudaSlice<f32>),
}

/// Indices into the per-step parameter buffer that lives on the device.
///
/// A captured CUDA graph freezes kernel arguments, so anything that changes per
/// token must be read from memory rather than passed by value. Slot 0 holds a
/// permanent zero, so call sites needing a constant offset use the same code
/// path instead of needing a second kernel variant.
pub const PARAM_ZERO: usize = 0;
pub const PARAM_TOKEN: usize = 1;
pub const PARAM_POS: usize = 2;
pub const PARAM_SEQ: usize = 3;
pub const PARAM_SLOT: usize = 4;
pub const PARAM_COUNT: usize = 5;

/// Threads per block for reduction kernels.
///
/// 256 rather than 1024: a GEMV block reduces one output row, and at
/// d_model = 768 there is not enough work per row to occupy 1024 threads, while
/// smaller blocks let more rows run concurrently.
const REDUCE_THREADS: u32 = 256;

/// cudarc's error types do not implement `std::error::Error`, so `?` cannot
/// convert them into `anyhow::Error` on its own. Debug formatting carries the
/// CUresult, which is what actually identifies the failure.
fn cu<T>(r: std::result::Result<T, DriverError>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("CUDA driver error: {e:?}"))
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    gemv: CudaFunction,
    gemv_vec4: CudaFunction,
    rmsnorm: CudaFunction,
    rope: CudaFunction,
    add_inplace: CudaFunction,
    silu_mul: CudaFunction,
    softmax: CudaFunction,
    embed: CudaFunction,
    attention: CudaFunction,
    gemv_i8: CudaFunction,
    gemv_i8_vec4: CudaFunction,
    gemv_i8_warp: CudaFunction,
    gemv_warp: CudaFunction,
    mlp_swiglu_i8: CudaFunction,
    mlp_swiglu_f32: CudaFunction,
    attention_partial: CudaFunction,
    attention_combine: CudaFunction,
    gemm_i8: CudaFunction,
    gemm_f32: CudaFunction,
    gemm_i8_wmma: CudaFunction,
    gemm_f32_wmma: CudaFunction,
    gemm_i8_wmma_big: CudaFunction,
    gemm_f32_wmma_big: CudaFunction,

    /// Streaming multiprocessor count, read from the device.
    ///
    /// Used to decide which GEMM tile to launch. Hard-coding the threshold
    /// would bake this laptop's GPU into the dispatch; the useful question is
    /// "does this launch produce enough blocks to fill *this* machine", which
    /// needs the actual count.
    sm_count: usize,

    /// Use tensor cores for prefill GEMMs.
    ///
    /// On by default; CRUCIBLE_GEMM=tiled selects the scalar tiled kernel,
    /// which is what the tensor-core path is measured against. The two differ
    /// numerically -- activations convert to half for the tensor-core path --
    /// so the comparison is on cross-entropy as well as speed.
    pub use_wmma: bool,
    rmsnorm_batch: CudaFunction,
    rope_batch: CudaFunction,
    swiglu_batch: CudaFunction,
    embed_batch_f32: CudaFunction,
    embed_batch_i8: CudaFunction,
    cache_store: CudaFunction,
    attention_prefill: CudaFunction,
    cache_store_paged: CudaFunction,
    /// Batched GEMV, indexed by log2 of the instantiated BMAX.
    gemv_batch_i8: [CudaFunction; 5],
    argmax_rows: CudaFunction,
    rope_rows: CudaFunction,
    cache_store_rows_paged: CudaFunction,
    attention_decode_paged: CudaFunction,
    attention_prefill_paged: CudaFunction,
    embed_i8: CudaFunction,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = cu(CudaContext::new(ordinal))?;

        // SAFETY: cudarc records a CUDA event per buffer and inserts
        // cuStreamWaitEvent on every kernel that touches it, to order work
        // across streams. That is exactly what graph capture forbids: a
        // captured launch waiting on an event recorded by uncaptured work
        // fails with CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
        //
        // The engine uses a single stream and issues everything in program
        // order, so the stream itself already provides the ordering those
        // events exist to guarantee. Disabling the tracking is therefore safe
        // here, and required for graphs to work at all.
        //
        // Must happen before any allocation: the flag is read when a buffer is
        // created, so buffers allocated earlier would keep their events.
        unsafe { ctx.disable_event_tracking() };

        // A dedicated stream, not ctx.default_stream(). CUDA forbids capturing
        // the legacy default stream, and capture fails on it with
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED regardless of what is queued.
        let stream = cu(ctx.new_stream())?;

        // NVRTC has no include path of its own, which is why the kernels avoid
        // headers. Tensor-core code cannot: mma.h and cuda_fp16.h are required.
        // These headers are documented as NVRTC-safe, and unlike nvcc they pull
        // in no host headers -- so the CUDA 13.0 / glibc 2.43 rsqrt conflict
        // that blocks nvcc does not apply here.
        let include_paths: Vec<String> = [
            "/usr/local/cuda/include",
            "/usr/local/cuda-13/include",
            "/usr/local/cuda-13.0/targets/x86_64-linux/include",
            "/usr/local/cuda-13.3/targets/x86_64-linux/include",
        ]
        .iter()
        .filter(|p| std::path::Path::new(p).exists())
        .map(|p| p.to_string())
        .collect();

        let opts = CompileOptions {
            arch: Some(ARCH),
            use_fast_math: Some(true),
            include_paths,
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNELS, opts)
            .map_err(|e| anyhow::anyhow!("NVRTC compilation failed: {e:?}"))?;
        let module = cu(ctx.load_module(ptx))?;

        Ok(Self {
            gemv: cu(module.load_function("gemv_f32"))?,
            gemv_vec4: cu(module.load_function("gemv_f32_vec4"))?,
            rmsnorm: cu(module.load_function("rmsnorm_f32"))?,
            rope: cu(module.load_function("rope_f32"))?,
            add_inplace: cu(module.load_function("add_inplace_f32"))?,
            silu_mul: cu(module.load_function("silu_mul_f32"))?,
            softmax: cu(module.load_function("softmax_f32"))?,
            embed: cu(module.load_function("embed_f32"))?,
            attention: cu(module.load_function("attention_decode_f32"))?,
            gemv_i8: cu(module.load_function("gemv_i8_f32"))?,
            gemv_i8_vec4: cu(module.load_function("gemv_i8_f32_vec4"))?,
            gemv_i8_warp: cu(module.load_function("gemv_i8_f32_warp"))?,
            gemv_warp: cu(module.load_function("gemv_f32_warp"))?,
            mlp_swiglu_i8: cu(module.load_function("mlp_swiglu_i8_warp"))?,
            mlp_swiglu_f32: cu(module.load_function("mlp_swiglu_f32"))?,
            attention_partial: cu(module.load_function("attention_partial_f32"))?,
            attention_combine: cu(module.load_function("attention_combine_f32"))?,
            gemm_i8: cu(module.load_function("gemm_i8_f32"))?,
            gemm_f32: cu(module.load_function("gemm_f32"))?,
            gemm_i8_wmma: cu(module.load_function("gemm_i8_wmma"))?,
            gemm_f32_wmma: cu(module.load_function("gemm_f32_wmma"))?,
            gemm_i8_wmma_big: cu(module.load_function("gemm_i8_wmma_big"))?,
            gemm_f32_wmma_big: cu(module.load_function("gemm_f32_wmma_big"))?,
            sm_count: {
                use cudarc::driver::sys::CUdevice_attribute_enum as Attr;
                let dev = cu(unsafe { cudarc::driver::result::device::get(ordinal as i32) })?;
                cu(unsafe {
                    cudarc::driver::result::device::get_attribute(
                        dev,
                        Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    )
                })? as usize
            },
            use_wmma: std::env::var("CRUCIBLE_GEMM").as_deref() != Ok("tiled"),
            rmsnorm_batch: cu(module.load_function("rmsnorm_batch_f32"))?,
            rope_batch: cu(module.load_function("rope_batch_f32"))?,
            swiglu_batch: cu(module.load_function("swiglu_batch_f32"))?,
            embed_batch_f32: cu(module.load_function("embed_batch_f32"))?,
            embed_batch_i8: cu(module.load_function("embed_batch_i8"))?,
            cache_store: cu(module.load_function("cache_store_f32"))?,
            attention_prefill: cu(module.load_function("attention_prefill_f32"))?,
            cache_store_paged: cu(module.load_function("cache_store_paged_f32"))?,
            argmax_rows: cu(module.load_function("argmax_rows_f32"))?,
            gemv_batch_i8: [
                cu(module.load_function("gemv_batch_i8_b1"))?,
                cu(module.load_function("gemv_batch_i8_b2"))?,
                cu(module.load_function("gemv_batch_i8_b4"))?,
                cu(module.load_function("gemv_batch_i8_b8"))?,
                cu(module.load_function("gemv_batch_i8_b16"))?,
            ],
            rope_rows: cu(module.load_function("rope_rows_f32"))?,
            cache_store_rows_paged: cu(module.load_function("cache_store_rows_paged_f32"))?,
            attention_decode_paged: cu(module.load_function("attention_decode_paged_f32"))?,
            attention_prefill_paged: cu(module.load_function("attention_prefill_paged_f32"))?,
            embed_i8: cu(module.load_function("embed_i8"))?,
            ctx,
            stream,
        })
    }

    pub fn name(&self) -> Result<String> {
        cu(self.ctx.name())
    }

    pub fn to_device(&self, host: &[f32]) -> Result<CudaSlice<f32>> {
        cu(self.stream.memcpy_stod(host))
    }

    pub fn alloc(&self, n: usize) -> Result<CudaSlice<f32>> {
        cu(self.stream.alloc_zeros::<f32>(n))
    }

    /// Row-wise argmax on the device: `[rows, cols]` floats to `[rows]` ids.
    ///
    /// Ties resolve to the lowest index, matching the host implementation the
    /// scheduler used before.
    pub fn argmax_rows(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        const THREADS: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c) = (rows as i32, cols as i32);
        let mut b = self.stream.launch_builder(&self.argmax_rows);
        b.arg(x).arg(out).arg(&r).arg(&c);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Copy back the first `n` token ids.
    pub fn to_host_i32_n(&self, dev: &CudaSlice<i32>, n: usize) -> Result<Vec<i32>> {
        let view = dev.slice(0..n);
        cu(self.stream.memcpy_dtov(&view))
    }

    pub fn to_host(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>> {
        cu(self.stream.memcpy_dtov(dev))
    }

    /// Copy back `len` elements starting at `start`.
    ///
    /// Lets one row of a `[batch, vocab]` buffer come back without dragging the
    /// whole thing across PCIe.
    pub fn to_host_range(
        &self,
        dev: &CudaSlice<f32>,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>> {
        let view = dev.slice(start..start + len);
        cu(self.stream.memcpy_dtov(&view))
    }

    /// Copy back only the first `n` elements.
    ///
    /// The batched logits buffer is sized for `max_batch`, so a batch of one
    /// would otherwise drag the whole 3.2 MB across PCIe every step to read
    /// 200 KB of it.
    pub fn to_host_n(&self, dev: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>> {
        let view = dev.slice(0..n);
        cu(self.stream.memcpy_dtov(&view))
    }

    pub fn sync(&self) -> Result<()> {
        cu(self.stream.synchronize())
    }

    /// y[offset..] = W · x, one block per output row.
    ///
    /// The offset lets K/V projections write straight into the KV cache instead
    /// of into scratch followed by a copy -- one fewer launch per layer, and at
    /// 12 layers per token that is 12 launches saved.
    pub fn gemv_at(
        &self,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        params: &CudaSlice<i32>,
        y_base: usize,
        y_idx: usize,
        accumulate: bool,
    ) -> Result<()> {
        // The f32 kernels do not implement accumulation. Silently overwriting
        // the residual instead of adding to it produces a model that still
        // generates fluent text, so fail loudly rather than let a caller find
        // out from a perplexity check.
        if accumulate {
            anyhow::bail!("accumulate is not implemented for the f32 path");
        }
        // Block per row. A warp-per-row variant exists and is used for int8,
        // but at f32 it measured 737/736/864 tok/s against 825 for this path --
        // a possible regression sitting inside its own 17% spread, so unproven
        // in either direction and not adopted. An f32 row carries four times
        // the bytes of an int8 one, so the block is far less starved.
        let vec4 = cols % 4 == 0;
        let func = if vec4 { &self.gemv_vec4 } else { &self.gemv };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c, base, idx) = (rows as i32, cols as i32, y_base as i32, y_idx as i32);
        let acc = i32::from(accumulate);
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(x).arg(y).arg(&r).arg(&c).arg(params).arg(&base).arg(&idx).arg(&acc);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Rotary embedding applied at a byte offset into a larger buffer.
    pub fn rope_at(
        &self,
        v: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        params: &CudaSlice<i32>,
        v_base: usize,
        v_idx: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((n_heads * head_dim / 2) as u32);
        let (nh, hd, pi, base, idx) = (
            n_heads as i32,
            head_dim as i32,
            PARAM_POS as i32,
            v_base as i32,
            v_idx as i32,
        );
        let mut b = self.stream.launch_builder(&self.rope);
        b.arg(v).arg(cos).arg(sin).arg(&nh).arg(&hd).arg(params).arg(&pi).arg(&base).arg(&idx);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Copy one embedding row into the residual stream.
    pub fn embed(
        &self,
        table: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        params: &CudaSlice<i32>,
        d: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(d as u32);
        let (ti, dd) = (PARAM_TOKEN as i32, d as i32);
        let mut b = self.stream.launch_builder(&self.embed);
        b.arg(table).arg(out).arg(params).arg(&ti).arg(&dd);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Fused single-token attention over the KV cache.
    ///
    /// Scores live in dynamic shared memory, so `seq_len` floats are requested
    /// per block. One block per query head.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        params: &CudaSlice<i32>,
        max_seq: usize,
        cache_stride: usize,
        layer_base: usize,
    ) -> Result<()> {
        // The kernel indexes from the start of a layer's region, so the layer
        // offset is folded into the pointer by slicing rather than passed in.
        let k = k_cache.slice(layer_base..);
        let v = v_cache.slice(layer_base..);
        // Shared memory is sized for the maximum context, not the current
        // length: a captured graph fixes its allocation, and sizing to the
        // current length would invalidate the graph as the sequence grows.
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            // scores[max_seq] followed by one partial vector per warp.
            shared_mem_bytes: ((max_seq + (REDUCE_THREADS as usize / 32) * head_dim)
                * std::mem::size_of::<f32>()) as u32,
        };
        let (nh, nkv, hd, si, cs) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            PARAM_SEQ as i32,
            cache_stride as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention);
        let ms = max_seq as i32;
        b.arg(q).arg(&k).arg(&v).arg(out)
            .arg(&nh).arg(&nkv).arg(&hd).arg(params).arg(&si).arg(&cs).arg(&ms);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Embedding lookup from an int8 table.
    pub fn embed_i8(
        &self,
        table: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        params: &CudaSlice<i32>,
        d: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(d as u32);
        let (ti, dd) = (PARAM_TOKEN as i32, d as i32);
        let mut b = self.stream.launch_builder(&self.embed_i8);
        b.arg(table).arg(scales).arg(out).arg(params).arg(&ti).arg(&dd);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Begin capturing kernel launches on this stream.
    ///
    /// The point is launch overhead. Decoding one token issues ~170 kernels,
    /// each costing microseconds of driver work, and at this model size that
    /// overhead dominates: a token takes ~2.3 ms while the arithmetic and
    /// memory traffic together account for a fraction of it. A graph submits
    /// the whole sequence as a single operation.
    ///
    /// Split into begin/end rather than taking a closure so the caller can
    /// borrow itself mutably while queueing -- a closure capturing `&mut self`
    /// cannot coexist with the `&self.gpu` borrow needed to call this.
    ///
    /// `Relaxed` mode so unrelated work elsewhere in the process is not caught
    /// up in the capture. Nothing between begin and end may synchronise: a sync
    /// during capture aborts it.
    pub fn begin_capture(&self) -> Result<()> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        cu(self
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED))
    }

    /// Finish capture and instantiate the graph for replay.
    ///
    /// Must be called even if queueing failed, or the stream stays in capture
    /// mode and every later launch on it fails.
    pub fn end_capture(&self) -> Result<CudaGraph> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        let graph = cu(self.stream.end_capture(
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        ))?
        .ok_or_else(|| anyhow::anyhow!("stream capture produced no graph"))?;
        cu(graph.upload())?;
        Ok(graph)
    }

    /// Replay a captured graph.
    pub fn graph_launch(&self, graph: &CudaGraph) -> Result<()> {
        cu(graph.launch())
    }

    /// Copy n floats from a device view into another device buffer.
    pub fn copy_rows(
        &self,
        src: &cudarc::driver::CudaView<f32>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        let mut view = dst.slice_mut(0..n);
        cu(self.stream.memcpy_dtod(src, &mut view))
    }

    /// Upload i32 parameters.
    pub fn to_device_i32(&self, host: &[i32]) -> Result<CudaSlice<i32>> {
        cu(self.stream.memcpy_stod(host))
    }

    /// Overwrite an existing device buffer in place.
    ///
    /// Used to update per-step parameters between graph replays: the graph
    /// holds a pointer to this exact buffer, so it must be written, never
    /// reallocated.
    pub fn write_i32(&self, dev: &mut CudaSlice<i32>, host: &[i32]) -> Result<()> {
        cu(self.stream.memcpy_htod(host, dev))
    }

    /// An all-zero parameter buffer, for call sites with no per-step values.
    fn zero_params(&self) -> Result<CudaSlice<i32>> {
        cu(self.stream.alloc_zeros::<i32>(PARAM_COUNT))
    }

    /// Upload int8 weights.
    pub fn to_device_i8(&self, host: &[i8]) -> Result<CudaSlice<i8>> {
        cu(self.stream.memcpy_stod(host))
    }

    /// y[offset..] = (W_int8 · x) * row_scale.
    ///
    /// The scale is applied once per output row, after reduction, rather than
    /// dequantising each weight before multiplying.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_i8_at(
        &self,
        w: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        params: &CudaSlice<i32>,
        y_base: usize,
        y_idx: usize,
        accumulate: bool,
    ) -> Result<()> {
        // One warp per row when a block would be starved. At int8 a 768-column
        // row is 192 char4 loads against 256 threads: a quarter of the block
        // idle, one load per active thread, then a full eight-warp reduction to
        // combine them. Warp-per-row measured 1170-1299 tok/s against 852 for
        // block-per-row, a 1.42x gain well outside the spread.
        //
        // The same switch did not help f32, where each row carries four times
        // the bytes and the block is not starved -- see gemv_at.
        let vec4 = cols % 4 == 0;
        let warp_per_row = vec4 && (cols / 4) < REDUCE_THREADS as usize;
        let (func, cfg) = if warp_per_row {
            let warps = (REDUCE_THREADS / 32) as usize;
            (
                &self.gemv_i8_warp,
                LaunchConfig {
                    grid_dim: (rows.div_ceil(warps) as u32, 1, 1),
                    block_dim: (REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                if vec4 { &self.gemv_i8_vec4 } else { &self.gemv_i8 },
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        };
        let (r, c, base, idx) = (rows as i32, cols as i32, y_base as i32, y_idx as i32);
        let acc = i32::from(accumulate);
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(scales).arg(x).arg(y).arg(&r).arg(&c).arg(params).arg(&base).arg(&idx).arg(&acc);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// y = W · x, one block per output row.
    pub fn gemv(
        &self,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // float4 loads need a multiple of four columns; every dimension in this
        // model qualifies, but fall back rather than silently corrupt if not.
        let func = if cols % 4 == 0 { &self.gemv_vec4 } else { &self.gemv };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows_i, cols_i, base, idx) =
            (rows as i32, cols as i32, 0i32, PARAM_ZERO as i32);
        let zeros = self.zero_params()?;
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(x).arg(y).arg(&rows_i).arg(&cols_i).arg(&zeros).arg(&base).arg(&idx);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// C[M,N] = A[M,K] * W[N,K]^T, the prefill counterpart to `gemv_at`.
    ///
    /// The weight layout is identical to the GEMV path, so nothing is repacked
    /// between prefill and decode.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        w: &Proj2,
        a: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) -> Result<()> {
        const TILE: u32 = 16;
        let (mi, ni, ki, acc) = (m as i32, n as i32, k as i32, i32::from(accumulate));

        if self.use_wmma {
            // Two tiles, chosen per launch.
            //
            // The 64x64 tile has better arithmetic intensity -- four times the
            // output per block for twice the loads -- but covers that output
            // with four times fewer blocks, so on a short prompt it starves the
            // GPU: at seq=128 with n_embd=768 it launches 24 blocks against the
            // 16-row tile's 96. Intensity only pays once the machine is full.
            //
            // Measured (int8 prefill, tok/s, 9 interleaved trials, 170.91 W
            // enforced / 3090 MHz; an independent 5-trial run at 148.86 W
            // reproduced every median to within 1%):
            //
            //   seq    16x64    64x64     auto
            //   128    18371    14971    18353
            //   256    26983    25408    28560
            //   512    35483    36821    39691
            //  1024    29453    34515    34900
            //
            // Neither fixed tile wins everywhere: 64x64 gives up ~19% at seq
            // 128 and takes ~17% at seq 1024.
            //
            // So the default picks per launch, on whether a launch would still
            // fill the GPU -- at least two blocks per SM, measured from the
            // device rather than hard-coded. That beats both fixed tiles
            // because the projections in one forward pass have very different N
            // (768, 2048, and 50304 for the lm_head) and want different tiles:
            // auto ties the small tile at seq 128 and wins by 5.8%, 11.9% and
            // 18.5% at 256/512/1024.
            //
            // The choice is numerically neutral. Both tiles walk K in the same
            // order with the same half conversion, so they produce bit-identical
            // logits; only speed changes. CRUCIBLE_GEMM=wmma-small / wmma-big
            // pin a fixed tile for benchmarking and debugging.
            let big_blocks = (m.div_ceil(64)) * (n.div_ceil(64));
            let use_big = match std::env::var("CRUCIBLE_GEMM").as_deref() {
                Ok("wmma-big") => true,
                Ok("wmma-small") => false,
                _ => big_blocks >= 2 * self.sm_count,
            };

            let block: u32 = if use_big { 64 } else { 16 };
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(64), (m as u32).div_ceil(block), 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            match w {
                Proj2::Int8(data, scales) => {
                    let f = if use_big { &self.gemm_i8_wmma_big } else { &self.gemm_i8_wmma };
                    let mut b = self.stream.launch_builder(f);
                    b.arg(*data).arg(*scales).arg(a).arg(c)
                        .arg(&mi).arg(&ni).arg(&ki).arg(&acc);
                    unsafe { cu(b.launch(cfg))? };
                }
                Proj2::F32(data) => {
                    let f = if use_big { &self.gemm_f32_wmma_big } else { &self.gemm_f32_wmma };
                    let mut b = self.stream.launch_builder(f);
                    b.arg(*data).arg(a).arg(c).arg(&mi).arg(&ni).arg(&ki).arg(&acc);
                    unsafe { cu(b.launch(cfg))? };
                }
            }
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(TILE),
                (m as u32).div_ceil(TILE),
                1,
            ),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        match w {
            Proj2::Int8(data, scales) => {
                let mut b = self.stream.launch_builder(&self.gemm_i8);
                b.arg(*data).arg(*scales).arg(a).arg(c)
                    .arg(&mi).arg(&ni).arg(&ki).arg(&acc);
                unsafe { cu(b.launch(cfg))? };
            }
            Proj2::F32(data) => {
                let mut b = self.stream.launch_builder(&self.gemm_f32);
                b.arg(*data).arg(a).arg(c).arg(&mi).arg(&ni).arg(&ki).arg(&acc);
                unsafe { cu(b.launch(cfg))? };
            }
        }
        Ok(())
    }

    /// RMSNorm over every row of a prompt: one launch instead of one per token.
    pub fn rmsnorm_batch(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, ni) = (rows as i32, n as i32);
        let mut b = self.stream.launch_builder(&self.rmsnorm_batch);
        b.arg(x).arg(weight).arg(out).arg(&r).arg(&ni).arg(&eps);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Rotary embedding across every position of a prompt.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_batch(
        &self,
        v: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        rows: usize,
        n_heads: usize,
        head_dim: usize,
        row_stride: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let total = (rows * n_heads * head_dim / 2) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        let (r, nh, hd, rs, po) = (
            rows as i32,
            n_heads as i32,
            head_dim as i32,
            row_stride as i32,
            pos_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.rope_batch);
        b.arg(v).arg(cos).arg(sin).arg(&r).arg(&nh).arg(&hd).arg(&rs).arg(&po);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn swiglu_batch(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.stream.launch_builder(&self.swiglu_batch);
        b.arg(gate).arg(up).arg(&ni);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn embed_batch(
        &self,
        table: &Proj2,
        tokens: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        d: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((rows * d) as u32);
        let (r, di) = (rows as i32, d as i32);
        match table {
            Proj2::F32(data) => {
                let mut b = self.stream.launch_builder(&self.embed_batch_f32);
                b.arg(*data).arg(tokens).arg(out).arg(&r).arg(&di);
                unsafe { cu(b.launch(cfg))? };
            }
            Proj2::Int8(data, scales) => {
                let mut b = self.stream.launch_builder(&self.embed_batch_i8);
                b.arg(*data).arg(*scales).arg(tokens).arg(out).arg(&r).arg(&di);
                unsafe { cu(b.launch(cfg))? };
            }
        }
        Ok(())
    }

    /// Place freshly computed K or V rows into a layer's cache region.
    pub fn cache_store(
        &self,
        src: &CudaSlice<f32>,
        cache: &mut CudaSlice<f32>,
        rows: usize,
        kv_dim: usize,
        layer_base: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((rows * kv_dim) as u32);
        let (r, kd, lb, po) = (
            rows as i32,
            kv_dim as i32,
            layer_base as i32,
            pos_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.cache_store);
        b.arg(src).arg(cache).arg(&r).arg(&kd).arg(&lb).arg(&po);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Scatter a dense `[rows, kv_dim]` block into the paged pool.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_store_paged(
        &self,
        src: &CudaSlice<f32>,
        pool: &mut CudaSlice<f32>,
        page_table: &CudaSlice<i32>,
        rows: usize,
        kv_dim: usize,
        n_layer: usize,
        layer: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((rows * kv_dim) as u32);
        let (r, kd, nl, l, po) = (
            rows as i32,
            kv_dim as i32,
            n_layer as i32,
            layer as i32,
            pos_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.cache_store_paged);
        b.arg(src).arg(pool).arg(page_table)
            .arg(&r).arg(&kd).arg(&nl).arg(&l).arg(&po);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Largest batch the batched-GEMV kernels are instantiated for.
    pub const GEMV_BATCH_MAX: usize = 16;

    /// Batched GEMV: `y[batch, rows] = x[batch, cols] @ w[rows, cols]^T`.
    ///
    /// int8 only. An f32 model still runs, through the GEMM, because f32
    /// weights are not the production decode path and adding a second
    /// instantiation family for them would be untested code.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_batch_i8(
        &self,
        w: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        batch: usize,
        accumulate: bool,
    ) -> Result<()> {
        if cols % 4 != 0 {
            anyhow::bail!("batched gemv needs cols divisible by 4, got {cols}");
        }
        if batch == 0 || batch > Self::GEMV_BATCH_MAX {
            anyhow::bail!("batched gemv supports 1..={} rows, got {batch}",
                          Self::GEMV_BATCH_MAX);
        }
        // Smallest instantiation that covers this batch.
        let idx = match batch {
            1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        };
        const THREADS: u32 = 256;
        let warps = THREADS / 32;
        let cfg = LaunchConfig {
            grid_dim: ((rows as u32).div_ceil(warps), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c, b, acc) = (
            rows as i32,
            cols as i32,
            batch as i32,
            i32::from(accumulate),
        );
        let mut bl = self.stream.launch_builder(&self.gemv_batch_i8[idx]);
        bl.arg(w).arg(scales).arg(x).arg(y).arg(&r).arg(&c).arg(&b).arg(&acc);
        unsafe { cu(bl.launch(cfg))? };
        Ok(())
    }

    /// RoPE with a per-row position, for a decode batch of mixed lengths.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_rows(
        &self,
        v: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        rows: usize,
        n_heads: usize,
        head_dim: usize,
        row_stride: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((rows * n_heads * head_dim / 2) as u32);
        let (r, nh, hd, rs) = (rows as i32, n_heads as i32, head_dim as i32, row_stride as i32);
        let mut b = self.stream.launch_builder(&self.rope_rows);
        b.arg(v).arg(cos).arg(sin).arg(&r).arg(&nh).arg(&hd).arg(&rs).arg(positions);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Scatter one KV row per request into each request's own page.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_store_rows_paged(
        &self,
        src: &CudaSlice<f32>,
        pool: &mut CudaSlice<f32>,
        page_tables: &CudaSlice<i32>,
        positions: &CudaSlice<i32>,
        rows: usize,
        kv_dim: usize,
        table_stride: usize,
        n_layer: usize,
        layer: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((rows * kv_dim) as u32);
        let (r, kd, ts, nl, l) = (
            rows as i32,
            kv_dim as i32,
            table_stride as i32,
            n_layer as i32,
            layer as i32,
        );
        let mut b = self.stream.launch_builder(&self.cache_store_rows_paged);
        b.arg(src).arg(pool).arg(page_tables).arg(positions)
            .arg(&r).arg(&kd).arg(&ts).arg(&nl).arg(&l);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Batched single-token decode attention over paged KV.
    ///
    /// `batch` blocks in grid.y, one page table and one sequence length per
    /// request. Batch 1 is the single-request case and takes the same path.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_paged(
        &self,
        q: &CudaSlice<f32>,
        k_pool: &CudaSlice<f32>,
        v_pool: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        page_tables: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>,
        batch: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        table_stride: usize,
        n_layer: usize,
        layer: usize,
        kv_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        // Shared memory is sized for the maximum context rather than the
        // current length, for the same reason as the contiguous kernel: a
        // captured graph fixes its allocation.
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, batch as u32, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: ((max_seq + (REDUCE_THREADS as usize / 32) * head_dim)
                * std::mem::size_of::<f32>()) as u32,
        };
        let (nh, nkv, hd, ts, nl, l, kd, ms) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            table_stride as i32,
            n_layer as i32,
            layer as i32,
            kv_dim as i32,
            max_seq as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention_decode_paged);
        b.arg(q).arg(k_pool).arg(v_pool).arg(out)
            .arg(page_tables).arg(seq_lens)
            .arg(&nh).arg(&nkv).arg(&hd).arg(&ts).arg(&nl).arg(&l).arg(&kd).arg(&ms);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Causal prefill attention over paged KV.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_paged(
        &self,
        q: &CudaSlice<f32>,
        k_pool: &CudaSlice<f32>,
        v_pool: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        page_table: &CudaSlice<i32>,
        rows: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_layer: usize,
        layer: usize,
        kv_dim: usize,
        max_seq: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, rows as u32, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: (max_seq * std::mem::size_of::<f32>()) as u32,
        };
        let (nh, nkv, hd, nl, l, kd, po) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            n_layer as i32,
            layer as i32,
            kv_dim as i32,
            pos_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention_prefill_paged);
        b.arg(q).arg(k_pool).arg(v_pool).arg(out).arg(page_table)
            .arg(&nh).arg(&nkv).arg(&hd).arg(&nl).arg(&l).arg(&kd).arg(&po);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Causal attention over an entire prompt: one block per (head, position).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        max_seq: usize,
        cache_stride: usize,
        layer_base: usize,
        pos_offset: usize,
    ) -> Result<()> {
        let k = k_cache.slice(layer_base..);
        let v = v_cache.slice(layer_base..);
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, rows as u32, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: (max_seq * std::mem::size_of::<f32>()) as u32,
        };
        let (nh, nkv, hd, cs, po) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            cache_stride as i32,
            pos_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention_prefill);
        b.arg(q).arg(&k).arg(&v).arg(out)
            .arg(&nh).arg(&nkv).arg(&hd).arg(&cs).arg(&po);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Attention over the KV cache, split across position chunks.
    ///
    /// Two kernels: each block reduces one chunk into a partial softmax, then a
    /// combine pass rescales and merges them. Exact, not approximate -- the
    /// rescaling by `exp(m_chunk - m_global)` is what makes splitting a softmax
    /// sound.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_split(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        partial_o: &mut CudaSlice<f32>,
        partial_m: &mut CudaSlice<f32>,
        partial_l: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        params: &CudaSlice<i32>,
        capacity: usize,
        cache_stride: usize,
        layer_base: usize,
    ) -> Result<()> {
        let n_chunks = attn_chunks(capacity);
        let k = k_cache.slice(layer_base..);
        let v = v_cache.slice(layer_base..);

        // scores[chunk] followed by one partial vector per warp.
        let shared = (ATTN_CHUNK + (REDUCE_THREADS as usize / 32) * head_dim)
            * std::mem::size_of::<f32>();
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, n_chunks as u32, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (nh, nkv, hd, si, cs, ch) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            PARAM_SEQ as i32,
            cache_stride as i32,
            ATTN_CHUNK as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention_partial);
        // Reborrow: the partials are written here and read by the combine
        // below, so the mutable references must survive both launches.
        b.arg(q).arg(&k).arg(&v).arg(&mut *partial_o).arg(&mut *partial_m).arg(&mut *partial_l)
            .arg(&nh).arg(&nkv).arg(&hd).arg(params).arg(&si).arg(&cs).arg(&ch);
        unsafe { cu(b.launch(cfg))? };

        let combine_cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc = n_chunks as i32;
        let mut b = self.stream.launch_builder(&self.attention_combine);
        b.arg(&*partial_o).arg(&*partial_m).arg(&*partial_l).arg(out)
            .arg(&nh).arg(&hd).arg(&nc);
        unsafe { cu(b.launch(combine_cfg))? };
        Ok(())
    }

    /// Fused SwiGLU: out = silu(gate . x) * (up . x).
    ///
    /// Replaces three kernels -- two projections and an elementwise product --
    /// with one, removing two dispatches per layer and two round trips through
    /// a `hidden`-sized buffer. CUDA graphs removed the CPU cost of launching,
    /// but each kernel still pays GPU-side dispatch, so kernel count continues
    /// to matter.
    pub fn mlp_swiglu(
        &self,
        gate: &Proj2,
        up: &Proj2,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let (r, c) = (rows as i32, cols as i32);
        match (gate, up) {
            (Proj2::Int8(gw, gs), Proj2::Int8(uw, us)) => {
                let warps = (REDUCE_THREADS / 32) as usize;
                let cfg = LaunchConfig {
                    grid_dim: (rows.div_ceil(warps) as u32, 1, 1),
                    block_dim: (REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(&self.mlp_swiglu_i8);
                b.arg(*gw).arg(*gs).arg(*uw).arg(*us).arg(x).arg(out).arg(&r).arg(&c);
                unsafe { cu(b.launch(cfg))? };
            }
            (Proj2::F32(gw), Proj2::F32(uw)) => {
                let cfg = LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(&self.mlp_swiglu_f32);
                b.arg(*gw).arg(*uw).arg(x).arg(out).arg(&r).arg(&c);
                unsafe { cu(b.launch(cfg))? };
            }
            _ => anyhow::bail!("gate and up must share a precision"),
        }
        Ok(())
    }

    pub fn rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.rmsnorm);
        b.arg(x).arg(weight).arg(out).arg(&n_i).arg(&eps);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn rope(
        &self,
        v: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
    ) -> Result<()> {
        let total = (n_heads * head_dim / 2) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        let mut host = vec![0i32; PARAM_COUNT];
        host[PARAM_POS] = pos as i32;
        let params = self.to_device_i32(&host)?;
        let (nh, hd, pi, base, idx) = (
            n_heads as i32,
            head_dim as i32,
            PARAM_POS as i32,
            0i32,
            PARAM_ZERO as i32,
        );
        let mut b = self.stream.launch_builder(&self.rope);
        b.arg(v).arg(cos).arg(sin).arg(&nh).arg(&hd).arg(&params).arg(&pi).arg(&base).arg(&idx);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn add_inplace(&self, dst: &mut CudaSlice<f32>, src: &CudaSlice<f32>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.add_inplace);
        b.arg(dst).arg(src).arg(&n_i);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn silu_mul(&self, gate: &mut CudaSlice<f32>, up: &CudaSlice<f32>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.silu_mul);
        b.arg(gate).arg(up).arg(&n_i);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    pub fn softmax(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.softmax);
        b.arg(x).arg(&n_i);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }
}

/// Power limit and clock ceiling currently enforced by the driver.
///
/// Recorded with every benchmark because this machine's power mode is
/// user-switchable between roughly 55 W and 175 W, and a number measured under
/// one cap says nothing about performance under another. An early set of GEMV
/// timings on this repo swung 168% purely because the GPU was in a 55 W eco
/// profile -- diagnosed at the time as a clock-governor fault, which it was
/// not. A measurement that does not record its power envelope is not
/// reproducible.
fn power_envelope() -> Option<(String, String)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=enforced.power.limit,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.trim().split(", ");
    Some((parts.next()?.to_string(), parts.next()?.to_string()))
}

fn print_envelope() {
    match power_envelope() {
        Some((watts, mhz)) => println!("power limit: {watts} W enforced, max SM clock {mhz} MHz"),
        None => println!("power limit: unknown (nvidia-smi unavailable)"),
    }
}

/// Largest relative difference between two vectors, ignoring near-zero values.
fn max_rel_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let scale = (x.abs().max(y.abs()) as f64).max(1e-6);
            ((*x as f64) - (*y as f64)).abs() / scale
        })
        .fold(0.0f64, f64::max)
}

/// Check every kernel against the CPU reference and report per-kernel error.
///
/// Exact equality is not the bar -- the GPU reduces in a different order and
/// `use_fast_math` trades a little accuracy for speed. What matters is that the
/// difference stays at rounding level rather than indicating a real defect.
pub fn validate() -> Result<()> {
    use crate::ops;

    let gpu = Gpu::new(0)?;
    println!("device: {}", gpu.name()?);
    println!("SMs: {}", gpu.sm_count);
    print_envelope();
    println!();

    let (rows, cols) = (2048usize, 768usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 97) as f32 - 48.0) / 64.0).collect();
    let x: Vec<f32> = (0..cols).map(|i| ((i % 31) as f32 - 15.0) / 16.0).collect();

    let mut cpu_y = vec![0.0f32; rows];
    ops::matvec(&w, rows, cols, &x, &mut cpu_y);

    let d_w = gpu.to_device(&w)?;
    let d_x = gpu.to_device(&x)?;
    let mut d_y = gpu.alloc(rows)?;
    gpu.gemv(&d_w, &d_x, &mut d_y, rows, cols)?;
    gpu.sync()?;
    let gpu_y = gpu.to_host(&d_y)?;
    println!("gemv        max rel diff {:.3e}", max_rel_diff(&cpu_y, &gpu_y));

    // rmsnorm
    let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i % 7) as f32 / 100.0).collect();
    let mut cpu_n = vec![0.0f32; cols];
    ops::rmsnorm(&x, &weight, 1e-6, &mut cpu_n);

    let d_weight = gpu.to_device(&weight)?;
    let mut d_n = gpu.alloc(cols)?;
    gpu.rmsnorm(&d_x, &d_weight, &mut d_n, cols, 1e-6)?;
    gpu.sync()?;
    println!("rmsnorm     max rel diff {:.3e}", max_rel_diff(&cpu_n, &gpu.to_host(&d_n)?));

    // softmax
    let scores: Vec<f32> = (0..1024).map(|i| ((i % 53) as f32 - 26.0) / 8.0).collect();
    let mut cpu_s = scores.clone();
    ops::softmax(&mut cpu_s);
    let mut d_s = gpu.to_device(&scores)?;
    gpu.softmax(&mut d_s, scores.len())?;
    gpu.sync()?;
    println!("softmax     max rel diff {:.3e}", max_rel_diff(&cpu_s, &gpu.to_host(&d_s)?));

    // silu_mul
    let gate: Vec<f32> = (0..2048).map(|i| ((i % 41) as f32 - 20.0) / 10.0).collect();
    let up: Vec<f32> = (0..2048).map(|i| ((i % 29) as f32 - 14.0) / 10.0).collect();
    let mut cpu_g = gate.clone();
    for (g, u) in cpu_g.iter_mut().zip(up.iter()) {
        *g = ops::silu(*g) * u;
    }
    let mut d_g = gpu.to_device(&gate)?;
    let d_u = gpu.to_device(&up)?;
    gpu.silu_mul(&mut d_g, &d_u, gate.len())?;
    gpu.sync()?;
    println!("silu_mul    max rel diff {:.3e}", max_rel_diff(&cpu_g, &gpu.to_host(&d_g)?));

    // rope
    let (n_heads, head_dim, pos) = (12usize, 64usize, 37usize);
    let table = ops::RopeTable::new(head_dim, 128, 10000.0);
    let vec_in: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0)
        .collect();
    let mut cpu_v = vec_in.clone();
    for h in 0..n_heads {
        table.apply(&mut cpu_v[h * head_dim..(h + 1) * head_dim], pos);
    }
    let d_cos = gpu.to_device(&table.cos)?;
    let d_sin = gpu.to_device(&table.sin)?;
    let mut d_v = gpu.to_device(&vec_in)?;
    gpu.rope(&mut d_v, &d_cos, &d_sin, n_heads, head_dim, pos)?;
    gpu.sync()?;
    println!("rope        max rel diff {:.3e}", max_rel_diff(&cpu_v, &gpu.to_host(&d_v)?));

    // argmax, against the exact host rule the scheduler used before.
    //
    // Tie-breaking and NaN handling are the whole point: a mismatch here shows
    // up as generated text that diverges on the first tie, which is far harder
    // to trace back than a wrong number.
    fn host_argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, x) in v.iter().enumerate() {
            if *x > v[best] {
                best = i;
            }
        }
        best
    }

    let nan = f32::NAN;
    let vocab = 50304usize;
    let mut dense: Vec<f32> = (0..vocab)
        .map(|i| ((i as f32) * 0.7391).sin() * 3.0 - 1.0)
        .collect();
    dense[31337] = 9.0; // unique maximum, deep in the row

    let mut tied = vec![-2.0f32; vocab];
    tied[100] = 5.0;
    tied[7000] = 5.0; // exact tie: the lower index must win

    let mut first = vec![-3.0f32; vocab];
    first[0] = 1.0;

    let mut last = vec![-3.0f32; vocab];
    last[vocab - 1] = 1.0;

    let mut close = vec![0.0f32; vocab];
    close[4242] = 1.000_000_1;
    close[4243] = 1.000_000_0;

    let all_neg: Vec<f32> = (0..vocab).map(|i| -1.0 - (i as f32) * 1e-4).collect();

    let mut nan_first = vec![1.0f32; vocab];
    nan_first[0] = nan; // host keeps index 0: nothing is > NaN

    let mut nan_mid = vec![1.0f32; vocab];
    nan_mid[500] = nan;
    nan_mid[900] = 4.0; // NaN must not displace the real maximum

    let all_nan = vec![nan; vocab];

    let cases: [(&str, &Vec<f32>); 8] = [
        ("dense", &dense),
        ("exact tie", &tied),
        ("max at 0", &first),
        ("max at last", &last),
        ("close top-2", &close),
        ("all negative", &all_neg),
        ("NaN at 0", &nan_first),
        ("NaN mid-row", &nan_mid),
    ];

    let rows = cases.len() + 1;
    let mut flat: Vec<f32> = Vec::with_capacity(rows * vocab);
    for (_, v) in cases.iter() {
        flat.extend_from_slice(v);
    }
    flat.extend_from_slice(&all_nan);

    let d_rows = gpu.to_device(&flat)?;
    let mut d_ids = gpu.to_device_i32(&vec![0i32; rows])?;
    gpu.argmax_rows(&d_rows, &mut d_ids, rows, vocab)?;
    gpu.sync()?;
    let got = gpu.to_host_i32_n(&d_ids, rows)?;

    println!();
    println!("{:<14} {:>10} {:>10} {:>8}", "argmax case", "host", "device", "match");
    let mut argmax_bad = 0usize;
    for (i, (name, v)) in cases.iter().enumerate() {
        let want = host_argmax(v);
        let ok = got[i] as usize == want;
        if !ok {
            argmax_bad += 1;
        }
        println!("{name:<14} {want:>10} {:>10} {:>8}", got[i], if ok { "ok" } else { "MISMATCH" });
    }
    {
        let want = host_argmax(&all_nan);
        let ok = got[rows - 1] as usize == want;
        if !ok {
            argmax_bad += 1;
        }
        println!("{:<14} {want:>10} {:>10} {:>8}", "all NaN", got[rows - 1],
                 if ok { "ok" } else { "MISMATCH" });
    }
    if argmax_bad > 0 {
        anyhow::bail!("device argmax disagreed with the host rule in {argmax_bad} case(s)");
    }
    println!();

    // gemm, both paths, against the same CPU reference.
    //
    // The tensor-core path converts activations to half, so it is held to a
    // looser tolerance than the scalar one -- but a looser tolerance is not no
    // tolerance. half carries 11 mantissa bits, so with K=768 accumulating in
    // f32 the error should land near 1e-3. Anything above 1e-2 means the tiling
    // or the fragment layout is wrong, not that half is imprecise.
    // Two properties this test data needs, both learned by getting them wrong.
    //
    // Mantissa-dense: an earlier version used (i % 89 - 44) / 128 -- small
    // integers over a power of two, which half represents exactly. Both paths
    // agreed bit-for-bit and the test reported 0.0 error, proving the tiling
    // was right and saying nothing about the precision question it exists to
    // answer.
    //
    // Single-sign: the fix after that used raw sin/cos, so a 768-term dot
    // product summed random signs and cancelled to near zero. Per-element
    // relative error against a near-zero result is meaningless -- it reported
    // ~2.0, which is what a sign flip on noise looks like, not a broken kernel.
    // Shifting both operands positive keeps the mantissas full while making the
    // sum accumulate monotonically, so relative error means what it says.
    let (gm, gn, gk) = (64usize, 512usize, 768usize);
    let gw: Vec<f32> = (0..gn * gk).map(|i| ((i as f32) * 0.7391).sin() * 0.4 + 0.6).collect();
    let ga: Vec<f32> = (0..gm * gk).map(|i| ((i as f32) * 1.2113).cos() * 0.3 + 0.5).collect();
    let mut cpu_c = vec![0.0f32; gm * gn];
    for row in 0..gm {
        for col in 0..gn {
            let mut acc = 0.0f32;
            for kk in 0..gk {
                acc += ga[row * gk + kk] * gw[col * gk + kk];
            }
            cpu_c[row * gn + col] = acc;
        }
    }

    let d_gw = gpu.to_device(&gw)?;
    let d_ga = gpu.to_device(&ga)?;
    for (label, wmma) in [("gemm f32", false), ("gemm f32 tc", true)] {
        let mut probe = Gpu::new(0)?;
        probe.use_wmma = wmma;
        let p_w = probe.to_device(&gw)?;
        let p_a = probe.to_device(&ga)?;
        let mut p_c = probe.alloc(gm * gn)?;
        probe.gemm(&Proj2::F32(&p_w), &p_a, &mut p_c, gm, gn, gk, false)?;
        probe.sync()?;
        println!("{label:11} max rel diff {:.3e}", max_rel_diff(&cpu_c, &probe.to_host(&p_c)?));
    }
    let _ = (&d_gw, &d_ga);

    // int8 path: quantise per output row exactly as quant.rs does, so the
    // reference includes quantisation error and the comparison isolates the
    // kernel.
    let mut qw = vec![0i8; gn * gk];
    let mut qs = vec![0.0f32; gn];
    for col in 0..gn {
        let row = &gw[col * gk..(col + 1) * gk];
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        qs[col] = scale;
        for (j, v) in row.iter().enumerate() {
            qw[col * gk + j] = (v / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let mut cpu_q = vec![0.0f32; gm * gn];
    for row in 0..gm {
        for col in 0..gn {
            let mut acc = 0.0f32;
            for kk in 0..gk {
                acc += ga[row * gk + kk] * qw[col * gk + kk] as f32;
            }
            cpu_q[row * gn + col] = acc * qs[col];
        }
    }
    for (label, wmma) in [("gemm i8", false), ("gemm i8 tc", true)] {
        let mut probe = Gpu::new(0)?;
        probe.use_wmma = wmma;
        let p_w = probe.to_device_i8(&qw)?;
        let p_s = probe.to_device(&qs)?;
        let p_a = probe.to_device(&ga)?;
        let mut p_c = probe.alloc(gm * gn)?;
        probe.gemm(&Proj2::Int8(&p_w, &p_s), &p_a, &mut p_c, gm, gn, gk, false)?;
        probe.sync()?;
        println!("{label:11} max rel diff {:.3e}", max_rel_diff(&cpu_q, &probe.to_host(&p_c)?));
    }

    Ok(())
}

/// Measure GEMV bandwidth, comparing kernel variants by paired trials.
///
/// GEMV reads the whole weight matrix and reuses nothing, so it is purely
/// bandwidth-bound; percentage of peak bandwidth is the meaningful score, not
/// TFLOP/s, which would flatter a memory-starved kernel.
///
/// The comparison is **paired**: both kernels run back to back inside each
/// trial, and the reported figure is the per-trial ratio between them. Running
/// all of kernel A then all of kernel B -- the obvious design, and this repo's
/// first attempt -- cannot separate a real difference from clock drift between
/// the two phases, and on this laptop clocks move enough to swamp any kernel
/// effect. Pairing cancels drift because both kernels see the same conditions
/// within a trial, so the ratio stays meaningful even when the absolute
/// throughput does not.
pub fn bench(rows: usize, cols: usize, iters: usize) -> Result<()> {
    const TRIALS: usize = 11;

    let gpu = Gpu::new(0)?;
    println!("device: {}", gpu.name()?);
    print_envelope();
    println!("gemv {rows} x {cols}, {iters} iters x {TRIALS} paired trials");
    println!();

    let w: Vec<f32> = (0..rows * cols).map(|i| (i % 101) as f32 / 100.0).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 31) as f32 / 30.0).collect();
    let d_w = gpu.to_device(&w)?;
    let d_x = gpu.to_device(&x)?;
    let mut d_y = gpu.alloc(rows)?;

    let bytes = (rows * cols * 4) as f64;

    // Long unbroken warm-up: clocks need sustained load to boost, and a short
    // warm-up leaves the first measured trials running at idle frequency --
    // which is what produced a 168 GB/s outlier against a 693 GB/s median.
    for _ in 0..(iters * 3) {
        gpu.gemv_scalar(&d_w, &d_x, &mut d_y, rows, cols)?;
    }
    gpu.sync()?;

    let mut timed = |vec4: bool, gpu: &Gpu, d_y: &mut CudaSlice<f32>| -> Result<f64> {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            if vec4 {
                gpu.gemv(&d_w, &d_x, d_y, rows, cols)?;
            } else {
                gpu.gemv_scalar(&d_w, &d_x, d_y, rows, cols)?;
            }
        }
        gpu.sync()?;
        Ok(bytes / (start.elapsed().as_secs_f64() / iters as f64) / 1e9)
    };

    let mut scalar = Vec::with_capacity(TRIALS);
    let mut vec4 = Vec::with_capacity(TRIALS);
    let mut ratios = Vec::with_capacity(TRIALS);

    for i in 0..TRIALS {
        // Alternate which kernel runs first, so neither systematically pays the
        // cost of following an idle gap.
        let (a, b) = if i % 2 == 0 {
            let a = timed(false, &gpu, &mut d_y)?;
            let b = timed(true, &gpu, &mut d_y)?;
            (a, b)
        } else {
            let b = timed(true, &gpu, &mut d_y)?;
            let a = timed(false, &gpu, &mut d_y)?;
            (a, b)
        };
        scalar.push(a);
        vec4.push(b);
        ratios.push(b / a);
    }

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let spread = |v: &Vec<f64>| (v[v.len() - 1] - v[0]) / v[v.len() / 2] * 100.0;

    let mut s_sorted = scalar.clone();
    let mut v_sorted = vec4.clone();
    let s_med = median(&mut s_sorted);
    let v_med = median(&mut v_sorted);
    let mut r_sorted = ratios.clone();
    let r_med = median(&mut r_sorted);

    println!("  scalar   {s_med:6.0} GB/s   [{:.0}-{:.0}]  spread {:4.1}%",
             s_sorted[0], s_sorted[TRIALS - 1], spread(&s_sorted));
    println!("  float4   {v_med:6.0} GB/s   [{:.0}-{:.0}]  spread {:4.1}%",
             v_sorted[0], v_sorted[TRIALS - 1], spread(&v_sorted));
    println!();

    let r_spread = (r_sorted[TRIALS - 1] - r_sorted[0]) / r_med * 100.0;
    println!("  paired ratio float4/scalar: {r_med:.3}   [{:.3}-{:.3}]  spread {r_spread:4.1}%",
             r_sorted[0], r_sorted[TRIALS - 1]);

    // Within a trial both kernels see the same clocks, so the ratio is the
    // trustworthy comparison even when absolute throughput drifts.
    let effect = (r_med - 1.0).abs() * 100.0;
    if effect < r_spread {
        println!("  -> {effect:.1}% difference against {r_spread:.1}% paired spread: not distinguishable");
    } else if r_med > 1.0 {
        println!("  -> float4 faster by {effect:.1}%, outside the paired spread");
    } else {
        println!("  -> scalar faster by {effect:.1}%, outside the paired spread");
    }

    let peak = s_sorted[TRIALS - 1].max(v_sorted[TRIALS - 1]);
    println!();
    println!("  best observed: {peak:.0} GB/s");
    Ok(())
}

impl Gpu {
    /// Scalar GEMV, for comparing against the float4 version.
    pub fn gemv_scalar(
        &self,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows_i, cols_i, base, idx) =
            (rows as i32, cols as i32, 0i32, PARAM_ZERO as i32);
        let zeros = self.zero_params()?;
        let mut b = self.stream.launch_builder(&self.gemv);
        b.arg(w).arg(x).arg(y).arg(&rows_i).arg(&cols_i).arg(&zeros).arg(&base).arg(&idx);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }
}
