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
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Kernel source, embedded so the binary is self-contained.
const KERNELS: &str = include_str!("../kernels/kernels.cu");

/// Blackwell. Compiling for an older architecture still runs, via PTX JIT, but
/// slower -- the exact failure this project checks for elsewhere.
const ARCH: &str = "compute_120";

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
    embed_i8: CudaFunction,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = cu(CudaContext::new(ordinal))?;
        let stream = ctx.default_stream();

        let opts = CompileOptions {
            arch: Some(ARCH),
            use_fast_math: Some(true),
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

    pub fn to_host(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>> {
        cu(self.stream.memcpy_dtov(dev))
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
        y_offset: usize,
    ) -> Result<()> {
        let func = if cols % 4 == 0 { &self.gemv_vec4 } else { &self.gemv };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c, o) = (rows as i32, cols as i32, y_offset as i32);
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(x).arg(y).arg(&r).arg(&c).arg(&o);
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
        pos: usize,
        v_offset: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((n_heads * head_dim / 2) as u32);
        let (nh, hd, p, o) = (
            n_heads as i32,
            head_dim as i32,
            pos as i32,
            v_offset as i32,
        );
        let mut b = self.stream.launch_builder(&self.rope);
        b.arg(v).arg(cos).arg(sin).arg(&nh).arg(&hd).arg(&p).arg(&o);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Copy one embedding row into the residual stream.
    pub fn embed(
        &self,
        table: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        token: usize,
        d: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(d as u32);
        let (t, dd) = (token as i32, d as i32);
        let mut b = self.stream.launch_builder(&self.embed);
        b.arg(table).arg(out).arg(&t).arg(&dd);
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
        seq_len: usize,
        cache_stride: usize,
        layer_base: usize,
    ) -> Result<()> {
        // The kernel indexes from the start of a layer's region, so the layer
        // offset is folded into the pointer by slicing rather than passed in.
        let k = k_cache.slice(layer_base..);
        let v = v_cache.slice(layer_base..);
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: (seq_len * std::mem::size_of::<f32>()) as u32,
        };
        let (nh, nkv, hd, sl, cs) = (
            n_head as i32,
            n_kv_head as i32,
            head_dim as i32,
            seq_len as i32,
            cache_stride as i32,
        );
        let mut b = self.stream.launch_builder(&self.attention);
        b.arg(q).arg(&k).arg(&v).arg(out)
            .arg(&nh).arg(&nkv).arg(&hd).arg(&sl).arg(&cs);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }

    /// Embedding lookup from an int8 table.
    pub fn embed_i8(
        &self,
        table: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        token: usize,
        d: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(d as u32);
        let (t, dd) = (token as i32, d as i32);
        let mut b = self.stream.launch_builder(&self.embed_i8);
        b.arg(table).arg(scales).arg(out).arg(&t).arg(&dd);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
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
        y_offset: usize,
    ) -> Result<()> {
        let func = if cols % 4 == 0 { &self.gemv_i8_vec4 } else { &self.gemv_i8 };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, c, o) = (rows as i32, cols as i32, y_offset as i32);
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(scales).arg(x).arg(y).arg(&r).arg(&c).arg(&o);
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
        let (rows_i, cols_i, zero) = (rows as i32, cols as i32, 0i32);
        let mut b = self.stream.launch_builder(func);
        b.arg(w).arg(x).arg(y).arg(&rows_i).arg(&cols_i).arg(&zero);
        unsafe { cu(b.launch(cfg))? };
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
        let (nh, hd, p, zero) = (n_heads as i32, head_dim as i32, pos as i32, 0i32);
        let mut b = self.stream.launch_builder(&self.rope);
        b.arg(v).arg(cos).arg(sin).arg(&nh).arg(&hd).arg(&p).arg(&zero);
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
        let (rows_i, cols_i, zero) = (rows as i32, cols as i32, 0i32);
        let mut b = self.stream.launch_builder(&self.gemv);
        b.arg(w).arg(x).arg(y).arg(&rows_i).arg(&cols_i).arg(&zero);
        unsafe { cu(b.launch(cfg))? };
        Ok(())
    }
}
