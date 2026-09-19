//! Direct paged-attention tensor oracles and isolated timing. No model weights
//! or serving scheduler are involved; model/sequence parity lives in the packed
//! validation command. All allocations and metadata copies precede timing.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys::CUevent_flags, CudaSlice};
use llm_engine::gpu::{Gpu, PrefillAttentionVariant};
use std::collections::BTreeMap;
use std::time::Instant;

const HEADS: usize = 12;
const KV_HEADS: usize = 3;
const HD: usize = 64;
const D: usize = HEADS * HD;
const KD: usize = KV_HEADS * HD;
const CONTEXT: usize = 1024;
const STRIDE: usize = CONTEXT / 16;
const LAYERS: usize = 3;
const LAYER: usize = 1;
const ABS_TOL: f64 = 2e-5;
const REL_TOL: f64 = 2e-5;
const REL_FLOOR: f64 = 1e-6;
const SENTINEL: f32 = -12345.25;

#[derive(Clone, Debug)]
struct Slice {
    id: usize,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug)]
struct Case {
    name: String,
    slices: Vec<Slice>,
    seed: u64,
    profile: usize,
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn value(id: usize, pos: usize, head: usize, dim: usize, salt: u64) -> f32 {
    // Counter based, mantissa-dense values independent of descriptor ordering,
    // physical page assignment and the other requests present in a call.
    let mut v = salt
        ^ (id as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (pos as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (head as u64 + 1).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (dim as u64 + 1).wrapping_mul(0xd6e8_feb8_6659_fd93);
    v ^= v >> 30;
    v = v.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94d0_49bb_1331_11eb);
    v ^= v >> 31;
    ((v >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn q_value(case: &Case, slice: &Slice, pos: usize, h: usize, d: usize) -> f32 {
    match case.profile {
        1 => 0.0, // uniform distributions, including signed/cancelling V
        2 if h % 4 == 0 => 0.0,
        2 => value(slice.id, pos, h, d, 19) * [1.0, 0.125, 8.0, -16.0][h % 4],
        3 => {
            if d == 0 {
                [16.0, -16.0, 1.0, -1.0][h % 4]
            } else {
                0.0
            }
        }
        _ => value(slice.id, pos, h, d, 19) * [0.25, 1.0, -2.0, 4.0][h % 4],
    }
}

fn k_value(case: &Case, id: usize, pos: usize, h: usize, d: usize) -> f32 {
    if case.profile == 3 && d == 0 {
        // Repeated running-maximum changes across every page boundary.
        pos as f32 * 0.125 + h as f32 * 0.375
    } else {
        value(id, pos, h, d, 31)
    }
}

struct Fixture {
    case: Case,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    current_k: CudaSlice<f32>,
    current_v: CudaSlice<f32>,
    tables: CudaSlice<i32>,
    owners: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    segments: CudaSlice<i32>,
    out: CudaSlice<f32>,
    rows: usize,
    max_chunk: usize,
    score_capacity: usize,
}

impl Fixture {
    fn new(gpu: &Gpu, case: &Case, poison_future: bool) -> Result<Self> {
        ensure!(
            !case.slices.is_empty() && case.slices.len() <= 16,
            "invalid request count"
        );
        ensure!(
            case.slices
                .iter()
                .all(|s| s.len > 0 && s.start + s.len <= CONTEXT),
            "invalid slice"
        );
        let rows: usize = case.slices.iter().map(|s| s.len).sum();
        ensure!(
            rows <= CONTEXT,
            "packed rows exceed production scratch capacity"
        );
        // Include an extra physical page beyond the last query wherever legal.
        // Invalid future keys/values are finite poison, so accidental reads
        // cannot disappear behind NaN comparison behavior.
        let pages: Vec<_> = case
            .slices
            .iter()
            .map(|s| (s.start + s.len + 16).min(CONTEXT).div_ceil(16))
            .collect();
        let total: usize = pages.iter().sum();
        let mut mapping: Vec<_> = (0..total).collect();
        let mut state = case.seed.max(1);
        for i in (1..mapping.len()).rev() {
            let j = random(&mut state) as usize % (i + 1);
            mapping.swap(i, j);
        }
        let mut tables = vec![-1; case.slices.len() * STRIDE];
        // Round-robin assignment makes adjacent logical pages of a request
        // distinct from adjacent physical pages, even before permutation.
        let mut next = 0;
        for page in 0..*pages.iter().max().unwrap() {
            for (request, &count) in pages.iter().enumerate() {
                if page < count {
                    tables[request * STRIDE + page] = mapping[next] as i32;
                    next += 1;
                }
            }
        }
        let mut k = vec![99.75; total * LAYERS * 16 * KD];
        let mut v = vec![-87.125; k.len()];
        let mut q = Vec::with_capacity(rows * D);
        let mut current_k = Vec::with_capacity(rows * KD);
        let mut current_v = Vec::with_capacity(rows * KD);
        let mut owners = Vec::with_capacity(rows);
        let mut positions = Vec::with_capacity(rows);
        let mut segments = Vec::with_capacity(case.slices.len() * 4);
        let mut packed_start = 0;
        for (request, s) in case.slices.iter().enumerate() {
            segments.extend([
                packed_start as i32,
                s.len as i32,
                s.start as i32,
                request as i32,
            ]);
            for pos in 0..pages[request] * 16 {
                let physical = tables[request * STRIDE + pos / 16] as usize;
                let offset = ((physical * LAYERS + LAYER) * 16 + pos % 16) * KD;
                for h in 0..KV_HEADS {
                    for d in 0..HD {
                        let future = poison_future && pos >= s.start + s.len;
                        k[offset + h * HD + d] = if future {
                            123.75
                        } else {
                            k_value(case, s.id, pos, h, d)
                        };
                        v[offset + h * HD + d] = if future {
                            -987.25
                        } else {
                            value(s.id, pos, h, d, 47)
                        };
                    }
                }
            }
            for pos in s.start..s.start + s.len {
                owners.push(request as i32);
                positions.push(pos as i32);
                for h in 0..HEADS {
                    for d in 0..HD {
                        q.push(q_value(case, s, pos, h, d));
                    }
                }
                for h in 0..KV_HEADS {
                    for d in 0..HD {
                        current_k.push(k_value(case, s.id, pos, h, d));
                        current_v.push(value(s.id, pos, h, d, 47));
                    }
                }
            }
            packed_start += s.len;
        }
        Ok(Self {
            case: case.clone(),
            q: gpu.to_device(&q)?,
            k: gpu.to_device(&k)?,
            v: gpu.to_device(&v)?,
            current_k: gpu.to_device(&current_k)?,
            current_v: gpu.to_device(&current_v)?,
            tables: gpu.to_device_i32(&tables)?,
            owners: gpu.to_device_i32(&owners)?,
            positions: gpu.to_device_i32(&positions)?,
            segments: gpu.to_device_i32(&segments)?,
            out: gpu.to_device(&vec![SENTINEL; rows * D + 64])?,
            rows,
            max_chunk: case.slices.iter().map(|s| s.len).max().unwrap(),
            score_capacity: case
                .slices
                .iter()
                .map(|s| s.start + s.len)
                .max()
                .unwrap()
                .div_ceil(256)
                * 256,
        })
    }

    fn launch(&mut self, gpu: &Gpu, variant: PrefillAttentionVariant) -> Result<()> {
        if variant.is_reference() {
            gpu.attention_prefill_packed(
                &self.q,
                &self.k,
                &self.v,
                &mut self.out,
                &self.tables,
                &self.owners,
                &self.positions,
                self.rows,
                STRIDE,
                HEADS,
                KV_HEADS,
                HD,
                LAYERS,
                LAYER,
                KD,
                CONTEXT,
            )
        } else {
            gpu.attention_prefill_tiled(
                &self.q,
                &self.k,
                &self.v,
                &self.current_k,
                &self.current_v,
                &mut self.out,
                &self.tables,
                &self.segments,
                self.case.slices.len(),
                self.max_chunk,
                STRIDE,
                HEADS,
                KV_HEADS,
                HD,
                LAYERS,
                LAYER,
                KD,
                variant,
                self.score_capacity,
            )
        }
    }

    fn output(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        let mut values = gpu.to_host_n(&self.out, self.rows * D + 64)?;
        ensure!(
            values[self.rows * D..]
                .iter()
                .all(|&v| v.to_bits() == SENTINEL.to_bits()),
            "{}: output guard overwritten",
            self.case.name
        );
        values.truncate(self.rows * D);
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "{}: nonfinite output",
            self.case.name
        );
        Ok(values)
    }

    fn reset_output(&mut self, gpu: &Gpu) -> Result<()> {
        let sentinel = vec![SENTINEL; self.rows * D + 64];
        let mut active = self.out.slice_mut(0..sentinel.len());
        gpu.stream
            .memcpy_htod(&sentinel, &mut active)
            .map_err(|e| anyhow::anyhow!("output sentinel upload: {e:?}"))
    }

    /// Keep the original device addresses and unused metadata alive while
    /// replacing only the active prefix, just as a 16 -> 1 -> 8 -> 2 serving
    /// transition does. Fresh allocations alone would miss stale-tail reads.
    fn replace_active(&mut self, gpu: &Gpu, source: &Fixture) -> Result<()> {
        macro_rules! copy {
            ($field:ident) => {{
                ensure!(
                    source.$field.len() <= self.$field.len(),
                    "persistent fixture capacity exceeded"
                );
                let src = source.$field.slice(..);
                let mut dst = self.$field.slice_mut(0..source.$field.len());
                gpu.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow::anyhow!("persistent fixture copy: {e:?}"))?;
            }};
        }
        copy!(q);
        copy!(k);
        copy!(v);
        copy!(current_k);
        copy!(current_v);
        copy!(tables);
        copy!(owners);
        copy!(positions);
        copy!(segments);
        copy!(out);
        self.case = source.case.clone();
        self.rows = source.rows;
        self.max_chunk = source.max_chunk;
        self.score_capacity = source.score_capacity;
        Ok(())
    }
}

#[derive(Default)]
struct Error {
    abs: f64,
    rel: f64,
    elements: usize,
    half_differences: usize,
    half_abs: f64,
}

fn compare(
    got: &[f32],
    expected: &[f32],
    label: &str,
    errors: &mut Error,
    bit_exact: bool,
) -> Result<()> {
    ensure!(got.len() == expected.len(), "{label}: tensor shape differs");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        ensure!(
            g.is_finite() && e.is_finite(),
            "{label}: nonfinite element {i}"
        );
        let difference = (g as f64 - e as f64).abs();
        errors.abs = errors.abs.max(difference);
        errors.rel = errors
            .rel
            .max(difference / (g.abs().max(e.abs()) as f64).max(REL_FLOOR));
        errors.elements += 1;
        // Production O-projection WMMA loads convert these f32 attention
        // outputs to half. Tiny drift across a half rounding midpoint can
        // become a much larger GEMM input difference; report without relaxing
        // either the online tolerance or the Exact bitwise gate.
        let g_half = half::f16::from_f32(g);
        let e_half = half::f16::from_f32(e);
        errors.half_differences += usize::from(g_half.to_bits() != e_half.to_bits());
        errors.half_abs = errors
            .half_abs
            .max((g_half.to_f32() as f64 - e_half.to_f32() as f64).abs());
        if bit_exact {
            ensure!(
                g.to_bits() == e.to_bits(),
                "{label}: bit-exact element {i}, reference={e:.9e} (0x{:08x}), candidate={g:.9e} (0x{:08x}), abs={difference:.9e}",
                e.to_bits(), g.to_bits()
            );
        }
        ensure!(
            difference <= ABS_TOL + REL_TOL * (e as f64).abs(),
            "{label}: element {i}, reference={e:.9e}, candidate={g:.9e}, abs={difference:.9e}"
        );
    }
    Ok(())
}

fn variants() -> Vec<PrefillAttentionVariant> {
    let mut variants = Vec::new();
    for query_tile in [1, 2, 4] {
        for history_tile in [16, 32, 64] {
            variants.push(PrefillAttentionVariant::Tiled {
                query_tile,
                history_tile,
                hybrid: false,
                cache_page_table: false,
            });
        }
    }
    for (hybrid, cache_page_table) in [(true, false), (false, true), (true, true)] {
        variants.push(PrefillAttentionVariant::Tiled {
            query_tile: 2,
            history_tile: 32,
            hybrid,
            cache_page_table,
        });
    }
    for query_tile in [1, 2, 4] {
        for history_tile in [16, 32, 64] {
            for (hybrid, cache_page_table) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                variants.push(PrefillAttentionVariant::Exact {
                    query_tile,
                    history_tile,
                    hybrid,
                    cache_page_table,
                });
            }
        }
    }
    variants
}

fn homogeneous(name: &str, count: usize, len: usize, start: usize, seed: u64) -> Case {
    Case {
        name: name.into(),
        slices: (0..count).map(|id| Slice { id, start, len }).collect(),
        seed,
        profile: 2,
    }
}

fn cases(seed: u64) -> Vec<Case> {
    let mut cases = Vec::new();
    for p in [
        0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023,
    ] {
        cases.push(homogeneous(
            &format!("boundary-{p}"),
            1,
            1,
            p,
            seed + p as u64,
        ));
    }
    for start in [1, 15, 16, 17, 63, 64, 127, 128, 255, 512, 960] {
        cases.push(homogeneous(
            &format!("offset-{start}"),
            3,
            64,
            start,
            seed + start as u64,
        ));
    }
    for len in [1, 16, 32, 64, 128, 256, 512, 768, 941, 1024] {
        cases.push(homogeneous(
            &format!("single-{len}"),
            1,
            len,
            0,
            seed + len as u64,
        ));
    }
    for (count, len) in [
        (16, 16),
        (8, 32),
        (4, 64),
        (4, 128),
        (4, 256),
        (8, 128),
        (2, 512),
    ] {
        cases.push(homogeneous(
            &format!("{count}x{len}"),
            count,
            len,
            0,
            seed + count as u64,
        ));
    }
    for profile in 0..4 {
        cases.push(Case {
            name: format!("ragged-profile-{profile}"),
            slices: vec![
                Slice {
                    id: 0,
                    start: 0,
                    len: 17,
                },
                Slice {
                    id: 1,
                    start: 15,
                    len: 33,
                },
                Slice {
                    id: 2,
                    start: 255,
                    len: 129,
                },
                Slice {
                    id: 3,
                    start: 512,
                    len: 257,
                },
            ],
            seed: seed + profile as u64,
            profile,
        });
    }
    cases
}

/// F64 mathematical oracle for selected query rows. The preserved GPU kernel
/// remains the all-element numerical oracle for every case and every variant.
fn cpu_rows(case: &Case, output: &[f32]) -> Result<()> {
    let mut base = 0;
    for s in &case.slices {
        let mut selected = vec![0, s.len / 2, s.len - 1];
        selected.sort_unstable();
        selected.dedup();
        for local in selected {
            let pos = s.start + local;
            for h in 0..HEADS {
                let scores: Vec<f64> = (0..=pos)
                    .map(|p| {
                        (0..HD)
                            .map(|d| {
                                q_value(case, s, pos, h, d) as f64
                                    * k_value(case, s.id, p, h / 4, d) as f64
                            })
                            .sum::<f64>()
                            / 8.0
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for d in 0..HD {
                    let expected: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(p, w)| w * value(s.id, p, h / 4, d, 47) as f64)
                        .sum::<f64>()
                        / sum;
                    let got = output[(base + local) * D + h * HD + d] as f64;
                    ensure!((got - expected).abs() <= ABS_TOL + REL_TOL * expected.abs(),
                        "{}: CPU mathematical oracle row {local} head {h} dim {d}: expected {expected:.9e}, got {got:.9e}", case.name);
                }
            }
        }
        base += s.len;
    }
    Ok(())
}

pub fn check(fuzz: usize, seed: u64) -> Result<()> {
    let gpu = Gpu::new(0)?;
    println!(
        "device: {}; deterministic seed: {seed}; synthetic geometry 12/3/64, layer 1 of 3",
        gpu.name()?
    );
    println!("element gate: abs <= {ABS_TOL:e} + {REL_TOL:e}*abs(reference); reported relative denominator=max(abs(reference),abs(candidate),{REL_FLOOR:e})");
    println!("Exact variants additionally require identical f32 bits to Reference for every tensor element; online variant tolerances are unchanged.");
    let variants = variants();
    let mut cases = cases(seed);
    let fixed = cases.len();
    let mut state = seed.max(1);
    for round in 0..fuzz {
        let count = 1 + random(&mut state) as usize % 16;
        let mut remaining = CONTEXT;
        let mut slices = Vec::new();
        for id in 0..count {
            let len = 1 + random(&mut state) as usize % (remaining - (count - id - 1)).min(257);
            let start = random(&mut state) as usize % (CONTEXT - len + 1);
            slices.push(Slice {
                id: round * 16 + id,
                start,
                len,
            });
            remaining -= len;
        }
        cases.push(Case {
            name: format!("fuzz-{round}"),
            slices,
            seed: random(&mut state),
            profile: round % 4,
        });
    }
    let mut errors: Vec<_> = variants.iter().map(|_| Error::default()).collect();
    let mut comparisons = 0;
    let mut future_checks = 0;
    for (index, case) in cases.iter().enumerate() {
        let mut fixture = Fixture::new(&gpu, case, false)?;
        fixture.launch(&gpu, PrefillAttentionVariant::Reference)?;
        let reference = fixture.output(&gpu)?;
        if index < fixed {
            cpu_rows(case, &reference)?;
        }
        let mut poison = if case.name.starts_with("boundary-")
            && case.slices[0].start + case.slices[0].len < CONTEXT
        {
            Some(Fixture::new(&gpu, case, true)?)
        } else {
            None
        };
        for (v, variant) in variants.iter().copied().enumerate() {
            fixture.reset_output(&gpu)?;
            fixture.launch(&gpu, variant)?;
            let output = fixture.output(&gpu)?;
            compare(
                &output,
                &reference,
                &format!("{} {}", case.name, variant.name()),
                &mut errors[v],
                matches!(variant, PrefillAttentionVariant::Exact { .. }),
            )?;
            comparisons += 1;
            if let Some(poison) = poison.as_mut() {
                poison.reset_output(&gpu)?;
                poison.launch(&gpu, variant)?;
                let poisoned = poison.output(&gpu)?;
                ensure!(
                    output
                        .iter()
                        .zip(&poisoned)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} {}: future K/V affected causal output",
                    case.name,
                    variant.name()
                );
                future_checks += 1;
            }
        }
        if index % 16 == 0 || index + 1 == cases.len() {
            println!("checked {}/{} cases", index + 1, cases.len());
        }
    }
    // A's query tile boundaries are stable while packed starts, owner indices,
    // adjacent requests and every physical page change. Exact self-comparison
    // detects isolation bugs separately from the allowed reference drift.
    let mut invariant_checks = 0;
    for variant in &variants {
        let mut expected = BTreeMap::new();
        let capacity = homogeneous("persistent-capacity", 16, 17, 511, seed);
        let mut fixture = Fixture::new(&gpu, &capacity, false)?;
        for count in [16, 1, 8, 2, 4, 16] {
            for order in 0..3 {
                let mut case = homogeneous(
                    "composition-permutation-reuse",
                    count,
                    17,
                    511,
                    seed + count as u64 + order,
                );
                if order == 1 {
                    case.slices.reverse();
                }
                if order == 2 {
                    case.slices.rotate_left(count / 2);
                }
                let replacement = Fixture::new(&gpu, &case, false)?;
                fixture.replace_active(&gpu, &replacement)?;
                fixture.launch(&gpu, *variant)?;
                let output = fixture.output(&gpu)?;
                for (row, slice) in case.slices.iter().enumerate() {
                    let bits: Vec<_> = output[row * 17 * D..(row + 1) * 17 * D]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    if let Some(want) = expected.get(&slice.id) {
                        ensure!(
                            &bits == want,
                            "{}: descriptor composition/order/page-reuse changed request {}",
                            variant.name(),
                            slice.id
                        );
                        invariant_checks += 1;
                    } else {
                        expected.insert(slice.id, bits);
                    }
                }
            }
        }
    }
    for (variant, e) in variants.iter().zip(errors) {
        println!(
            "tensor_error,{},{},{:.9e},{:.9e}",
            variant.name(),
            e.elements,
            e.abs,
            e.rel
        );
        println!(
            "tensor_half_rounding_error,{},{},{:.9e}",
            variant.name(),
            e.half_differences,
            e.half_abs
        );
    }
    println!("PASS: {} fixed + {fuzz} fuzz cases; {comparisons} reference tensor comparisons; {future_checks} exact future-poison checks; {invariant_checks} exact composition/permutation/page-remap checks", fixed);
    println!("Model f32/int8 generation, RNG, cancellation and runtime page reuse: run gpu-packed-prefill-check separately for each attention variant.");
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[m - 1] + values[m]) * 0.5
    } else {
        values[m]
    }
}

pub fn bench(iters: usize, trials: usize, requested: &str, filter: &str, seed: u64) -> Result<()> {
    ensure!(
        iters > 0 && trials >= 3,
        "iters must be positive; at least three paired trials required"
    );
    let mut variants = vec![PrefillAttentionVariant::Reference];
    for s in requested
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let v: PrefillAttentionVariant = s
            .parse()
            .with_context(|| format!("invalid attention variant {s:?}"))?;
        if !variants.contains(&v) {
            variants.push(v);
        }
    }
    ensure!(variants.len() > 1, "benchmark requires a candidate variant");
    let gpu = Gpu::new(0)?;
    println!(
        "# device={}; seed={seed}; timing excludes allocations, H2D and D2H; synthetic f32 Q/K/V",
        gpu.name()?
    );
    for variant in &variants {
        let resources = gpu.attention_kernel_resources(*variant, CONTEXT, STRIDE)?;
        println!(
            "# resources,{},{}",
            variant.name(),
            serde_json::to_string(&resources)?
        );
    }
    println!("# global resources use worst-case score_capacity={CONTEXT}; per-case Exact resources use the launch capacity");
    println!("# event timing surrounds repeated eager launches and includes submission gaps; replay timing is a temporary graph of repeated attention kernels, not a serving graph");
    println!("case,requests,rows,starts,lengths,history_positions,variant,round,iters,eager_event_us,eager_wall_us,replay_event_us");
    let mut cases = cases(seed);
    cases.retain(|c| {
        !c.name.starts_with("boundary-")
            && !c.name.starts_with("offset-")
            && !c.name.starts_with("ragged-profile-")
    });
    for start in [0, 64, 256, 512, 960] {
        cases.push(homogeneous(
            &format!("history-64-at-{start}"),
            4,
            64,
            start,
            seed + start as u64,
        ));
    }
    // A 4x512 request set requires two 1024-row scheduled calls. Keep each
    // call capacity-valid and expose the second call's longer causal history.
    cases.push(homogeneous("scheduled-4x512-first", 4, 256, 0, seed));
    cases.push(homogeneous("scheduled-4x512-second", 4, 256, 256, seed));
    cases.push(Case {
        name: "heterogeneous".into(),
        slices: vec![
            Slice {
                id: 0,
                start: 0,
                len: 17,
            },
            Slice {
                id: 1,
                start: 63,
                len: 64,
            },
            Slice {
                id: 2,
                start: 255,
                len: 129,
            },
            Slice {
                id: 3,
                start: 512,
                len: 257,
            },
        ],
        seed,
        profile: 2,
    });
    if !filter.is_empty() {
        let names: Vec<_> = filter.split(',').map(str::trim).collect();
        cases.retain(|c| names.contains(&c.name.as_str()));
    }
    ensure!(!cases.is_empty(), "case filter selected no shapes");
    for case in cases {
        let mut fixture = Fixture::new(&gpu, &case, false)?;
        fixture.launch(&gpu, PrefillAttentionVariant::Reference)?;
        let reference = fixture.output(&gpu)?;
        let mut graphs = Vec::new();
        for variant in &variants {
            if matches!(variant, PrefillAttentionVariant::Exact { .. }) {
                let resources =
                    gpu.attention_kernel_resources(*variant, fixture.score_capacity, STRIDE)?;
                println!(
                    "# case_resources,{},{},score_capacity={},{}",
                    case.name,
                    variant.name(),
                    fixture.score_capacity,
                    serde_json::to_string(&resources)?
                );
            }
            fixture.reset_output(&gpu)?;
            for _ in 0..5 {
                fixture.launch(&gpu, *variant)?;
            }
            compare(
                &fixture.output(&gpu)?,
                &reference,
                &case.name,
                &mut Error::default(),
                matches!(variant, PrefillAttentionVariant::Exact { .. }),
            )?;
            gpu.sync()?;
            gpu.begin_capture()?;
            let queued = (|| {
                for _ in 0..iters {
                    fixture.launch(&gpu, *variant)?;
                }
                Ok::<_, anyhow::Error>(())
            })();
            let graph = gpu.end_capture();
            queued?;
            let graph = graph?;
            for _ in 0..3 {
                gpu.graph_launch(&graph)?;
            }
            gpu.sync()?;
            graphs.push(graph);
        }
        let start = gpu
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow::anyhow!("event creation: {e:?}"))?;
        let end = gpu
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow::anyhow!("event creation: {e:?}"))?;
        let starts = case
            .slices
            .iter()
            .map(|s| s.start.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let lengths = case
            .slices
            .iter()
            .map(|s| s.len.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let history: usize = case
            .slices
            .iter()
            .map(|s| s.len * (s.start + 1) + s.len * (s.len - 1) / 2)
            .sum();
        let mut summaries: Vec<Vec<f64>> = variants.iter().map(|_| Vec::new()).collect();
        for round in 0..trials {
            let order: Vec<_> = if round % 2 == 0 {
                (0..variants.len()).collect()
            } else {
                (0..variants.len()).rev().collect()
            };
            for index in order {
                let variant = variants[index];
                gpu.sync()?;
                start
                    .record(&gpu.stream)
                    .map_err(|e| anyhow::anyhow!("event record: {e:?}"))?;
                let wall = Instant::now();
                for _ in 0..iters {
                    fixture.launch(&gpu, variant)?;
                }
                end.record(&gpu.stream)
                    .map_err(|e| anyhow::anyhow!("event record: {e:?}"))?;
                gpu.sync()?;
                let wall_us = wall.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let eager_us = start
                    .elapsed_ms(&end)
                    .map_err(|e| anyhow::anyhow!("event elapsed: {e:?}"))?
                    as f64
                    * 1000.0
                    / iters as f64;
                start
                    .record(&gpu.stream)
                    .map_err(|e| anyhow::anyhow!("event record: {e:?}"))?;
                gpu.graph_launch(&graphs[index])?;
                end.record(&gpu.stream)
                    .map_err(|e| anyhow::anyhow!("event record: {e:?}"))?;
                gpu.sync()?;
                let replay_us = start
                    .elapsed_ms(&end)
                    .map_err(|e| anyhow::anyhow!("event elapsed: {e:?}"))?
                    as f64
                    * 1000.0
                    / iters as f64;
                summaries[index].push(replay_us);
                println!("{},{},{},{starts},{lengths},{history},{},{round},{iters},{eager_us:.6},{wall_us:.6},{replay_us:.6}",case.name,case.slices.len(),fixture.rows,variant.name());
            }
        }
        for (variant, mut samples) in variants.iter().zip(summaries) {
            let mid = median(&mut samples);
            println!(
                "# median,{},{},{mid:.6},min={:.6},max={:.6}",
                case.name,
                variant.name(),
                samples[0],
                samples[samples.len() - 1]
            );
        }
    }
    Ok(())
}
