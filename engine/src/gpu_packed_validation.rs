//! Adversarial packed-prefill checks and paired microbenchmarks.
//!
//! This is a CLI validation module, deliberately outside the serving path.
//! Independent monolithic requests are the oracle. Request IDs, token values,
//! physical pages, chunk offsets and row order all vary independently.

use anyhow::{bail, ensure, Context, Result};
use llm_engine::gpu::{PrefillAttentionVariant, TOPK_MAX};
use llm_engine::gpu_model::{GpuModel, PackedPrefillRequest, Precision};
use llm_engine::paged::{SequencePages, PAGE_TOKENS};
use llm_engine::runtime::{FinishReason, Request, Runtime};
use llm_engine::sampling::{self, GenerationConfig, Rng};
use llm_engine::{Config, Weights};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone)]
struct Spec {
    id: u64,
    prompt: Vec<usize>,
    config: GenerationConfig,
}

fn spec(id: u64, len: usize, vocab: usize, steps: usize) -> Spec {
    let (temperature, top_k) =
        [(0.0, 40), (0.9, 5), (0.8, 40), (0.7, 128), (0.8, 500)][id as usize % 5];
    let mut state = id.wrapping_mul(977).wrapping_add(1234);
    let mut prompt: Vec<usize> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % vocab
        })
        .collect();
    // Same current token, radically different prefixes: a weak isolation test
    // with unrelated current tokens can conceal cross-request attention reads.
    prompt[len - 1] = 42 % vocab;
    Spec {
        id,
        prompt,
        config: GenerationConfig {
            max_tokens: steps,
            temperature,
            top_k,
            seed: id.wrapping_mul(1237) + 7,
        },
    }
}

fn load(cfg: &Config, weights: &Weights, precision: Precision) -> Result<GpuModel> {
    let mut model = GpuModel::load_with(cfg.clone(), weights, cfg.block_size, precision)?;
    model.enable_paging(cfg.block_size.div_ceil(PAGE_TOKENS) * 16, 16)?;
    model.set_prefill_graph(false);
    Ok(model)
}

fn allocate(model: &mut GpuModel, len: usize) -> Result<SequencePages> {
    let mut seq = SequencePages::new();
    seq.grow(model.page_pool_mut(), len)?;
    Ok(seq)
}

fn continuation(
    model: &mut GpuModel,
    spec: &Spec,
    table: &[i32],
    first: usize,
    rng: &mut Rng,
) -> Result<Vec<usize>> {
    let mut tokens = vec![first];
    let mut decode_tables = vec![0; model.max_batch() * model.table_stride()];
    decode_tables[..table.len()].copy_from_slice(table);
    let mut decode_lengths = vec![0; model.max_batch()];
    for generated in 1..spec.config.max_tokens {
        let position = spec.prompt.len() + generated - 1;
        decode_lengths[0] = (position + 1) as i32;
        let selection = model.decode_batch_mixed(
            &[*tokens.last().context("first token missing")?],
            &[position],
            &decode_tables,
            &decode_lengths[..1],
            &[],
            &[0],
        )?;
        tokens.push(sampling::sample(&selection.full, &spec.config, rng));
    }
    Ok(tokens)
}

fn independent(model: &mut GpuModel, spec: &Spec) -> Result<(Vec<usize>, Vec<f32>)> {
    let mut seq = allocate(model, spec.prompt.len() + spec.config.max_tokens)?;
    let table = seq.table_padded(model.table_stride());
    let logits = model.prefill_chunk(&spec.prompt, &table, 0, true)?;
    let mut rng = Rng::new(spec.config.seed);
    let first = sampling::sample(&logits, &spec.config, &mut rng);
    let output = continuation(model, spec, &table, first, &mut rng)?;
    seq.release(model.page_pool_mut())?;
    Ok((output, logits))
}

/// Execute varying packed compositions and then continue every request from
/// its own pages. Full rows here are diagnostic only; production byte counts
/// are checked separately using mixed per-row routing without diagnostic copies.
fn packed(
    model: &mut GpuModel,
    cfg: &Config,
    specs: &[Spec],
    chunk: usize,
    permutation: usize,
    prefixes: bool,
    references: &BTreeMap<u64, (Vec<usize>, Vec<f32>)>,
) -> Result<(usize, f32)> {
    let mut sequences = Vec::new();
    for spec in specs {
        sequences.push(allocate(model, spec.prompt.len() + spec.config.max_tokens)?);
    }
    let tables: Vec<Vec<i32>> = sequences
        .iter()
        .map(|s| s.table_padded(model.table_stride()))
        .collect();
    let mut consumed = vec![0; specs.len()];
    let mut first = vec![None; specs.len()];
    let mut rngs: Vec<Rng> = specs.iter().map(|s| Rng::new(s.config.seed)).collect();
    let mut max_delta = 0.0f32;
    if prefixes {
        for (index, spec) in specs.iter().enumerate() {
            let prefix =
                [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 127][index % 11].min(spec.prompt.len() - 1);
            if prefix > 0 {
                model.prefill_chunk(&spec.prompt[..prefix], &tables[index], 0, false)?;
                consumed[index] = prefix;
            }
        }
    }
    let mut calls = 0;
    while first.iter().any(Option::is_none) {
        let mut order: Vec<usize> = (0..specs.len())
            .filter(|&i| consumed[i] < specs[i].prompt.len())
            .collect();
        let count = order.len();
        if permutation == 1 {
            order.reverse();
        }
        if permutation == 2 && count > 0 {
            order.rotate_left(calls % count);
        }
        let mut remaining = cfg.block_size;
        let mut plan = Vec::new();
        for index in order {
            let len = chunk
                .min(specs[index].prompt.len() - consumed[index])
                .min(remaining);
            if len == 0 {
                break;
            }
            plan.push((index, consumed[index], len));
            remaining -= len;
        }
        ensure!(!plan.is_empty(), "packed plan made no progress");
        let descriptors: Vec<_> = plan
            .iter()
            .map(|&(index, start, len)| PackedPrefillRequest {
                tokens: &specs[index].prompt[start..start + len],
                page_table: &tables[index],
                pos_offset: start,
                want_logits: start + len == specs[index].prompt.len(),
            })
            .collect();
        let final_indices: Vec<usize> = plan
            .iter()
            .filter_map(|&(index, start, len)| {
                (start + len == specs[index].prompt.len()).then_some(index)
            })
            .collect();
        let topk_rows: Vec<_> = final_indices
            .iter()
            .enumerate()
            .filter_map(|(row, &index)| {
                let config = &specs[index].config;
                (config.temperature > 0.0 && config.top_k <= TOPK_MAX)
                    .then_some((row, config.top_k))
            })
            .collect();
        let full_rows: Vec<_> = final_indices
            .iter()
            .enumerate()
            .filter_map(|(row, &index)| {
                let config = &specs[index].config;
                (config.temperature > 0.0 && config.top_k > TOPK_MAX).then_some(row)
            })
            .collect();
        let selected = model.prefill_packed(&descriptors, &topk_rows, &full_rows)?;
        let expected_d2h = final_indices.len() * 4
            + if topk_rows.is_empty() {
                0
            } else {
                final_indices.len() * TOPK_MAX * 8
            }
            + full_rows.len() * cfg.vocab_size * 4;
        ensure!(
            selected.d2h_bytes == expected_d2h,
            "mixed packed D2H routing mismatch"
        );
        // A second identical execution retrieves diagnostic logits. Production
        // routing intentionally disallows overlapping top-k/full-row requests.
        let diagnostic = if final_indices.is_empty() {
            Vec::new()
        } else {
            let rows: Vec<_> = (0..final_indices.len()).collect();
            calls += 1;
            model.prefill_packed(&descriptors, &[], &rows)?.full
        };
        ensure!(
            selected.ids.len() == final_indices.len(),
            "wrong number of final rows"
        );
        if final_indices.is_empty() {
            ensure!(selected.d2h_bytes == 0, "non-final batch copied logits");
        }
        for (row, &index) in final_indices.iter().enumerate() {
            let spec = &specs[index];
            let logits = &diagnostic[row * cfg.vocab_size..(row + 1) * cfg.vocab_size];
            if let Some(full_index) = full_rows.iter().position(|&r| r == row) {
                ensure!(
                    &selected.full[full_index * cfg.vocab_size..(full_index + 1) * cfg.vocab_size]
                        == logits,
                    "full-logit fallback row mapping mismatch"
                );
            }
            let reference = &references[&spec.id].1;
            ensure!(
                logits.iter().all(|value| value.is_finite()),
                "nonfinite packed logits request {}",
                spec.id
            );
            for (value, expected) in logits.iter().zip(reference) {
                max_delta = max_delta.max((value - expected).abs());
            }
            let mut oracle_rng = rngs[index].clone();
            let oracle_token = sampling::sample(logits, &spec.config, &mut oracle_rng);
            let token = if spec.config.temperature <= 0.0 {
                selected.ids[row]
            } else if spec.config.top_k <= TOPK_MAX {
                let candidates: Vec<_> = (0..spec.config.top_k)
                    .map(|k| {
                        let offset = row * TOPK_MAX + k;
                        (
                            selected.cand_ids[offset] as usize,
                            selected.cand_vals[offset],
                        )
                    })
                    .collect();
                ensure!(
                    candidates == sampling::top_k(logits, spec.config.top_k),
                    "candidate mismatch request {}",
                    spec.id
                );
                sampling::sample_candidates(&candidates, &spec.config, &mut rngs[index])
            } else {
                sampling::sample(logits, &spec.config, &mut rngs[index])
            };
            ensure!(
                token == oracle_token && rngs[index].state() == oracle_rng.state(),
                "mixed sampling/RNG mismatch request {}",
                spec.id
            );
            first[index] = Some(token);
        }
        for (index, _, len) in plan {
            consumed[index] += len;
        }
        calls += 1;
    }
    for (index, spec) in specs.iter().enumerate() {
        let got = continuation(
            model,
            spec,
            &tables[index],
            first[index].context("missing final row")?,
            &mut rngs[index],
        )?;
        ensure!(got == references[&spec.id].0,
                "packed request {} differs (chunk {chunk}, permutation {permutation}, prefixes {prefixes})\nexpected {:?}\ngot {:?}",
                spec.id, references[&spec.id].0, got);
    }
    // Reverse release order changes the next batch's physical page mapping;
    // the model and persistent metadata buffers deliberately survive every run.
    for seq in sequences.iter_mut().rev() {
        seq.release(model.page_pool_mut())?;
    }
    ensure!(model.page_pool().used_pages() == 0, "packed pages leaked");
    Ok((calls, max_delta))
}

fn runtime_checks(
    model: GpuModel,
    cfg: &Config,
    weights: &Weights,
    precision: Precision,
    specs: &[Spec],
    concurrent_specs: &[Spec],
    references: &BTreeMap<u64, (Vec<usize>, Vec<f32>)>,
) -> Result<usize> {
    let attention = model.prefill_attention();
    let mut runtime = Runtime::new(model)?;
    runtime.set_batched_prefill(true);
    runtime.set_chunked_prefill(true);
    runtime.set_prefill_chunk(37);
    runtime.set_prefill_token_budget(131)?;
    runtime.set_max_prefill_requests(16)?;
    let mut comparisons = 0;
    // Independent continuations alone miss arithmetic that changes at a decode
    // batch-size crossover. Keep all sixteen requests decoding together, then
    // retire alternating short requests so swap_remove moves the eight long
    // survivors into new slots. Compare every token with the batch-one oracle.
    let original_batch_graph = runtime.model().batch_graph_enabled();
    runtime.set_chunked_prefill(false);
    runtime.set_prefill_token_budget(concurrent_specs.iter().map(|s| s.prompt.len()).sum())?;
    for graph in [false, true] {
        runtime.model_mut().set_batch_graph(graph);
        for reverse in [false, true] {
            let mut order: Vec<_> = concurrent_specs.iter().collect();
            if reverse {
                order.reverse();
            }
            for spec in order {
                runtime.submit(Request {
                    id: spec.id,
                    prompt: spec.prompt.clone(),
                    config: spec.config.clone(),
                })?;
            }
            let trace = runtime.run_to_completion(64)?;
            ensure!(runtime.is_idle(), "concurrent decode did not drain");
            let mut widths: Vec<_> = trace.iter().map(|s| s.decoded).filter(|&n| n > 0).collect();
            widths.dedup();
            ensure!(widths == [16, 8],
                "concurrent decode did not exercise 16 -> 8: graph={graph} reverse={reverse} widths={widths:?}");
            ensure!(
                trace.iter().filter(|s| s.prefill_batches > 0).count() == 1
                    && trace[0].prefill_final_rows == 16,
                "concurrent requests did not finish prefill together"
            );
            let completed = runtime.completed();
            ensure!(completed.len() == 16, "concurrent completion count changed");
            for expected in concurrent_specs {
                let matches: Vec<_> = completed.iter().filter(|c| c.id == expected.id).collect();
                ensure!(
                    matches.len() == 1,
                    "concurrent completion missing/duplicated {}",
                    expected.id
                );
                let completion = matches[0];
                ensure!(completion.reason == FinishReason::Length
                    && completion.tokens == references[&expected.id].0,
                    "concurrent decode differs from independent request {}: graph={graph} reverse={reverse} expected={:?} got={:?}",
                    expected.id, references[&expected.id].0, completion.tokens);
                comparisons += 1;
            }
            ensure!(
                runtime.free_pages() == runtime.model().page_pool().n_pages(),
                "concurrent decode leaked pages"
            );
        }
    }
    runtime.model_mut().set_batch_graph(original_batch_graph);
    runtime.set_chunked_prefill(true);
    runtime.set_prefill_token_budget(131)?;
    println!("  16 concurrent mixed greedy/top-k 5/40/128/500 requests, shrink 16 -> 8, eager/graph and reversed order: 64 exact sequences");
    let cancellation_start = comparisons;
    for cancel_at in [0, 1, 3, 6] {
        for spec in specs.iter().take(4) {
            runtime.submit(Request {
                id: spec.id,
                prompt: spec.prompt.clone(),
                config: spec.config.clone(),
            })?;
        }
        let victim = spec(
            9000 + cancel_at,
            511.min(cfg.block_size - 32),
            cfg.vocab_size,
            16,
        );
        runtime.submit(Request {
            id: victim.id,
            prompt: victim.prompt,
            config: victim.config,
        })?;
        for _ in 0..cancel_at {
            runtime.step()?;
        }
        ensure!(
            runtime.cancel(victim.id)?,
            "cancellation did not find request"
        );
        // Admit E immediately; it may inherit the victim's released pages.
        let replacement = &specs[4];
        runtime.submit(Request {
            id: replacement.id,
            prompt: replacement.prompt.clone(),
            config: replacement.config.clone(),
        })?;
        runtime.run_to_completion(4096)?;
        ensure!(runtime.is_idle(), "runtime failed to drain");
        let completed = runtime.completed();
        let expected_count = if cancel_at == 0 { 5 } else { 6 };
        ensure!(completed.len() == expected_count,
                "unexpected completion count after cancellation at {cancel_at}: expected {expected_count}, got {}", completed.len());
        ensure!(
            completed.iter().filter(|c| c.id == victim.id).count() == usize::from(cancel_at != 0),
            "pending/resident cancellation completion semantics changed"
        );
        for expected in specs.iter().take(5) {
            ensure!(
                completed.iter().filter(|c| c.id == expected.id).count() == 1,
                "missing/duplicate survivor {}",
                expected.id
            );
        }
        for completion in completed {
            if completion.id == victim.id {
                ensure!(
                    completion.reason == FinishReason::Cancelled,
                    "wrong cancellation reason"
                );
            } else {
                ensure!(
                    completion.tokens == references[&completion.id].0,
                    "survivor/page-reuse mismatch request {}",
                    completion.id
                );
                comparisons += 1;
            }
        }
        ensure!(
            runtime.free_pages() == runtime.model().page_pool().n_pages(),
            "runtime leaked pages"
        );
    }
    println!(
        "  cancellation at 0/1/3/6 boundaries and immediate page reuse: {} exact survivors",
        comparisons - cancellation_start
    );
    for batched in [false, true] {
        runtime.set_batched_prefill(batched);
        for spec in specs.iter().take(5) {
            let mut config = spec.config.clone();
            config.max_tokens = 1;
            runtime.submit(Request {
                id: spec.id,
                prompt: spec.prompt.clone(),
                config,
            })?;
        }
        runtime.run_to_completion(4096)?;
        let completed = runtime.completed();
        ensure!(
            completed.len() == 5,
            "max_tokens=1 completion count changed"
        );
        for completion in completed {
            ensure!(
                completion.tokens == references[&completion.id].0[..1],
                "max_tokens=1 produced wrong token count/value"
            );
            comparisons += 1;
        }
    }
    drop(runtime);
    // Physical capacity is fixed before Runtime constructs its reservation
    // ledger; reconfiguring the model underneath a live runtime is invalid.
    let mut pressure_model = GpuModel::load_with(cfg.clone(), weights, cfg.block_size, precision)?;
    pressure_model.set_prefill_attention(attention)?;
    pressure_model.enable_paging(32, 16)?;
    let mut runtime = Runtime::new(pressure_model)?;
    runtime.set_batched_prefill(true);
    runtime.set_chunked_prefill(true);
    runtime.set_prefill_chunk(37);
    runtime.set_prefill_token_budget(131)?;
    for spec in specs.iter().take(5) {
        runtime.submit(Request {
            id: spec.id,
            prompt: spec.prompt.clone(),
            config: spec.config.clone(),
        })?;
    }
    runtime.step()?;
    ensure!(
        runtime.pending_len() > 0,
        "constrained pool did not exercise admission backpressure"
    );
    runtime.run_to_completion(4096)?;
    let completed = runtime.completed();
    ensure!(
        completed.len() == 5,
        "page-pressure completion count changed"
    );
    for completion in completed {
        ensure!(
            completion.tokens == references[&completion.id].0,
            "page-pressure request output changed"
        );
        comparisons += 1;
    }
    ensure!(runtime.free_pages() == 32, "constrained pool leaked pages");
    println!(
        "  max_tokens=1 packed/reference and constrained 32-page admission/decode growth: exact"
    );
    Ok(comparisons)
}

/// Reject malformed descriptors before GPU writes, then immediately exercise
/// both selection paths and an unrelated resident sequence. Checking only a
/// freshly overwritten prompt would conceal damage to a neighbor's cached KV.
fn malformed_metadata_check(model: &mut GpuModel, cfg: &Config) -> Result<usize> {
    let anchor = spec(77_001, 33, cfg.vocab_size, 2);
    let work = spec(77_002, 17, cfg.vocab_size, 1);
    let mut anchor_pages = allocate(model, 34)?;
    let mut work_pages = allocate(model, 17)?;
    let anchor_table = anchor_pages.table_padded(model.table_stride());
    let work_table = work_pages.table_padded(model.table_stride());
    let anchor_logits = model.prefill_chunk(&anchor.prompt, &anchor_table, 0, true)?;
    let anchor_token = sampling::argmax(&anchor_logits);
    let expected = model.prefill_chunk(&work.prompt, &work_table, 0, true)?;
    let mut decode_tables = vec![0; model.max_batch() * model.table_stride()];
    decode_tables[..model.table_stride()].copy_from_slice(&anchor_table);
    let anchor_next = model
        .decode_batch_mixed(&[anchor_token], &[33], &decode_tables, &[34], &[], &[0])?
        .full;
    let good = PackedPrefillRequest {
        tokens: &work.prompt,
        page_table: &work_table,
        pos_offset: 0,
        want_logits: true,
    };
    let same_bits = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    let recover = |model: &mut GpuModel, label: &str| -> Result<()> {
        let after = model
            .decode_batch_mixed(&[anchor_token], &[33], &decode_tables, &[34], &[], &[0])?
            .full;
        ensure!(
            same_bits(&after, &anchor_next),
            "{label}: malformed input changed unrelated resident KV"
        );
        let packed = model.prefill_packed(std::slice::from_ref(&good), &[], &[0])?;
        ensure!(
            same_bits(&packed.full, &expected),
            "{label}: packed recovery logits changed"
        );
        let single = model.prefill_single_mixed(&good, &[], &[0])?;
        ensure!(
            same_bits(&single.full, &expected),
            "{label}: singleton recovery logits changed"
        );
        let compact = model.prefill_single_mixed(&good, &[], &[])?;
        ensure!(
            compact.ids == vec![sampling::argmax(&expected)] && compact.d2h_bytes == 4,
            "{label}: singleton recovery lost greedy compact selection"
        );
        Ok(())
    };
    let old_graph = model.prefill_graph();
    model.set_prefill_graph(true);
    let mut rejected = 0usize;
    macro_rules! reject {
        ($label:literal, $call:expr) => {{
            ensure!(
                $call.is_err(),
                concat!($label, ": malformed metadata accepted")
            );
            recover(model, $label)?;
            rejected += 1;
        }};
    }
    reject!("paging batch capacity", model.enable_paging(1, 17));
    reject!("empty batch", model.prefill_packed(&[], &[], &[]));
    let empty = PackedPrefillRequest {
        tokens: &[],
        ..good
    };
    reject!("empty chunk", model.prefill_single_mixed(&empty, &[], &[]));
    let overflow = PackedPrefillRequest {
        pos_offset: usize::MAX,
        ..good
    };
    reject!(
        "position overflow",
        model.prefill_single_mixed(&overflow, &[], &[])
    );
    let past_end = PackedPrefillRequest {
        pos_offset: cfg.block_size,
        ..good
    };
    reject!(
        "context overflow",
        model.prefill_single_mixed(&past_end, &[], &[])
    );
    let huge_tokens = vec![0; model.prefill_token_capacity() + 1];
    let huge = PackedPrefillRequest {
        tokens: &huge_tokens,
        ..good
    };
    reject!(
        "packed row overflow",
        model.prefill_packed(&[huge], &[], &[])
    );
    let short = PackedPrefillRequest {
        page_table: &work_table[..1],
        ..good
    };
    reject!(
        "short page table",
        model.prefill_single_mixed(&short, &[], &[])
    );
    let mut negative_table = work_table.clone();
    negative_table[0] = -1;
    let negative = PackedPrefillRequest {
        page_table: &negative_table,
        ..good
    };
    reject!(
        "negative physical page",
        model.prefill_single_mixed(&negative, &[], &[])
    );
    let mut outside_table = work_table.clone();
    outside_table[0] = model.page_pool().n_pages() as i32;
    let outside = PackedPrefillRequest {
        page_table: &outside_table,
        ..good
    };
    reject!(
        "physical page outside pool",
        model.prefill_single_mixed(&outside, &[], &[])
    );
    let mut alias_table = work_table.clone();
    alias_table[1] = alias_table[0];
    let alias = PackedPrefillRequest {
        page_table: &alias_table,
        ..good
    };
    reject!(
        "logical page alias",
        model.prefill_single_mixed(&alias, &[], &[])
    );
    let aliases = [
        PackedPrefillRequest { ..good },
        PackedPrefillRequest { ..good },
    ];
    reject!(
        "cross-request page alias",
        model.prefill_packed(&aliases, &[], &[])
    );
    let mut bad_tokens = work.prompt.clone();
    bad_tokens[0] = cfg.vocab_size;
    let bad_token = PackedPrefillRequest {
        tokens: &bad_tokens,
        ..good
    };
    reject!(
        "token outside vocabulary",
        model.prefill_single_mixed(&bad_token, &[], &[])
    );
    let too_many: Vec<_> = (0..=model.prefill_request_capacity())
        .map(|_| PackedPrefillRequest { ..good })
        .collect();
    reject!(
        "request capacity",
        model.prefill_packed(&too_many, &[], &[])
    );
    reject!(
        "zero top-k",
        model.prefill_single_mixed(&good, &[(0, 0)], &[])
    );
    reject!(
        "oversize device top-k",
        model.prefill_single_mixed(&good, &[(0, TOPK_MAX + 1)], &[])
    );
    reject!(
        "top-k row outside finals",
        model.prefill_single_mixed(&good, &[(1, 5)], &[])
    );
    reject!(
        "full row outside finals",
        model.prefill_single_mixed(&good, &[], &[1])
    );
    reject!(
        "duplicate full row",
        model.prefill_single_mixed(&good, &[], &[0, 0])
    );
    reject!(
        "overlapping selection routes",
        model.prefill_single_mixed(&good, &[(0, 5)], &[0])
    );
    let non_final = PackedPrefillRequest {
        want_logits: false,
        ..good
    };
    reject!(
        "selection without final row",
        model.prefill_single_mixed(&non_final, &[(0, 5)], &[])
    );
    model.prefill_packed(std::slice::from_ref(&good), &[], &[])?;
    ensure!(
        model.time_packed_replay(18, 1, 1).is_err(),
        "stale replay shape accepted"
    );
    model.prefill_single_mixed(&good, &[], &[])?;
    ensure!(
        model.time_packed_replay(17, 1, 1).is_err(),
        "singleton left packed replay metadata live"
    );
    model.set_prefill_graph(old_graph);
    anchor_pages.release(model.page_pool_mut())?;
    work_pages.release(model.page_pool_mut())?;
    ensure!(
        model.page_pool().used_pages() == 0,
        "malformed-input checks leaked pages"
    );
    println!("  {rejected} malformed descriptor/selection cases rejected; neighbor KV and packed/singleton recovery bit-exact");
    Ok(rejected)
}

/// Replay one boundary-corpus request with a fixed oracle token history. This
/// deliberately has no tolerance gate: it explains a failed strict check and
/// leaves the normal generated-sequence/RNG checks unchanged.
fn diagnose_request(
    cfg: &Config,
    reference: &mut GpuModel,
    candidate: &mut GpuModel,
    steps: usize,
    id: u64,
) -> Result<()> {
    ensure!(
        (100..=118).contains(&id),
        "diagnose-request supports boundary corpus IDs 100..=118"
    );
    ensure!(
        steps <= 32,
        "diagnose-request is bounded to at most 32 generated tokens"
    );
    let lengths = [
        1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 941,
    ];
    let corpus: Vec<_> = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| spec(100 + i as u64, len, cfg.vocab_size, steps))
        .collect();
    let specs = if id <= 115 {
        &corpus[..16]
    } else {
        &corpus[3..]
    };
    let target = specs
        .iter()
        .position(|s| s.id == id)
        .context("diagnostic target missing")?;
    let spec = &specs[target];
    let attention = candidate.prefill_attention();
    println!("diagnose_request,id={id},prompt_len={},temperature={},top_k={},seed={},steps={steps},chunk=32,permutation=0,prefixes=false,requests={}",
        spec.prompt.len(), spec.config.temperature, spec.config.top_k, spec.config.seed, specs.len());
    println!("diagnose_history: every candidate decode consumes the monolithic Reference token at the preceding step; RNG starts from the same request seed in each pass");

    let mut sequence = allocate(reference, spec.prompt.len() + steps)?;
    let table = sequence.table_padded(reference.table_stride());
    let mut logits = reference.prefill_chunk(&spec.prompt, &table, 0, true)?;
    let mut oracle_logits = Vec::with_capacity(steps);
    let mut oracle_tokens = Vec::with_capacity(steps);
    let mut rng = Rng::new(spec.config.seed);
    for generated in 0..steps {
        let token = sampling::sample(&logits, &spec.config, &mut rng);
        oracle_tokens.push(token);
        oracle_logits.push(logits);
        if generated + 1 < steps {
            logits = diagnostic_decode(reference, &table, spec.prompt.len() + generated, token)?;
        } else {
            break;
        }
    }
    sequence.release(reference.page_pool_mut())?;
    println!("diagnose_oracle_tokens,{oracle_tokens:?}");

    // The Reference packed control separates effects of packed/chunked GEMM
    // execution from differences introduced by the attention candidate.
    for (label, variant) in [
        ("packed_reference", PrefillAttentionVariant::Reference),
        ("packed_candidate", attention),
    ] {
        candidate.set_prefill_attention(variant)?;
        let (mut sequences, tables, mut logits) =
            diagnostic_prefill(candidate, cfg, specs, target)?;
        let mut rng = Rng::new(spec.config.seed);
        let mut oracle_rng = Rng::new(spec.config.seed);
        let mut first_difference = None;
        for generated in 0..steps {
            let before = rng.state();
            let mut draw_rng = rng.clone();
            let draw = if spec.config.temperature > 0.0 {
                Some(draw_rng.next_f32())
            } else {
                None
            };
            let got = sampling::sample(&logits, &spec.config, &mut rng);
            let expected =
                sampling::sample(&oracle_logits[generated], &spec.config, &mut oracle_rng);
            ensure!(
                expected == oracle_tokens[generated],
                "diagnostic oracle replay changed"
            );
            ensure!(
                rng.state() == oracle_rng.state(),
                "diagnostic RNG advancement differs at token {}",
                generated + 1
            );
            if got != expected && first_difference.is_none() {
                first_difference = Some(generated + 1);
            }
            let expected_logits = &oracle_logits[generated];
            ensure!(
                logits.len() == expected_logits.len()
                    && logits.iter().all(|x| x.is_finite())
                    && expected_logits.iter().all(|x| x.is_finite()),
                "invalid diagnostic logits"
            );
            let mut max_abs = 0.0f64;
            let mut squared = 0.0f64;
            let mut bit_differences = 0usize;
            let mut worst = 0usize;
            for (index, (&a, &b)) in expected_logits.iter().zip(&logits).enumerate() {
                let difference = (a as f64 - b as f64).abs();
                if difference > max_abs {
                    max_abs = difference;
                    worst = index;
                }
                squared += difference * difference;
                bit_differences += usize::from(a.to_bits() != b.to_bits());
            }
            let rms = (squared / logits.len() as f64).sqrt();
            let draw_text = draw
                .map(|d| format!("{d:.9}"))
                .unwrap_or_else(|| "none".into());
            println!("diagnose_step,{label},variant={},token={},input={},expected={expected},got={got},equal={},max_abs={max_abs:.9e},rms={rms:.9e},different_bits={bit_differences},worst_id={worst},worst_reference={:.9e},worst_candidate={:.9e},rng_before={before},rng_after={},draw={draw_text}",
                variant.name(), generated + 1, if generated == 0 { spec.prompt[spec.prompt.len() - 1] } else { oracle_tokens[generated - 1] },
                expected == got, expected_logits[worst], logits[worst], rng.state());
            if generated == 0 || got != expected {
                diagnostic_sampling(
                    "reference",
                    expected_logits,
                    &spec.config,
                    expected,
                    got,
                    draw,
                );
                diagnostic_sampling(label, &logits, &spec.config, got, expected, draw);
                let expected_ids: std::collections::BTreeSet<_> =
                    sampling::top_k(expected_logits, spec.config.top_k)
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();
                let overlap = sampling::top_k(&logits, spec.config.top_k)
                    .iter()
                    .filter(|(id, _)| expected_ids.contains(id))
                    .count();
                println!(
                    "diagnose_candidates,{label},token={},top_k_overlap={overlap}/{}",
                    generated + 1,
                    expected_ids.len()
                );
            }
            if generated + 1 < steps {
                logits = diagnostic_decode(
                    candidate,
                    &tables[target],
                    spec.prompt.len() + generated,
                    expected,
                )?;
            }
        }
        for seq in sequences.iter_mut().rev() {
            seq.release(candidate.page_pool_mut())?;
        }
        ensure!(
            candidate.page_pool().used_pages() == 0,
            "diagnostic pages leaked"
        );
        println!(
            "diagnose_first_different_choice,{label},{}",
            first_difference
                .map(|s| s.to_string())
                .unwrap_or_else(|| "none".into())
        );
    }
    candidate.set_prefill_attention(attention)?;
    println!("Diagnostic complete; run the unmodified strict check without --diagnose-request for the acceptance result.");
    Ok(())
}

fn diagnostic_decode(
    model: &mut GpuModel,
    table: &[i32],
    position: usize,
    token: usize,
) -> Result<Vec<f32>> {
    let mut tables = vec![0; model.max_batch() * model.table_stride()];
    tables[..table.len()].copy_from_slice(table);
    Ok(model
        .decode_batch_mixed(
            &[token],
            &[position],
            &tables,
            &[(position + 1) as i32],
            &[],
            &[0],
        )?
        .full)
}

fn diagnostic_prefill(
    model: &mut GpuModel,
    cfg: &Config,
    specs: &[Spec],
    target: usize,
) -> Result<(Vec<SequencePages>, Vec<Vec<i32>>, Vec<f32>)> {
    let mut sequences = Vec::new();
    for spec in specs {
        sequences.push(allocate(model, spec.prompt.len() + spec.config.max_tokens)?);
    }
    let tables: Vec<_> = sequences
        .iter()
        .map(|s| s.table_padded(model.table_stride()))
        .collect();
    let mut consumed = vec![0; specs.len()];
    let mut target_logits = None;
    while consumed.iter().zip(specs).any(|(&n, s)| n < s.prompt.len()) {
        let mut remaining = cfg.block_size;
        let mut plan = Vec::new();
        for (index, spec) in specs.iter().enumerate() {
            if consumed[index] == spec.prompt.len() {
                continue;
            }
            let len = 32.min(spec.prompt.len() - consumed[index]).min(remaining);
            if len == 0 {
                break;
            }
            plan.push((index, consumed[index], len));
            remaining -= len;
        }
        ensure!(!plan.is_empty(), "diagnostic packed plan made no progress");
        let descriptors: Vec<_> = plan
            .iter()
            .map(|&(index, start, len)| PackedPrefillRequest {
                tokens: &specs[index].prompt[start..start + len],
                page_table: &tables[index],
                pos_offset: start,
                want_logits: start + len == specs[index].prompt.len(),
            })
            .collect();
        let finals: Vec<_> = plan
            .iter()
            .filter_map(|&(index, start, len)| {
                (start + len == specs[index].prompt.len()).then_some(index)
            })
            .collect();
        let topk_rows: Vec<_> = finals
            .iter()
            .enumerate()
            .filter_map(|(row, &index)| {
                let c = &specs[index].config;
                (c.temperature > 0.0 && c.top_k <= TOPK_MAX).then_some((row, c.top_k))
            })
            .collect();
        let full_rows: Vec<_> = finals
            .iter()
            .enumerate()
            .filter_map(|(row, &index)| {
                let c = &specs[index].config;
                (c.temperature > 0.0 && c.top_k > TOPK_MAX).then_some(row)
            })
            .collect();
        model.prefill_packed(&descriptors, &topk_rows, &full_rows)?;
        if !finals.is_empty() {
            let rows: Vec<_> = (0..finals.len()).collect();
            let diagnostic = model.prefill_packed(&descriptors, &[], &rows)?.full;
            if let Some(row) = finals.iter().position(|&index| index == target) {
                target_logits =
                    Some(diagnostic[row * cfg.vocab_size..(row + 1) * cfg.vocab_size].to_vec());
            }
        }
        for (index, _, len) in plan {
            consumed[index] += len;
        }
    }
    Ok((
        sequences,
        tables,
        target_logits.context("diagnostic final row missing")?,
    ))
}

fn diagnostic_sampling(
    label: &str,
    logits: &[f32],
    config: &GenerationConfig,
    chosen: usize,
    other: usize,
    draw: Option<f32>,
) {
    let k = config.top_k.clamp(1, logits.len());
    let ranked = sampling::top_k(logits, (k + 1).min(logits.len()));
    let candidates = &ranked[..k];
    let best = candidates[0].1;
    let gap = if k < ranked.len() {
        candidates[k - 1].1 - ranked[k].1
    } else {
        f32::NAN
    };
    let top_gap = if ranked.len() > 1 {
        best - ranked[1].1
    } else {
        f32::NAN
    };
    println!("diagnose_distribution,{label},top1_gap={top_gap:.9e},topk_cutoff_gap={gap:.9e},selected_logit={:.9e},other_logit={:.9e},selected_minus_other={:.9e}",
        logits[chosen], logits[other], logits[chosen] - logits[other]);
    if config.temperature <= 0.0 {
        for (rank, &(id, value)) in candidates.iter().enumerate() {
            if rank < 8 || id == chosen || id == other {
                println!("diagnose_candidate,{label},rank={},id={id},logit={value:.9e},chosen={},other={}",
                    rank + 1, id == chosen, id == other);
            }
        }
        return;
    }
    let weights: Vec<_> = candidates
        .iter()
        .map(|(_, value)| ((*value - best) / config.temperature).exp())
        .collect();
    let sum: f64 = weights.iter().map(|&w| w as f64).sum();
    let mut cumulative = 0.0f64;
    for (rank, (&(id, value), &weight)) in candidates.iter().zip(&weights).enumerate() {
        let lower = cumulative / sum;
        cumulative += weight as f64;
        let upper = cumulative / sum;
        if rank < 8 || id == chosen || id == other {
            let margin = draw
                .map(|d| ((d as f64) - lower).min(upper - d as f64))
                .unwrap_or(f64::NAN);
            println!("diagnose_candidate,{label},rank={},id={id},logit={value:.9e},probability={:.9e},cdf_lower={lower:.12},cdf_upper={upper:.12},draw_margin={margin:.9e},chosen={},other={}",
                rank + 1, weight as f64 / sum, id == chosen, id == other);
        }
    }
}

pub fn check(
    dir: PathBuf,
    quant: &str,
    steps: usize,
    fuzz: usize,
    diagnostic: Option<u64>,
) -> Result<()> {
    let cfg = Config::from_file(dir.join("config.json"))?;
    ensure!(
        steps > 0 && cfg.block_size > steps + 941,
        "requires context > 941 + steps"
    );
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant).context("unknown precision")?;
    let mut reference = load(&cfg, &weights, precision)?;
    reference.set_prefill_attention(PrefillAttentionVariant::Reference)?;
    let mut candidate = load(&cfg, &weights, precision)?;
    println!("Attention A/B: canonical reference versus {}; precision {quant}; exact token and seeded RNG comparison", candidate.prefill_attention().name());
    if let Some(id) = diagnostic {
        return diagnose_request(&cfg, &mut reference, &mut candidate, steps, id);
    }
    malformed_metadata_check(&mut candidate, &cfg)?;
    let lengths = [
        1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 941,
    ];
    let specs: Vec<_> = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| spec(100 + i as u64, len, cfg.vocab_size, steps))
        .collect();
    let mut references = BTreeMap::new();
    for spec in &specs {
        references.insert(spec.id, independent(&mut reference, spec)?);
    }
    let mut comparisons = 0;
    let mut calls = 0;
    let mut max_delta = 0.0f32;
    for chunk in [32, 64, 128, 256, 37, 73, 131] {
        for (group, set) in [(&specs[..16]), (&specs[3..])].into_iter().enumerate() {
            for order in 0..3 {
                let (n, delta) = packed(
                    &mut candidate,
                    &cfg,
                    set,
                    chunk,
                    order,
                    order != 0,
                    &references,
                )?;
                calls += n;
                max_delta = max_delta.max(delta);
                comparisons += set.len();
            }
            println!("  boundaries group {group}, chunk {chunk}: three permutations exact");
        }
    }
    for count in [1, 2, 4, 8, 16, 1, 8, 2] {
        let (n, delta) = packed(
            &mut candidate,
            &cfg,
            &specs[..count],
            73,
            2,
            true,
            &references,
        )?;
        calls += n;
        max_delta = max_delta.max(delta);
        comparisons += count;
    }
    println!("  composition and stale metadata 1/2/4/8/16/1/8/2: exact");
    for count in [1, 2, 3, 4, 8, 16] {
        let set: Vec<_> = (0..count)
            .map(|i| spec(1000 + i as u64, 17, cfg.vocab_size, steps))
            .collect();
        for spec in &set {
            references.insert(spec.id, independent(&mut reference, spec)?);
        }
        let (n, delta) = packed(&mut candidate, &cfg, &set, 17, 1, false, &references)?;
        calls += n;
        max_delta = max_delta.max(delta);
        comparisons += set.len();
    }
    println!("  simultaneous final rows 1/2/3/4/8/16 with mixed sampling: exact");
    let mut random = 0x2026_0905_0042u64;
    for round in 0..fuzz {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let count = 2 + random as usize % 15;
        let set: Vec<_> = (0..count)
            .map(|index| {
                let id = 10_000 + round as u64 * 16 + index as u64;
                let len = 1 + (random.rotate_left(index as u32) as usize % 512);
                spec(id, len, cfg.vocab_size, steps)
            })
            .collect();
        for spec in &set {
            references.insert(spec.id, independent(&mut reference, spec)?);
        }
        let chunk = [37, 73, 131, 64, 128, 256][round % 6];
        let (n, delta) = packed(
            &mut candidate,
            &cfg,
            &set,
            chunk,
            round % 3,
            true,
            &references,
        )?;
        calls += n;
        max_delta = max_delta.max(delta);
        comparisons += count;
        println!("  fixed-seed fuzz set {round}: {count} requests exact");
    }
    let concurrent_specs: Vec<_> = (0..16)
        .map(|i| {
            spec(
                20_000 + i as u64,
                17 + i,
                cfg.vocab_size,
                if i % 2 == 0 { 6 } else { 12 },
            )
        })
        .collect();
    for spec in &concurrent_specs {
        references.insert(spec.id, independent(&mut reference, spec)?);
    }
    comparisons += runtime_checks(
        candidate,
        &cfg,
        &weights,
        precision,
        &specs[10..],
        &concurrent_specs,
        &references,
    )?;
    println!("PASS: {comparisons} exact generated sequences; {calls} packed GPU calls; {fuzz} fixed-seed fuzz sets");
    println!("Maximum final-logit absolute difference vs monolithic: {max_delta:.6e} (reported, no token tolerance)");
    Ok(())
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) * 0.5
    } else {
        samples[middle]
    }
}

pub fn bench(
    dir: PathBuf,
    quant: &str,
    iters: usize,
    batches: &str,
    chunks: &str,
    packed_only: bool,
) -> Result<()> {
    ensure!(iters > 0, "iters must be positive");
    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant).context("unknown precision")?;
    let mut model = load(&cfg, &weights, precision)?;
    let parse = |values: &str| -> Result<Vec<usize>> {
        values
            .split(',')
            .map(|value| {
                let number = value
                    .trim()
                    .parse::<usize>()
                    .context("invalid benchmark shape")?;
                ensure!(number > 0, "benchmark shapes must be positive");
                Ok(number)
            })
            .collect()
    };
    let batches = parse(batches)?;
    let chunks = parse(chunks)?;
    ensure!(
        batches
            .iter()
            .all(|&n| n <= model.prefill_request_capacity()),
        "batch exceeds packed capacity"
    );
    ensure!(
        chunks.iter().all(|&n| n <= model.prefill_token_capacity()),
        "chunk exceeds packed capacity"
    );
    println!("requests,chunk,rows,finals,serial_eager_ms,serial_graph_ms,packed_wall_ms,packed_gpu_ms,speedup,packed_spread_pct,d2h_bytes");
    if packed_only {
        println!("# CUDA-event profiling: three rounds per shape; stage_ms is the mean summed stage interval per pass; operations_per_pass aggregates all layers.");
        println!("# Event intervals exclude metadata H2D and output readback; event insertion can perturb execution and intervals include host submission gaps. Compare with the uninstrumented wall/replay timings above.");
        println!("# packed_event,requests,chunk,rows,finals,round,iters,stage,operations_per_pass,stage_ms");
    }
    for count in batches {
        for &chunk in &chunks {
            if count * chunk > cfg.block_size {
                continue;
            }
            let prompts: Vec<_> = (0..count)
                .map(|i| spec(i as u64 + 100, chunk, cfg.vocab_size, 1).prompt)
                .collect();
            let mut pages = Vec::new();
            for _ in 0..count {
                pages.push(allocate(&mut model, chunk)?);
            }
            let tables: Vec<_> = pages
                .iter()
                .map(|s| s.table_padded(model.table_stride()))
                .collect();
            let descriptors: Vec<_> = (0..count)
                .map(|i| PackedPrefillRequest {
                    tokens: &prompts[i],
                    page_table: &tables[i],
                    pos_offset: 0,
                    want_logits: true,
                })
                .collect();
            // Warm both paths/captures before paired measurements.
            for _ in 0..3 {
                if !packed_only {
                    model.set_prefill_graph(true);
                    for i in 0..count {
                        model.prefill_chunk(&prompts[i], &tables[i], 0, true)?;
                    }
                }
                model.prefill_packed(&descriptors, &[], &[])?;
            }
            let mut eager = Vec::new();
            let mut graph = Vec::new();
            let mut packed_times = Vec::new();
            let mut bytes = 0;
            for iteration in 0..iters {
                let modes = if iteration % 2 == 0 {
                    [0, 1, 2]
                } else {
                    [2, 1, 0]
                };
                for mode in modes {
                    if packed_only && mode != 2 {
                        continue;
                    }
                    model.gpu.sync()?;
                    let start = Instant::now();
                    if mode == 2 {
                        bytes = model.prefill_packed(&descriptors, &[], &[])?.d2h_bytes;
                    } else {
                        model.set_prefill_graph(mode == 1);
                        for i in 0..count {
                            model.prefill_chunk(&prompts[i], &tables[i], 0, true)?;
                        }
                    }
                    model.gpu.sync()?;
                    let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
                    match mode {
                        0 => eager.push(milliseconds),
                        1 => graph.push(milliseconds),
                        _ => packed_times.push(milliseconds),
                    }
                }
            }
            model.prefill_packed(&descriptors, &[], &[])?;
            let gpu = model.time_packed_replay(count * chunk, count, iters)? * 1000.0;
            let eager = if eager.is_empty() {
                f64::NAN
            } else {
                median(&mut eager)
            };
            let graph = if graph.is_empty() {
                f64::NAN
            } else {
                median(&mut graph)
            };
            let wall = median(&mut packed_times);
            let spread = (packed_times[packed_times.len() - 1] - packed_times[0]) / wall * 100.0;
            ensure!(
                bytes == count * 4,
                "greedy packed D2H copied more than token IDs"
            );
            println!("{count},{chunk},{},{count},{eager:.4},{graph:.4},{wall:.4},{gpu:.4},{:.3},{spread:.2},{bytes}", count * chunk, graph / wall);
            if packed_only {
                for round in 1..=3 {
                    let profile = model.profile_packed_prefill(&descriptors, &[], &[], iters)?;
                    let mut stage_total = 0.0;
                    let mut operations = 0;
                    for stage in &profile.stages {
                        println!(
                            "packed_event,{count},{chunk},{},{count},{round},{iters},{},{},{:.6}",
                            count * chunk,
                            stage.name,
                            stage.calls,
                            stage.milliseconds
                        );
                        stage_total += stage.milliseconds;
                        operations += stage.calls;
                    }
                    println!("packed_event,{count},{chunk},{},{count},{round},{iters},stage_sum,{operations},{stage_total:.6}", count * chunk);
                    println!("packed_event,{count},{chunk},{},{count},{round},{iters},event_total,,{:.6}",
                        count * chunk, profile.device_milliseconds);
                }
            }
            for seq in &mut pages {
                seq.release(model.page_pool_mut())?;
            }
        }
    }
    if model.page_pool().used_pages() != 0 {
        bail!("benchmark leaked pages");
    }
    Ok(())
}
