//! Opt-in runtime timing diagnostics. All request histories and printing live
//! here, outside the serving path; HTTP TTFT remains measured by its client.

use anyhow::{ensure, Context, Result};
use llm_engine::gpu_model::{GpuModel, Precision};
use llm_engine::paged::PAGE_TOKENS;
use llm_engine::prefill::PrefillBatchPlan;
use llm_engine::runtime::{FinishReason, Request, Runtime};
use llm_engine::{Config, Weights};
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct RequestTrace {
    id: u64,
    prompt_len: usize,
    admitted_at: Option<Instant>,
    last_call_end: Option<Instant>,
    prefill_wait: Duration,
    shared_call_wall: Duration,
    calls: usize,
    first_token: Option<(Instant, usize)>,
}

struct CallTrace {
    step: u64,
    plan: PrefillBatchPlan,
    planning: Duration,
    execution: Duration,
    packed: bool,
}

struct Trace {
    submitted_at: Instant,
    wall: Duration,
    requests: Vec<RequestTrace>,
    calls: Vec<CallTrace>,
}

fn run(runtime: &mut Runtime, lengths: &[usize], steps: usize) -> Result<Trace> {
    ensure!(runtime.is_idle(), "trace must start with an idle runtime");
    let free_before = runtime.free_pages();
    let vocab = runtime.model().cfg.vocab_size;
    let requests: Vec<_> = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let id = 1001 + i as u64;
            let prompt = (0..len)
                .map(|pos| (id as usize * 977 + pos * 613 + pos * pos * 17) % vocab)
                .collect();
            Request::greedy(id, prompt, steps)
        })
        .collect();
    let records = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| RequestTrace {
            id: 1001 + i as u64,
            prompt_len: len,
            admitted_at: None,
            last_call_end: None,
            prefill_wait: Duration::ZERO,
            shared_call_wall: Duration::ZERO,
            calls: 0,
            first_token: None,
        })
        .collect();
    let mut trace = Trace {
        submitted_at: Instant::now(),
        wall: Duration::ZERO,
        requests: records,
        calls: Vec::new(),
    };
    // One burst with no intervening step: every request is pending together.
    for request in requests {
        runtime.submit(request)?;
    }
    while !runtime.is_idle() {
        let info = runtime.step()?;
        let observed_at = Instant::now();
        for &id in &info.admitted {
            let record = &mut trace.requests[(id - 1001) as usize];
            record.admitted_at = Some(info.admitted_at.context("missing admission timestamp")?);
        }
        if info.prefill_batches > 0 {
            ensure!(
                info.prefill_batches == 1,
                "trace expects one packed-policy plan per step"
            );
            let started = info
                .prefill_started_at
                .context("missing prefill timestamp")?;
            let ended = started + info.prefill_execution_duration;
            let plan = runtime.last_prefill_plan();
            for slice in &plan.slices {
                let record = &mut trace.requests[(slice.request_id - 1001) as usize];
                let waiting_since = record
                    .last_call_end
                    .or(record.admitted_at)
                    .context("prefill before admission")?;
                record.prefill_wait += started.duration_since(waiting_since);
                record.shared_call_wall += info.prefill_execution_duration;
                record.last_call_end = Some(ended);
                record.calls += 1;
            }
            trace.calls.push(CallTrace {
                step: info.step,
                plan: plan.clone(),
                planning: info.prefill_planning_duration,
                execution: info.prefill_execution_duration,
                packed: info.packed_prefill_batches > 0,
            });
        } else {
            ensure!(
                info.prefill_started_at.is_none()
                    && info.prefill_planning_duration.is_zero()
                    && info.prefill_execution_duration.is_zero(),
                "decode-only step recorded prefill timing"
            );
        }
        for &(id, token) in &info.tokens {
            trace.requests[(id - 1001) as usize]
                .first_token
                .get_or_insert((observed_at, token));
        }
    }
    trace.wall = trace.submitted_at.elapsed();
    let completions = runtime.completed();
    ensure!(
        completions.len() == lengths.len(),
        "missing trace completions"
    );
    ensure!(
        completions
            .iter()
            .all(|c| c.reason == FinishReason::Length && c.tokens.len() == steps),
        "trace request did not finish its token budget"
    );
    ensure!(runtime.free_pages() == free_before, "trace leaked KV pages");
    ensure!(
        trace.requests.iter().all(|r| r.first_token.is_some()),
        "missing first token"
    );
    Ok(trace)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub fn trace(
    dir: PathBuf,
    quant: &str,
    lengths: &str,
    steps: usize,
    max_batch: usize,
    budget: Option<usize>,
    trials: usize,
) -> Result<()> {
    ensure!(
        steps > 0 && trials > 0 && max_batch > 0,
        "steps, trials and max-batch must be positive"
    );
    let cfg = Config::from_file(dir.join("config.json"))?;
    let lengths: Vec<usize> = lengths
        .split(',')
        .map(|v| {
            let len = v.trim().parse::<usize>().context("invalid prompt length")?;
            ensure!(
                len > 0
                    && len
                        .checked_add(steps - 1)
                        .is_some_and(|n| n <= cfg.block_size),
                "prompt {len} plus output exceeds context {}",
                cfg.block_size
            );
            Ok(len)
        })
        .collect::<Result<_>>()?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant).context("unknown precision")?;
    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(cfg.block_size.div_ceil(PAGE_TOKENS) * max_batch, max_batch)?;
    let mut runtime = Runtime::new(model)?;
    runtime.set_batched_prefill(true);
    runtime.set_chunked_prefill(false);
    if let Some(budget) = budget {
        runtime.set_prefill_token_budget(budget)?;
    }

    // Warm exact workload/captures before timing; preserve all printing until
    // after the measured runs, including per-plan compositions.
    let warm = run(&mut runtime, &lengths, steps)?;
    let mut traces = Vec::with_capacity(trials);
    for _ in 0..trials {
        let trace = run(&mut runtime, &lengths, steps)?;
        ensure!(
            trace
                .requests
                .iter()
                .zip(&warm.requests)
                .all(|(a, b)| { a.first_token.map(|(_, t)| t) == b.first_token.map(|(_, t)| t) }),
            "first token changed between identical trace workloads"
        );
        traces.push(trace);
    }

    println!("runtime-only submission burst; excludes HTTP, tokenization and network delivery");
    println!("queue_wait ends at admission completion; prefill_wait includes scheduler/CPU planning and intervening decode");
    println!("shared_call_wall is each participating call's full host wall time, not exclusive GPU time; do not sum it across requests");
    println!("first_token_observed is Runtime::step return; queue_wait + prefill_wait + shared_call_wall + post_call = runtime_ttft");
    println!(
        "budget={},max_batch={},steps={},warmups=1,trials={}",
        runtime.prefill_token_budget(),
        max_batch,
        steps,
        trials
    );
    println!("trial,id,prompt_tokens,calls,queue_wait_ms,prefill_wait_ms,shared_call_wall_ms,post_call_ms,runtime_ttft_ms,first_token");
    for (trial, trace) in traces.iter().enumerate() {
        for record in &trace.requests {
            let admitted = record.admitted_at.context("missing admission")?;
            let (first_at, token) = record.first_token.context("missing first token")?;
            let ended = record.last_call_end.context("missing prefill call")?;
            println!(
                "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                trial + 1,
                record.id,
                record.prompt_len,
                record.calls,
                ms(admitted.duration_since(trace.submitted_at)),
                ms(record.prefill_wait),
                ms(record.shared_call_wall),
                ms(first_at.duration_since(ended)),
                ms(first_at.duration_since(trace.submitted_at)),
                token
            );
        }
    }
    println!("trial,step,path,tokens,final_rows,planning_us,call_wall_ms,request_slices(id:offset+length:final)");
    for (trial, trace) in traces.iter().enumerate() {
        for call in &trace.calls {
            let slices = call
                .plan
                .slices
                .iter()
                .map(|slice| {
                    format!(
                        "{}:{}+{}:{}",
                        slice.request_id,
                        slice.prompt_start,
                        slice.len,
                        usize::from(slice.is_final)
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            println!(
                "{},{},{},{},{},{:.3},{:.6},{}",
                trial + 1,
                call.step,
                if call.packed { "packed" } else { "singleton" },
                call.plan.tokens,
                call.plan.final_rows,
                ms(call.planning) * 1000.0,
                ms(call.execution),
                slices
            );
        }
    }
    let planning: Duration = traces
        .iter()
        .flat_map(|t| &t.calls)
        .map(|c| c.planning)
        .sum();
    let execution: Duration = traces
        .iter()
        .flat_map(|t| &t.calls)
        .map(|c| c.execution)
        .sum();
    let wall: Duration = traces.iter().map(|t| t.wall).sum();
    let calls: usize = traces.iter().map(|t| t.calls.len()).sum();
    println!("aggregate: calls={},planning_us={:.3},planning_us_per_call={:.3},prefill_call_wall_ms={:.3},runtime_wall_ms={:.3},planning_pct_prefill={:.6},planning_pct_runtime={:.6}",
        calls, ms(planning) * 1000.0, ms(planning) * 1000.0 / calls as f64,
        ms(execution), ms(wall), 100.0 * planning.as_secs_f64() / (planning + execution).as_secs_f64(),
        100.0 * planning.as_secs_f64() / wall.as_secs_f64());
    Ok(())
}
