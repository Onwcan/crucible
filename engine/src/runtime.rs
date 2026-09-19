//! Continuous batching with a single GPU owner, paged KV and bounded prefill.
//!
//! Admission is FCFS and reserves the maximum pages a request can require,
//! while physical generation pages still grow lazily. Active plus prefilling
//! requests never exceed max_batch. Each request owns its pages and RNG;
//! decoding slots may move without changing request identity.
//!
//! Packed prefill is enabled by default with an explicit A/B control. A step
//! admits cheap metadata, decodes existing
//! streams, executes one token-budgeted prefill plan, then retires completions.
//! Plans take one bounded slice per request from a round-robin queue. Unfinished
//! slices rejoin its back; arrivals join there too. A survivor receives work
//! within R nonempty plans when at most R requests are resident. Real packed
//! rows, requests per call and optional per-request chunks all have explicit
//! limits; the CPU-only planner is tested independently of CUDA.
//!
//! The reference path remains available: monolithic prefill runs before decode,
//! while opt-in single-request chunking runs after decode. Existing prefill
//! CUDA graphs remain on for that reference. Packed execution initially uses
//! eager kernels; it shares transformer arithmetic with the reference.
//!
//! Cancellation is observed between scheduler steps. Once a plan is submitted,
//! that bounded batch finishes before cancellation is applied. All GPU work
//! uses the same ordered stream, so pages returned at a boundary cannot be
//! overwritten by a new owner before earlier work completes. A runtime error
//! is fatal to its owning server thread; corrupted state is never reused.
use anyhow::{bail, Result};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::gpu_model::{GpuModel, PackedPrefillRequest};
use crate::paged::SequencePages;
use crate::prefill::{
    request_page_reservation, validate_request, validate_request_identity, PageReservations,
    PrefillBatchPlan, PrefillBudget, PrefillWork,
};
use crate::sampling::{self, GenerationConfig, Rng};

/// Work submitted to the runtime.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<usize>,
    /// Immutable for the request's lifetime. Every resident request may have a
    /// different one; nothing is imposed on a batch as a whole.
    pub config: GenerationConfig,
}

impl Request {
    /// A greedy request, which is what every caller got before sampling
    /// existed.
    pub fn greedy(id: u64, prompt: Vec<usize>, max_new_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            config: GenerationConfig::greedy(max_new_tokens),
        }
    }
}

/// Why a request left the active set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached its token budget.
    Length,
    /// Withdrawn before finishing, typically because the client went away.
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

/// A request that has left the active set.
#[derive(Debug, Clone)]
pub struct Completion {
    pub id: u64,
    pub prompt_len: usize,
    pub tokens: Vec<usize>,
    /// Step index at which it left the batch, for deterministic tests.
    pub finished_at: u64,
    pub reason: FinishReason,
}

/// A request whose prompt is still being consumed.
///
/// The state that used to be implicit in "between `pending` and `active`" and
/// lasted only as long as one blocking call. Making it explicit is what lets a
/// prompt be advanced a chunk at a time, and what gives cancellation something
/// to find when a client leaves mid-prefill.
struct Prefilling {
    id: u64,
    seq: SequencePages,
    prompt: Vec<usize>,
    /// Prompt tokens already written to this request's pages. The next chunk
    /// starts here, and this is the `pos_offset` the model is given.
    done: usize,
    config: GenerationConfig,
    reserved_pages: usize,
}

/// One resident request.
struct Active {
    id: u64,
    seq: SequencePages,
    prompt_len: usize,
    /// The token to feed at the next decode step.
    next_token: usize,
    generated: Vec<usize>,
    config: GenerationConfig,
    reserved_pages: usize,
    /// This request's own RNG.
    ///
    /// Owned by the request, not by the slot: `retire` and `cancel` use
    /// `swap_remove`, so slot indices are reused by unrelated requests between
    /// steps. State keyed on slot position would make a request's sampled
    /// sequence depend on who else happened to be in the batch, which is
    /// exactly the property that must not hold.
    rng: Rng,
}

/// What one `step` did, so a test or benchmark can assert on it.
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    pub step: u64,
    pub admitted: Vec<u64>,
    pub decoded: usize,
    pub finished: Vec<u64>,
    /// Tokens produced this step, as (request id, token). Admission also
    /// produces a token -- prefill's final logits are the request's first
    /// output -- so a newly admitted request appears here too.
    ///
    /// This exists so a streaming server can forward tokens as they are made
    /// rather than waiting for completion. It reports what the scheduler
    /// already did; it does not change what it does.
    pub tokens: Vec<(u64, usize)>,
    pub active_after: usize,
    pub pending_after: usize,
    pub free_pages: usize,
    /// Bytes this step's decode copied device to host.
    ///
    /// Measured rather than derived: the size of this number is the whole point
    /// of selecting tokens on the device, so a benchmark should not be
    /// reporting a formula that could drift from what the code does.
    pub d2h_bytes: usize,
    /// Prompt tokens this step consumed, and how many chunks that took.
    pub prefill_tokens: usize,
    pub prefill_chunks: usize,
    /// Requests still working through their prompt when the step ended.
    pub prefilling_after: usize,
    /// Calls, slices and completed prompts, counted without GPU synchronization.
    pub prefill_batches: usize,
    pub packed_prefill_batches: usize,
    pub packed_prefill_tokens: usize,
    pub prefill_requests: usize,
    pub prefill_final_rows: usize,
    pub last_prefill_batch_requests: usize,
    pub last_prefill_batch_tokens: usize,
    pub max_prefill_batch_tokens: usize,
    pub prefill_d2h_bytes: usize,
    /// Admission completion for this step, sampled only when requests enter.
    pub admitted_at: Option<Instant>,
    /// Start of the first model prefill call; absent on decode-only/idle steps.
    pub prefill_started_at: Option<Instant>,
    /// CPU plan construction, page descriptors and sampling-route preparation.
    pub prefill_planning_duration: Duration,
    /// Host wall time inside model prefill calls, including required transfers.
    /// This is not a CUDA-event measurement or an exclusive per-request cost.
    pub prefill_execution_duration: Duration,
}

pub struct Runtime {
    model: GpuModel,
    active: Vec<Active>,
    /// Requests that hold pages and are partway through their prompt.
    prefilling: VecDeque<Prefilling>,
    pending: VecDeque<Request>,
    done: Vec<Completion>,
    step_no: u64,
    max_batch: usize,
    /// Optional per-request slice cap. The packed aggregate has its own budget.
    prefill_chunk: usize,
    /// Enables the per-request cap. When false, packed work is still bounded
    /// by the aggregate token budget; the reference consumes whole prompts.
    chunked_prefill: bool,
    batched_prefill: bool,
    prefill_token_budget: usize,
    max_prefill_requests: usize,
    prefill_plan: PrefillBatchPlan,
    prefill_tables: Vec<i32>,
    /// Allocations remain lazy; admission cannot consume reserved decode growth.
    page_reservations: PageReservations,
}

/// Prompt tokens per prefill chunk when chunking is enabled.
///
/// 128 is the least-bad of the measured sizes: it halves the worst-case gap
/// where 256 does not, without the throughput collapse of 32 and 64. It is not
/// a good default, which is why chunking is off unless asked for.
pub const DEFAULT_PREFILL_CHUNK: usize = 128;
/// The measured static budget: combines short prompts without imposing the
/// repeated small-call cost on an isolated full-context prompt. See README.
pub const DEFAULT_PREFILL_TOKEN_BUDGET: usize = 1024;

fn env_positive(name: &str, default: usize, capacity: usize) -> Result<usize> {
    let value = match std::env::var(name) {
        Ok(v) => v
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("{name} must be a positive integer"))?,
        Err(std::env::VarError::NotPresent) => default.min(capacity),
        Err(e) => bail!("could not read {name}: {e}"),
    };
    if value == 0 || value > capacity {
        bail!("{name} must be in 1..={capacity}, got {value}");
    }
    Ok(value)
}

impl Runtime {
    /// `model` must already have paging enabled; the pool it allocated is the
    /// runtime's entire memory budget.
    pub fn new(model: GpuModel) -> Result<Self> {
        if !model.use_paged() {
            bail!("runtime requires a model with paging enabled");
        }
        let max_batch = model.max_batch();
        let capacity = model.prefill_token_capacity();
        let prefill_chunk =
            env_positive("CRUCIBLE_PREFILL_CHUNK", DEFAULT_PREFILL_CHUNK, capacity)?;
        let prefill_token_budget = env_positive(
            "CRUCIBLE_PREFILL_TOKEN_BUDGET",
            DEFAULT_PREFILL_TOKEN_BUDGET,
            capacity,
        )?;
        let request_capacity = model.prefill_request_capacity();
        let max_prefill_requests = env_positive(
            "CRUCIBLE_MAX_PREFILL_REQUESTS",
            request_capacity,
            request_capacity,
        )?;
        let prefill_tables = vec![0; max_batch * model.table_stride()];
        let page_reservations = PageReservations::new(model.page_pool().n_pages());
        Ok(Self {
            model,
            active: Vec::with_capacity(max_batch),
            prefilling: VecDeque::with_capacity(max_batch),
            pending: VecDeque::new(),
            done: Vec::new(),
            step_no: 0,
            max_batch,
            prefill_chunk,
            // Off unless asked for: measured slower on every metric but the
            // worst-case gap, and that one improves by only 1.7x.
            chunked_prefill: std::env::var("CRUCIBLE_CHUNKED_PREFILL").as_deref() == Ok("1"),
            batched_prefill: std::env::var("CRUCIBLE_BATCHED_PREFILL").as_deref() != Ok("0"),
            prefill_token_budget,
            max_prefill_requests,
            prefill_plan: PrefillBatchPlan::with_capacity(max_batch),
            prefill_tables,
            page_reservations,
        })
    }

    /// Prompt tokens consumed per step.
    pub fn set_prefill_chunk(&mut self, tokens: usize) {
        self.prefill_chunk = tokens.clamp(1, self.model.prefill_token_capacity());
    }

    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    /// Enable the per-request chunk cap; independent of packed execution.
    pub fn set_chunked_prefill(&mut self, on: bool) {
        self.chunked_prefill = on;
    }

    pub fn chunked_prefill(&self) -> bool {
        self.chunked_prefill
    }

    pub fn set_batched_prefill(&mut self, on: bool) {
        self.batched_prefill = on;
    }

    pub fn batched_prefill(&self) -> bool {
        self.batched_prefill
    }

    /// Borrow the most recently executed packed-policy plan without copying it.
    /// Inspect only when the returned StepInfo has prefill_batches > 0 and
    /// batched_prefill is enabled; decode-only steps retain the previous plan.
    pub fn last_prefill_plan(&self) -> &PrefillBatchPlan {
        &self.prefill_plan
    }

    pub fn set_prefill_token_budget(&mut self, tokens: usize) -> Result<()> {
        self.prefill_budget(tokens, self.max_prefill_requests)
            .validate(
                self.model.prefill_token_capacity(),
                self.model.prefill_request_capacity(),
            )?;
        self.prefill_token_budget = tokens;
        Ok(())
    }

    pub fn prefill_token_budget(&self) -> usize {
        self.prefill_token_budget
    }

    pub fn set_max_prefill_requests(&mut self, requests: usize) -> Result<()> {
        self.prefill_budget(self.prefill_token_budget, requests)
            .validate(
                self.model.prefill_token_capacity(),
                self.model.prefill_request_capacity(),
            )?;
        self.max_prefill_requests = requests;
        Ok(())
    }

    fn prefill_budget(&self, tokens: usize, requests: usize) -> PrefillBudget {
        PrefillBudget {
            tokens,
            requests,
            chunk: if self.chunked_prefill {
                self.prefill_chunk
            } else {
                self.model.prefill_token_capacity()
            },
        }
    }

    /// Requests holding pages but not yet decoding.
    pub fn prefilling_len(&self) -> usize {
        self.prefilling.len()
    }

    /// Reject malformed work and duplicate live identity before queue or KV
    /// state changes. A rejected caller cannot invalidate neighboring requests.
    pub fn submit(&mut self, req: Request) -> Result<()> {
        validate_request(
            &req.prompt,
            &req.config,
            self.model.cfg.vocab_size,
            self.model.prefill_token_capacity(),
            self.model.page_pool().n_pages(),
        )?;
        validate_request_identity(
            req.id,
            self.pending
                .iter()
                .map(|r| r.id)
                .chain(self.prefilling.iter().map(|r| r.id))
                .chain(self.active.iter().map(|r| r.id)),
        )?;
        self.pending.push_back(req);
        Ok(())
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.pending.is_empty() && self.prefilling.is_empty()
    }

    /// Drain everything that has finished since the last call.
    pub fn completed(&mut self) -> Vec<Completion> {
        std::mem::take(&mut self.done)
    }

    pub fn model(&self) -> &GpuModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut GpuModel {
        &mut self.model
    }

    pub fn free_pages(&self) -> usize {
        self.model.page_pool().free_pages()
    }

    /// Pages currently held by resident requests, and slots wasted inside them.
    ///
    /// Counts prefilling requests too: they hold their prompt's pages from
    /// admission, so leaving them out would under-report occupancy exactly
    /// when a long prompt is being consumed.
    pub fn residency(&self) -> (usize, usize) {
        let pages: usize = self.active.iter().map(|a| a.seq.n_pages()).sum::<usize>()
            + self
                .prefilling
                .iter()
                .map(|p| p.seq.n_pages())
                .sum::<usize>();
        let wasted: usize = self
            .active
            .iter()
            .map(|a| a.seq.wasted_slots())
            .sum::<usize>()
            + self
                .prefilling
                .iter()
                .map(|p| p.seq.wasted_slots())
                .sum::<usize>();
        (pages, wasted)
    }

    /// Admit what fits, decode every active request one token, retire whoever
    /// finished.
    pub fn step(&mut self) -> Result<StepInfo> {
        let mut info = StepInfo {
            step: self.step_no,
            ..Default::default()
        };

        // Admission is now cheap: it takes pages and nothing else, so a step
        // that accepts a long prompt costs the same as one that accepts a short
        // one. The prompt itself is consumed below, in bounded pieces.
        info.admitted = self.admit()?;
        if !info.admitted.is_empty() {
            info.admitted_at = Some(Instant::now());
        }

        // Monolithic control path: prompts are consumed whole, before any
        // decoding, which is what the runtime did before chunking.
        if !self.chunked_prefill && !self.batched_prefill {
            let (t, c) = self.advance_prefill(&mut info, usize::MAX)?;
            info.prefill_tokens += t;
            info.prefill_chunks += c;
            info.prefill_batches += c;
            info.prefill_requests += c;
            // A first token can complete a request. Do not decode it a second
            // time when max_tokens=1, including on the preserved reference path.
            info.finished.extend(self.retire()?);
        }

        if !self.active.is_empty() {
            let (decoded, d2h) = self.decode_active(&mut info.tokens)?;
            info.decoded = decoded;
            info.d2h_bytes = d2h;
        }

        // Decode first, then exactly one bounded chunk. A decode step is
        // guaranteed before every chunk and a chunk after every decode step, so
        // neither class can starve the other and there is no ratio to tune.
        if self.batched_prefill {
            self.advance_packed_prefill(&mut info)?;
        } else if self.chunked_prefill {
            let (t, c) = self.advance_prefill(&mut info, self.prefill_chunk)?;
            info.prefill_tokens += t;
            info.prefill_chunks += c;
            info.prefill_batches += c;
            info.prefill_requests += c;
        }

        info.finished.extend(self.retire()?);

        info.prefilling_after = self.prefilling.len();
        info.active_after = self.active.len();
        info.pending_after = self.pending.len();
        info.free_pages = self.model.page_pool().free_pages();
        self.step_no += 1;
        Ok(info)
    }

    /// Move pending requests into the prefilling set while slots and pages allow.
    ///
    /// This is where a request's prompt pages are taken, and it is all this
    /// does: no GPU work, so the cost of admitting a request no longer depends
    /// on how long its prompt is. A prompt that cannot get pages stays queued
    /// and stops admission for this step -- FCFS, so a large request is not
    /// starved by smaller ones queued behind it.
    ///
    /// Prompt pages are still reserved in full rather than progressively. The
    /// request either fits now or waits, which keeps allocator exhaustion a
    /// decision made once, at a point where nothing has been written and
    /// nothing has to be unwound. Generation pages continue to grow one step at
    /// a time, as before.
    fn admit(&mut self) -> Result<Vec<u64>> {
        let mut admitted = Vec::new();
        // Prefilling requests hold a slot: they own pages and are on their way
        // into the batch, so counting only `active` would let the scheduler
        // over-admit and then find no room.
        while self.active.len() + self.prefilling.len() < self.max_batch {
            let Some(req) = self.pending.front() else {
                break;
            };
            if req.prompt.is_empty() {
                bail!("request {} has an empty prompt", req.id);
            }

            let reservation = request_page_reservation(
                req.prompt.len(),
                req.config.max_tokens,
                self.model.prefill_token_capacity(),
            )?;
            if !self.page_reservations.try_reserve(reservation)? {
                break;
            }

            let mut seq = SequencePages::new();
            if seq
                .grow(self.model.page_pool_mut(), req.prompt.len())
                .is_err()
            {
                self.page_reservations.release(reservation)?;
                // Not enough pages right now. Leave it queued; a retirement
                // later this step or next will free some.
                break;
            }

            let req = self.pending.pop_front().expect("front checked above");
            admitted.push(req.id);
            self.prefilling.push_back(Prefilling {
                id: req.id,
                seq,
                prompt: req.prompt,
                done: 0,
                config: req.config,
                reserved_pages: reservation,
            });
        }
        Ok(admitted)
    }

    /// Consume up to `budget` prompt tokens for the oldest prefilling request.
    ///
    /// Returns the tokens consumed and the chunks it took. A chunk is written
    /// straight into the request's own pages at `pos_offset = done`, so there
    /// is no staging buffer and no copy: prefill and decode share one page
    /// mapping for the whole life of the request.
    ///
    /// Attention for a chunk row `r` covers `0 ..= done + r`, which is the
    /// entire prefix already cached plus the causally earlier part of this
    /// chunk. Nothing is recomputed and nothing is missed, and because the
    /// kernel walks that range in the same order regardless of where the chunk
    /// boundaries fall, the cache it produces does not depend on them.
    fn advance_prefill(&mut self, info: &mut StepInfo, budget: usize) -> Result<(usize, usize)> {
        let mut tokens = 0usize;
        let mut chunks = 0usize;
        let stride = self.model.table_stride();

        // One chunk per call in the chunked policy; the control path passes an
        // unbounded budget and loops until the prompt is gone.
        loop {
            let Some(p) = self.prefilling.front() else {
                break;
            };
            let remaining = p.prompt.len() - p.done;
            let take = remaining.min(budget.max(1));
            let last = take == remaining;

            let (id, done, table) = {
                let p = &self.prefilling[0];
                (p.id, p.done, p.seq.table_padded(stride))
            };
            let chunk: Vec<usize> = {
                let p = &self.prefilling[0];
                p.prompt[p.done..p.done + take].to_vec()
            };
            // Only the last chunk's logits become a token; earlier ones skip
            // the lm_head projection and its device-to-host copy entirely.
            let started = Instant::now();
            info.prefill_started_at.get_or_insert(started);
            let logits = self.model.prefill_chunk(&chunk, &table, done, last)?;
            info.prefill_execution_duration += started.elapsed();

            tokens += take;
            chunks += 1;
            info.last_prefill_batch_requests = 1;
            info.last_prefill_batch_tokens = take;
            info.max_prefill_batch_tokens = info.max_prefill_batch_tokens.max(take);
            self.prefilling[0].done += take;

            if !last {
                if budget == usize::MAX {
                    continue;
                }
                break;
            }

            // Prompt consumed. The final position's logits are this request's
            // first token, and its RNG is created here and used immediately --
            // the first sampled token draws the first random number, exactly as
            // running alone would.
            let p = self
                .prefilling
                .pop_front()
                .expect("prefill front was executed");
            let mut rng = Rng::new(p.config.seed);
            let first = sampling::sample(&logits, &p.config, &mut rng);
            info.tokens.push((id, first));
            info.prefill_final_rows += 1;
            info.prefill_d2h_bytes += logits.len() * std::mem::size_of::<f32>();
            self.active.push(Active {
                id: p.id,
                seq: p.seq,
                prompt_len: p.prompt.len(),
                next_token: first,
                generated: vec![first],
                config: p.config,
                reserved_pages: p.reserved_pages,
                rng,
            });
            if budget == usize::MAX {
                continue;
            }
            break;
        }
        Ok((tokens, chunks))
    }

    /// Execute one round-robin plan. Cancellation is observed before a step;
    /// once this plan exists it completes as one bounded unit. No page or queue
    /// mutation occurs between descriptor construction and the GPU submission.
    fn advance_packed_prefill(&mut self, info: &mut StepInfo) -> Result<()> {
        if self.prefilling.is_empty() {
            return Ok(());
        }
        let planning_started = Instant::now();
        let budget = self.prefill_budget(self.prefill_token_budget, self.max_prefill_requests);
        self.prefill_plan.build(
            self.prefilling.iter().map(|p| PrefillWork {
                request_id: p.id,
                prompt_len: p.prompt.len(),
                done: p.done,
            }),
            budget,
        )?;
        let stride = self.model.table_stride();
        let count = self.prefill_plan.slices.len();
        self.prefill_tables[..count * stride].fill(0);
        let mut topk_rows = Vec::with_capacity(self.prefill_plan.final_rows);
        let mut full_rows = Vec::with_capacity(self.prefill_plan.final_rows);
        let mut final_row = 0;
        let vocab = self.model.cfg.vocab_size;
        let cap = self.model.topk_capacity();
        let device_topk = self.model.device_topk();
        for (i, (p, slice)) in self
            .prefilling
            .iter()
            .zip(&self.prefill_plan.slices)
            .enumerate()
        {
            if p.id != slice.request_id || p.done != slice.prompt_start {
                bail!(
                    "prefill plan no longer matches request {}",
                    slice.request_id
                );
            }
            let table = &mut self.prefill_tables[i * stride..(i + 1) * stride];
            for (out, page) in table.iter_mut().zip(p.seq.pages()) {
                *out = *page as i32;
            }
            if slice.is_final {
                if !p.config.is_greedy() {
                    let k = p.config.top_k.clamp(1, vocab);
                    if device_topk && k <= cap {
                        topk_rows.push((final_row, k));
                    } else {
                        full_rows.push(final_row);
                    }
                }
                final_row += 1;
            }
        }
        // Token data is borrowed from resident prompts. Only the small array
        // of request views is temporary; token and page metadata capacity is
        // persistent, and the GPU owns all its scratch allocations.
        let chunks: Vec<_> = self
            .prefilling
            .iter()
            .zip(&self.prefill_plan.slices)
            .enumerate()
            .map(|(i, (p, slice))| PackedPrefillRequest {
                tokens: &p.prompt[slice.prompt_start..slice.prompt_start + slice.len],
                page_table: &self.prefill_tables[i * stride..(i + 1) * stride],
                pos_offset: slice.prompt_start,
                want_logits: slice.is_final,
            })
            .collect();
        // Paired measurements put the crossover at two requests: a singleton
        // benefits from the existing exact-length graph, while two or more
        // requests amortize the transformer over their combined token rows.
        let packed = count > 1;
        let started = Instant::now();
        info.prefill_planning_duration = started.duration_since(planning_started);
        info.prefill_started_at = Some(started);
        let selection = if packed {
            self.model.prefill_packed(&chunks, &topk_rows, &full_rows)?
        } else {
            self.model
                .prefill_single_mixed(&chunks[0], &topk_rows, &full_rows)?
        };
        info.prefill_execution_duration = started.elapsed();
        drop(chunks);

        final_row = 0;
        let mut full_index = 0;
        for slice in &self.prefill_plan.slices {
            let mut p = self
                .prefilling
                .pop_front()
                .expect("one queue entry per planned slice");
            debug_assert_eq!(p.id, slice.request_id);
            p.done += slice.len;
            if !slice.is_final {
                self.prefilling.push_back(p);
                continue;
            }
            let mut rng = Rng::new(p.config.seed);
            let first = if p.config.is_greedy() {
                selection.ids[final_row]
            } else if full_rows.get(full_index) == Some(&final_row) {
                let base = full_index * vocab;
                full_index += 1;
                sampling::sample(&selection.full[base..base + vocab], &p.config, &mut rng)
            } else {
                let k = p.config.top_k.clamp(1, vocab);
                let base = final_row * cap;
                let mut candidates = Vec::with_capacity(k);
                for j in 0..k {
                    let id = selection.cand_ids[base + j];
                    if id < 0 {
                        bail!(
                            "packed top-k returned fewer than {k} candidates for request {}",
                            p.id
                        );
                    }
                    candidates.push((id as usize, selection.cand_vals[base + j]));
                }
                sampling::sample_candidates(&candidates, &p.config, &mut rng)
            };
            info.tokens.push((p.id, first));
            self.active.push(Active {
                id: p.id,
                seq: p.seq,
                prompt_len: p.prompt.len(),
                next_token: first,
                generated: vec![first],
                config: p.config,
                reserved_pages: p.reserved_pages,
                rng,
            });
            final_row += 1;
        }
        info.prefill_tokens += self.prefill_plan.tokens;
        info.prefill_chunks += count;
        info.prefill_batches += 1;
        info.packed_prefill_batches += usize::from(packed);
        if packed {
            info.packed_prefill_tokens += self.prefill_plan.tokens;
        }
        info.prefill_requests += count;
        info.prefill_final_rows += self.prefill_plan.final_rows;
        info.last_prefill_batch_requests = count;
        info.last_prefill_batch_tokens = self.prefill_plan.tokens;
        info.max_prefill_batch_tokens = self.prefill_plan.tokens;
        info.prefill_d2h_bytes += selection.d2h_bytes;
        Ok(())
    }

    /// One batched decode step across every active request.
    ///
    /// Returns the rows decoded and the bytes the step copied back.
    fn decode_active(&mut self, produced: &mut Vec<(u64, usize)>) -> Result<(usize, usize)> {
        let n = self.active.len();
        let stride = self.model.table_stride();

        let mut tokens = Vec::with_capacity(n);
        let mut positions = Vec::with_capacity(n);
        let mut lens = Vec::with_capacity(n);
        let mut tables = vec![0i32; self.max_batch * stride];

        for (i, a) in self.active.iter_mut().enumerate() {
            // The new token occupies the next logical position, so the page for
            // it must exist before the projection writes there.
            let pos = a.seq.len();
            a.seq.grow(self.model.page_pool_mut(), 1)?;
            tokens.push(a.next_token);
            positions.push(pos);
            lens.push((pos + 1) as i32);
            let t = a.seq.table_padded(stride);
            tables[i * stride..(i + 1) * stride].copy_from_slice(&t);
        }

        // What each row needs beyond an argmax id. Greedy rows need nothing
        // more: the device already reduced them to one id, and copying 200 KB
        // per row to rediscover it would undo the argmax work entirely.
        //
        // A sampled row goes to the device top-k kernel when its k fits the
        // kernel capacity, and to the full-logit path when it does not. That
        // decision is per row, not per batch: one request asking for an
        // unusually large top-k must not drag the other fifteen back onto the
        // slow path with it.
        let vocab = self.model.cfg.vocab_size;
        let cap = self.model.topk_capacity();
        let device_topk = self.model.device_topk();
        let mut topk_rows: Vec<(usize, usize)> = Vec::new();
        let mut full_rows: Vec<usize> = Vec::new();
        for (i, a) in self.active.iter().enumerate() {
            if a.config.is_greedy() {
                continue;
            }
            // The same clamp the reference sampler applies, so the kernel is
            // asked for exactly the candidate set `sampling::top_k` would build.
            let k = a.config.top_k.clamp(1, vocab);
            if device_topk && k <= cap {
                topk_rows.push((i, k));
            } else {
                full_rows.push(i);
            }
        }

        // The transformer forward pass stays batched regardless: only token
        // selection diverges, after the logits exist.
        let (next, d2h): (Vec<usize>, usize) = if topk_rows.is_empty()
            && full_rows.is_empty()
            && self.model.device_argmax()
        {
            // Unchanged greedy fast path: n * 4 bytes back, no logits move.
            let ids = self
                .model
                .decode_batch_tokens(&tokens, &positions, &tables, &lens)?;
            let bytes = n * std::mem::size_of::<i32>();
            (ids, bytes)
        } else {
            let sel = self
                .model
                .decode_batch_mixed(&tokens, &positions, &tables, &lens, &topk_rows, &full_rows)?;
            let mut k_of_row = vec![0usize; n];
            for &(r, k) in &topk_rows {
                k_of_row[r] = k;
            }
            let mut out = Vec::with_capacity(n);
            let mut full_chunks = sel.full.chunks_exact(vocab);
            let mut next_full = full_rows.iter().copied().peekable();
            for (i, a) in self.active.iter_mut().enumerate() {
                if a.config.is_greedy() {
                    out.push(sel.ids[i]);
                } else if next_full.peek() == Some(&i) {
                    next_full.next();
                    let row = full_chunks.next().expect("one row per full request");
                    out.push(sampling::sample(row, &a.config, &mut a.rng));
                } else {
                    // Already in canonical order, so the sampler consumes
                    // these exactly as it consumes a host-built candidate
                    // list -- same function, same arithmetic, same token.
                    let k = k_of_row[i];
                    let base = i * cap;
                    let mut cands = Vec::with_capacity(k);
                    for j in 0..k {
                        let id = sel.cand_ids[base + j];
                        if id < 0 {
                            bail!("device top-k returned {j} candidates for row {i}, wanted {k}");
                        }
                        cands.push((id as usize, sel.cand_vals[base + j]));
                    }
                    out.push(sampling::sample_candidates(&cands, &a.config, &mut a.rng));
                }
            }
            (out, sel.d2h_bytes)
        };

        for (a, tok) in self.active.iter_mut().zip(next) {
            a.next_token = tok;
            a.generated.push(tok);
            produced.push((a.id, tok));
        }
        Ok((n, d2h))
    }

    /// Remove finished requests and hand their pages back.
    ///
    /// Removal is by swap, so the surviving requests' slot order changes. That
    /// is deliberate: every per-request quantity travels in the metadata arrays
    /// rebuilt each step, so nothing is tied to a slot index across steps. If
    /// anything were, this is where it would break.
    fn retire(&mut self) -> Result<Vec<u64>> {
        let mut finished = Vec::new();
        let mut i = 0;
        while i < self.active.len() {
            // generated holds the prefill token plus one per decode step.
            if self.active[i].generated.len() >= self.active[i].config.max_tokens {
                let mut a = self.active.swap_remove(i);
                a.seq.release(self.model.page_pool_mut())?;
                self.page_reservations.release(a.reserved_pages)?;
                finished.push(a.id);
                self.done.push(Completion {
                    id: a.id,
                    prompt_len: a.prompt_len,
                    tokens: a.generated,
                    finished_at: self.step_no,
                    reason: FinishReason::Length,
                });
            } else {
                i += 1;
            }
        }
        Ok(finished)
    }

    /// Withdraw a request, whether queued or resident.
    ///
    /// Returns whether anything was found. A resident request hands its pages
    /// straight back, so the slot and its memory are available to the next
    /// admission on the same step -- an abandoned generation must not keep
    /// occupying the batch until it reaches max_tokens.
    ///
    /// Cancellation takes effect between steps, never inside one: a step is a
    /// single fused GPU graph launch and cannot be interrupted partway.
    pub fn cancel(&mut self, id: u64) -> Result<bool> {
        if let Some(pos) = self.pending.iter().position(|r| r.id == id) {
            self.pending.remove(pos);
            return Ok(true);
        }
        // Partway through its prompt: it holds pages and has written some of
        // them, but has never produced a token. Removing it here is what stops
        // any further chunk being scheduled for it, and the pages go back in
        // the same call, so a client that leaves mid-prefill costs exactly the
        // work already done and nothing more.
        if let Some(pos) = self.prefilling.iter().position(|p| p.id == id) {
            let mut p = self.prefilling.remove(pos).expect("position found above");
            p.seq.release(self.model.page_pool_mut())?;
            self.page_reservations.release(p.reserved_pages)?;
            self.done.push(Completion {
                id: p.id,
                prompt_len: p.prompt.len(),
                tokens: Vec::new(),
                finished_at: self.step_no,
                reason: FinishReason::Cancelled,
            });
            return Ok(true);
        }
        if let Some(pos) = self.active.iter().position(|a| a.id == id) {
            let mut a = self.active.swap_remove(pos);
            a.seq.release(self.model.page_pool_mut())?;
            self.page_reservations.release(a.reserved_pages)?;
            self.done.push(Completion {
                id: a.id,
                prompt_len: a.prompt_len,
                tokens: a.generated,
                finished_at: self.step_no,
                reason: FinishReason::Cancelled,
            });
            return Ok(true);
        }
        Ok(false)
    }

    /// Ids of the requests currently resident, in slot order.
    pub fn active_ids(&self) -> Vec<u64> {
        self.active.iter().map(|a| a.id).collect()
    }

    /// Run until every submitted request has finished.
    pub fn run_to_completion(&mut self, max_steps: usize) -> Result<Vec<StepInfo>> {
        let mut steps = Vec::new();
        while !self.is_idle() {
            if steps.len() >= max_steps {
                bail!("runtime did not drain within {max_steps} steps");
            }
            steps.push(self.step()?);
        }
        Ok(steps)
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
