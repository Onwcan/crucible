//! Continuous-batching runtime: many requests resident at once, entering and
//! leaving while the others keep decoding.
//!
//! The engine underneath is unchanged -- same weights, same kernels, same
//! arithmetic. What changes is that a decode step carries one row per active
//! request instead of one row total, and that a request's KV lives in pages
//! that can be handed back the moment it finishes.
//!
//! # Why this is not "batching"
//!
//! Static batching runs a fixed group to completion: a batch of four where one
//! request wants 8 tokens and another wants 300 spends most of its life
//! computing padding for the request that already finished. Continuous batching
//! removes a finished request from the batch immediately, reclaims its pages,
//! and admits a waiting one in its place. Requests never have to start
//! together, and lengths are never padded to the longest member -- each
//! request's attention loop is bounded by its own length.
//!
//! # Scheduling policy
//!
//! First-come-first-served admission, bounded by two things: `max_batch` slots
//! and free pages. A request that cannot get pages stays pending rather than
//! failing, so the pool acts as backpressure instead of an error surface. This
//! is deliberately the simplest policy that exercises the machinery; anything
//! cleverer (priorities, preemption, prefix sharing) is a scheduling question,
//! not a correctness one, and belongs after this is measured.

use anyhow::{bail, Result};
use std::collections::VecDeque;

use crate::gpu_model::GpuModel;
use crate::paged::SequencePages;
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

/// One resident request.
struct Active {
    id: u64,
    seq: SequencePages,
    prompt_len: usize,
    /// The token to feed at the next decode step.
    next_token: usize,
    generated: Vec<usize>,
    config: GenerationConfig,
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
}

pub struct Runtime {
    model: GpuModel,
    active: Vec<Active>,
    pending: VecDeque<Request>,
    done: Vec<Completion>,
    step_no: u64,
    max_batch: usize,
}

impl Runtime {
    /// `model` must already have paging enabled; the pool it allocated is the
    /// runtime's entire memory budget.
    pub fn new(model: GpuModel) -> Result<Self> {
        if !model.use_paged() {
            bail!("runtime requires a model with paging enabled");
        }
        let max_batch = model.max_batch();
        Ok(Self {
            model,
            active: Vec::new(),
            pending: VecDeque::new(),
            done: Vec::new(),
            step_no: 0,
            max_batch,
        })
    }

    pub fn submit(&mut self, req: Request) {
        self.pending.push_back(req);
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.pending.is_empty()
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
    pub fn residency(&self) -> (usize, usize) {
        let pages = self.active.iter().map(|a| a.seq.n_pages()).sum();
        let wasted = self.active.iter().map(|a| a.seq.wasted_slots()).sum();
        (pages, wasted)
    }

    /// Admit what fits, decode every active request one token, retire whoever
    /// finished.
    pub fn step(&mut self) -> Result<StepInfo> {
        let mut info = StepInfo {
            step: self.step_no,
            ..Default::default()
        };

        info.admitted = self.admit(&mut info.tokens)?;

        if !self.active.is_empty() {
            let (decoded, d2h) = self.decode_active(&mut info.tokens)?;
            info.decoded = decoded;
            info.d2h_bytes = d2h;
        }
        info.finished = self.retire()?;

        info.active_after = self.active.len();
        info.pending_after = self.pending.len();
        info.free_pages = self.model.page_pool().free_pages();
        self.step_no += 1;
        Ok(info)
    }

    /// Move pending requests into the active set while slots and pages allow.
    ///
    /// Admission prefills the prompt, which is where a new request's pages are
    /// taken. A prompt that does not fit leaves the request pending and stops
    /// admission for this step -- FCFS, so a large request is not starved by
    /// smaller ones queued behind it.
    fn admit(&mut self, first_tokens: &mut Vec<(u64, usize)>) -> Result<Vec<u64>> {
        let mut admitted = Vec::new();
        while self.active.len() < self.max_batch {
            let Some(req) = self.pending.front() else { break };
            if req.prompt.is_empty() {
                bail!("request {} has an empty prompt", req.id);
            }

            let mut seq = SequencePages::new();
            if seq.grow(self.model.page_pool_mut(), req.prompt.len()).is_err() {
                // Not enough pages right now. Leave it queued; a retirement
                // later this step or next will free some.
                break;
            }

            let req = self.pending.pop_front().expect("front checked above");
            let table = seq.table_padded(self.model.table_stride());
            let logits = self.model.prefill_request(&req.prompt, &table, 0)?;
            // Prefill's final logits are this request's first token, so its RNG
            // is created here and used immediately -- the first sampled token
            // draws the first random number, exactly as running alone would.
            let mut rng = Rng::new(req.config.seed);
            let first = sampling::sample(&logits, &req.config, &mut rng);

            admitted.push(req.id);
            first_tokens.push((req.id, first));
            self.active.push(Active {
                id: req.id,
                seq,
                prompt_len: req.prompt.len(),
                next_token: first,
                generated: vec![first],
                config: req.config,
                rng,
            });
        }
        Ok(admitted)
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
        let (next, d2h): (Vec<usize>, usize) =
            if topk_rows.is_empty() && full_rows.is_empty() && self.model.device_argmax() {
                // Unchanged greedy fast path: n * 4 bytes back, no logits move.
                let ids = self
                    .model
                    .decode_batch_tokens(&tokens, &positions, &tables, &lens)?;
                let bytes = n * std::mem::size_of::<i32>();
                (ids, bytes)
            } else {
                let sel = self.model.decode_batch_mixed(
                    &tokens, &positions, &tables, &lens, &topk_rows, &full_rows,
                )?;
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
        if let Some(pos) = self.active.iter().position(|a| a.id == id) {
            let mut a = self.active.swap_remove(pos);
            a.seq.release(self.model.page_pool_mut())?;
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
