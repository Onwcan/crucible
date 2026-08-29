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

/// Work submitted to the runtime.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<usize>,
    pub max_new_tokens: usize,
}

/// A request that has left the active set.
#[derive(Debug, Clone)]
pub struct Completion {
    pub id: u64,
    pub prompt_len: usize,
    pub tokens: Vec<usize>,
    /// Step index at which it left the batch, for deterministic tests.
    pub finished_at: u64,
}

/// One resident request.
struct Active {
    id: u64,
    seq: SequencePages,
    prompt_len: usize,
    /// The token to feed at the next decode step.
    next_token: usize,
    generated: Vec<usize>,
    max_new: usize,
}

/// What one `step` did, so a test or benchmark can assert on it.
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    pub step: u64,
    pub admitted: Vec<u64>,
    pub decoded: usize,
    pub finished: Vec<u64>,
    pub active_after: usize,
    pub pending_after: usize,
    pub free_pages: usize,
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

        info.admitted = self.admit()?;

        if !self.active.is_empty() {
            info.decoded = self.decode_active()?;
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
    fn admit(&mut self) -> Result<Vec<u64>> {
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
            let first = argmax(&logits);

            admitted.push(req.id);
            self.active.push(Active {
                id: req.id,
                seq,
                prompt_len: req.prompt.len(),
                next_token: first,
                generated: vec![first],
                max_new: req.max_new_tokens,
            });
        }
        Ok(admitted)
    }

    /// One batched decode step across every active request.
    fn decode_active(&mut self) -> Result<usize> {
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

        let vocab = self.model.cfg.vocab_size;
        let logits = self.model.decode_batch(&tokens, &positions, &tables, &lens)?;

        for (i, a) in self.active.iter_mut().enumerate() {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let next = argmax(row);
            a.next_token = next;
            a.generated.push(next);
        }
        Ok(n)
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
            if self.active[i].generated.len() >= self.active[i].max_new {
                let mut a = self.active.swap_remove(i);
                a.seq.release(self.model.page_pool_mut())?;
                finished.push(a.id);
                self.done.push(Completion {
                    id: a.id,
                    prompt_len: a.prompt_len,
                    tokens: a.generated,
                    finished_at: self.step_no,
                });
            } else {
                i += 1;
            }
        }
        Ok(finished)
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
