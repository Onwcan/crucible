//! Deterministic, CUDA-independent planning for packed prompt work.
//!
//! The runtime supplies its round-robin queue in order. Each plan visits each
//! request at most once; after execution, unfinished slices move to the queue's
//! back and completed slices leave it. New admissions also join at the back.
//! With at most R resident requests and a positive budget, a waiting survivor
//! therefore receives work within R nonempty plans, including under arrivals
//! and cancellations. A plan carries request identities, never resident slots.

use anyhow::{bail, Result};

/// Validate a direct runtime submission before it can own pages or reach CUDA.
pub fn validate_request(
    prompt: &[usize],
    config: &crate::sampling::GenerationConfig,
    vocab: usize,
    context: usize,
    pool_pages: usize,
) -> Result<()> {
    crate::sampling::validate(config, vocab).map_err(anyhow::Error::msg)?;
    let pages = request_page_reservation(prompt.len(), config.max_tokens, context)?;
    if pages > pool_pages {
        bail!("request needs {pages} KV pages but pool capacity is {pool_pages}");
    }
    if let Some(token) = prompt
        .iter()
        .find(|&&id| id >= vocab || id > i32::MAX as usize)
    {
        bail!("request token {token} outside vocabulary {vocab}");
    }
    Ok(())
}

/// IDs may be reused after completion, but may never name two live requests.
pub fn validate_request_identity(id: u64, live: impl IntoIterator<Item = u64>) -> Result<()> {
    if live.into_iter().any(|other| other == id) {
        bail!("request id {id} is already queued or resident");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillWork {
    pub request_id: u64,
    pub prompt_len: usize,
    pub done: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillSlice {
    pub request_id: u64,
    pub packed_row_start: usize,
    pub prompt_start: usize,
    pub len: usize,
    pub is_final: bool,
}

/// All limits apply to real token rows; no semantic padding is introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillBudget {
    pub tokens: usize,
    pub requests: usize,
    pub chunk: usize,
}

impl PrefillBudget {
    pub fn validate(self, token_capacity: usize, request_capacity: usize) -> Result<()> {
        if self.tokens == 0 || self.tokens > token_capacity {
            bail!(
                "prefill token budget must be in 1..={token_capacity}, got {}",
                self.tokens
            );
        }
        if self.requests == 0 || self.requests > request_capacity {
            bail!(
                "prefill request limit must be in 1..={request_capacity}, got {}",
                self.requests
            );
        }
        if self.chunk == 0 || self.chunk > token_capacity {
            bail!(
                "prefill chunk cap must be in 1..={token_capacity}, got {}",
                self.chunk
            );
        }
        Ok(())
    }
}

/// Reused by the runtime so constructing a batch does not allocate per token.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PrefillBatchPlan {
    pub slices: Vec<PrefillSlice>,
    pub tokens: usize,
    pub final_rows: usize,
}

impl PrefillBatchPlan {
    pub fn with_capacity(requests: usize) -> Self {
        Self {
            slices: Vec::with_capacity(requests),
            ..Self::default()
        }
    }

    pub fn build(
        &mut self,
        work: impl IntoIterator<Item = PrefillWork>,
        budget: PrefillBudget,
    ) -> Result<()> {
        self.slices.clear();
        self.tokens = 0;
        self.final_rows = 0;
        if budget.tokens == 0 || budget.requests == 0 || budget.chunk == 0 {
            bail!("prefill planning limits must all be positive");
        }
        for request in work {
            if self.slices.len() == budget.requests || self.tokens == budget.tokens {
                break;
            }
            if request.done >= request.prompt_len {
                bail!(
                    "request {} has invalid prefill progress {}/{}",
                    request.request_id,
                    request.done,
                    request.prompt_len
                );
            }
            if self
                .slices
                .iter()
                .any(|s| s.request_id == request.request_id)
            {
                bail!(
                    "request {} appears twice in a prefill plan",
                    request.request_id
                );
            }
            let remaining = request.prompt_len - request.done;
            let len = remaining.min(budget.chunk).min(budget.tokens - self.tokens);
            let is_final = len == remaining;
            self.slices.push(PrefillSlice {
                request_id: request.request_id,
                packed_row_start: self.tokens,
                prompt_start: request.done,
                len,
                is_final,
            });
            self.tokens += len;
            self.final_rows += usize::from(is_final);
        }
        Ok(())
    }
}

/// Pages a request can need through its final generated token. The first token
/// comes from prefill logits; only the remaining output tokens require decode
/// writes. Admission reserves this maximum logically while allocating pages
/// lazily, preventing prompt admissions from spending a decoder's future pages.
pub fn request_page_reservation(prompt: usize, generated: usize, context: usize) -> Result<usize> {
    if prompt == 0 || generated == 0 {
        bail!("prompt length and output token budget must be positive");
    }
    let positions = prompt
        .checked_add(generated - 1)
        .ok_or_else(|| anyhow::anyhow!("request token count overflows usize"))?;
    if positions > context {
        bail!("request needs {positions} cache positions, capacity is {context}");
    }
    Ok(positions.div_ceil(crate::paged::PAGE_TOKENS))
}

/// Admission's logical page budget, independent of physical lazy allocation.
#[derive(Debug)]
pub struct PageReservations {
    capacity: usize,
    held: usize,
}

impl PageReservations {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, held: 0 }
    }

    pub fn try_reserve(&mut self, pages: usize) -> Result<bool> {
        if pages == 0 || pages > self.capacity {
            bail!(
                "request needs {pages} KV pages but pool capacity is {}",
                self.capacity
            );
        }
        if pages > self.capacity - self.held {
            return Ok(false);
        }
        self.held += pages;
        Ok(true)
    }

    pub fn release(&mut self, pages: usize) -> Result<()> {
        self.held = self
            .held
            .checked_sub(pages)
            .ok_or_else(|| anyhow::anyhow!("released more KV reservation than held"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn work(id: u64, len: usize, done: usize) -> PrefillWork {
        PrefillWork {
            request_id: id,
            prompt_len: len,
            done,
        }
    }

    fn plan(requests: &[PrefillWork], tokens: usize, chunk: usize) -> PrefillBatchPlan {
        let mut p = PrefillBatchPlan::default();
        p.build(
            requests.iter().copied(),
            PrefillBudget {
                tokens,
                requests: 16,
                chunk,
            },
        )
        .unwrap();
        p
    }

    #[test]
    fn exact_fit_and_one_token_over_have_contiguous_rows() {
        let input = [work(9, 16, 0), work(2, 17, 0)];
        let exact = plan(&input, 33, 33);
        assert_eq!((exact.tokens, exact.final_rows), (33, 2));
        assert_eq!(exact.slices[1].packed_row_start, 16);
        let bounded = plan(&input, 32, 32);
        assert_eq!((bounded.tokens, bounded.final_rows), (32, 1));
        assert_eq!(bounded.slices[1].len, 16);
        assert!(!bounded.slices[1].is_final);
    }

    #[test]
    fn many_tiny_requests_share_one_plan() {
        let input: Vec<_> = (0..16).map(|id| work(id, 8, 0)).collect();
        let p = plan(&input, 128, 64);
        assert_eq!((p.tokens, p.slices.len(), p.final_rows), (128, 16, 16));
        assert!(p
            .slices
            .iter()
            .enumerate()
            .all(|(i, s)| s.packed_row_start == i * 8));
    }

    #[test]
    fn huge_request_cannot_take_more_than_its_cap() {
        let p = plan(&[work(1, 941, 0), work(2, 8, 0), work(3, 17, 0)], 128, 64);
        assert_eq!((p.tokens, p.final_rows), (89, 2));
        assert_eq!(p.slices[0].len, 64);
    }

    #[test]
    fn offsets_and_final_rows_belong_to_request_identity() {
        let p = plan(
            &[work(90, 130, 73), work(12, 65, 64), work(70, 200, 37)],
            128,
            64,
        );
        assert_eq!(
            p.slices[0],
            PrefillSlice {
                request_id: 90,
                packed_row_start: 0,
                prompt_start: 73,
                len: 57,
                is_final: true,
            }
        );
        assert_eq!(p.slices[1].packed_row_start, 57);
        assert_eq!(p.slices[2].prompt_start, 37);
        assert_eq!(p.final_rows, 2);
    }

    #[test]
    fn request_limit_and_reused_plan_respect_active_counts() {
        let input: Vec<_> = (0..16).map(|id| work(id, 8, 0)).collect();
        let mut p = plan(&input, 128, 64);
        p.build(
            [work(45, 17, 0)],
            PrefillBudget {
                tokens: 32,
                requests: 2,
                chunk: 32,
            },
        )
        .unwrap();
        assert_eq!((p.tokens, p.slices.len(), p.final_rows), (17, 1, 1));
        p.build(
            input,
            PrefillBudget {
                tokens: 128,
                requests: 2,
                chunk: 64,
            },
        )
        .unwrap();
        assert_eq!((p.tokens, p.slices.len()), (16, 2));
    }

    #[test]
    fn fairness_survives_arrivals_cancellation_and_completion() {
        let mut queue = VecDeque::from([work(1, 941, 0), work(2, 8, 0), work(3, 3, 0)]);
        let mut seen_long = Vec::new();
        let mut p = PrefillBatchPlan::with_capacity(4);
        for step in 0..32 {
            // Cancel a waiting peer and continuously insert younger peers.
            if step == 2 {
                queue.retain(|r| r.request_id != 2);
            }
            if queue.len() < 4 {
                queue.push_back(work(100 + step, 1, 0));
            }
            p.build(
                queue.iter().copied(),
                PrefillBudget {
                    tokens: 1,
                    requests: 4,
                    chunk: 1,
                },
            )
            .unwrap();
            let slice = p.slices[0];
            let mut r = queue.pop_front().unwrap();
            assert_eq!(r.request_id, slice.request_id);
            if r.request_id == 1 {
                seen_long.push(step);
            }
            r.done += slice.len;
            if !slice.is_final {
                queue.push_back(r);
            }
        }
        assert!(seen_long.len() >= 8);
        assert!(seen_long.windows(2).all(|pair| pair[1] - pair[0] <= 4));
    }

    #[test]
    fn identical_input_produces_identical_plan() {
        let input = [work(42, 33, 17), work(7, 941, 128), work(99, 15, 0)];
        assert_eq!(plan(&input, 128, 64), plan(&input, 128, 64));
    }

    #[test]
    fn malformed_work_and_zero_or_excessive_limits_are_rejected() {
        let valid = PrefillBudget {
            tokens: 128,
            requests: 16,
            chunk: 64,
        };
        assert!(valid.validate(1024, 16).is_ok());
        for bad in [
            PrefillBudget { tokens: 0, ..valid },
            PrefillBudget {
                requests: 0,
                ..valid
            },
            PrefillBudget { chunk: 0, ..valid },
            PrefillBudget {
                tokens: 1025,
                ..valid
            },
            PrefillBudget {
                requests: 17,
                ..valid
            },
            PrefillBudget {
                chunk: 1025,
                ..valid
            },
        ] {
            assert!(bad.validate(1024, 16).is_err());
        }
        let mut p = PrefillBatchPlan::default();
        assert!(p.build([work(1, 0, 0)], valid).is_err());
        assert!(p.build([work(1, 8, 8)], valid).is_err());
        assert!(p.build([work(1, 8, 0), work(1, 8, 0)], valid).is_err());
    }

    #[test]
    fn reservations_include_decode_growth_and_one_token_completion() {
        assert_eq!(request_page_reservation(16, 1, 1024).unwrap(), 1);
        assert_eq!(request_page_reservation(16, 2, 1024).unwrap(), 2);
        assert_eq!(request_page_reservation(17, 16, 1024).unwrap(), 2);
        assert_eq!(request_page_reservation(17, 17, 1024).unwrap(), 3);
        assert!(request_page_reservation(0, 1, 1024).is_err());
        assert!(request_page_reservation(1, 0, 1024).is_err());
        assert!(request_page_reservation(1024, 2, 1024).is_err());
        assert!(request_page_reservation(usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn page_pressure_keeps_future_decode_pages_and_cancellation_returns_budget() {
        use crate::paged::{PagePool, SequencePages};
        let mut pool = PagePool::new(4, 12, 192);
        let mut reservations = PageReservations::new(pool.n_pages());
        let mut first = SequencePages::new();
        let mut second = SequencePages::new();
        // Two physically tiny prompts each need two pages through completion.
        assert!(reservations
            .try_reserve(request_page_reservation(16, 17, 1024).unwrap())
            .unwrap());
        first.grow(&mut pool, 16).unwrap();
        assert!(reservations
            .try_reserve(request_page_reservation(16, 17, 1024).unwrap())
            .unwrap());
        second.grow(&mut pool, 16).unwrap();
        assert_eq!(pool.free_pages(), 2);
        assert!(
            !reservations.try_reserve(1).unwrap(),
            "physically free pages belong to decoder growth"
        );
        first.grow(&mut pool, 1).unwrap();
        second.grow(&mut pool, 1).unwrap();
        assert_eq!(pool.free_pages(), 0);
        first.release(&mut pool).unwrap();
        reservations.release(2).unwrap();
        assert!(reservations.try_reserve(2).unwrap());
        let mut replacement = SequencePages::new();
        replacement.grow(&mut pool, 32).unwrap();
        assert!(replacement
            .pages()
            .iter()
            .all(|page| !second.pages().contains(page)));
        assert!(!reservations.try_reserve(1).unwrap());
    }

    #[test]
    fn impossible_and_failed_reservations_do_not_leak_capacity() {
        let mut reservations = PageReservations::new(4);
        assert!(reservations.try_reserve(5).is_err());
        assert!(reservations.try_reserve(0).is_err());
        assert!(reservations.try_reserve(3).unwrap());
        assert!(!reservations.try_reserve(2).unwrap());
        reservations.release(3).unwrap();
        assert!(reservations.try_reserve(4).unwrap());
        assert!(reservations.release(5).is_err());
        reservations.release(4).unwrap();
        assert!(reservations.try_reserve(4).unwrap());
    }

    #[test]
    fn invalid_submissions_are_rejected_before_page_admission() {
        use crate::sampling::GenerationConfig;
        let config = GenerationConfig::greedy(1);
        assert!(validate_request(&[1, 99], &config, 100, 1024, 64).is_ok());
        assert!(validate_request(&[], &config, 100, 1024, 64).is_err());
        assert!(validate_request(&[100], &config, 100, 1024, 64).is_err());
        assert!(validate_request(&[usize::MAX], &config, usize::MAX, 1024, 64).is_err());
        assert!(validate_request(&[1], &GenerationConfig::greedy(0), 100, 1024, 64).is_err());
        assert!(validate_request(&[1; 16], &GenerationConfig::greedy(2), 100, 1024, 1).is_err());
        assert!(validate_request(&[1; 16], &GenerationConfig::greedy(2), 100, 16, 64).is_err());
        let bad_sampling = GenerationConfig {
            temperature: f32::NAN,
            ..config
        };
        assert!(validate_request(&[1], &bad_sampling, 100, 1024, 64).is_err());
    }

    #[test]
    fn identity_checks_cover_all_states_and_allow_reuse_after_retirement() {
        let pending = [42, 100];
        let prefilling = [9, 87];
        let active = [5, 16];
        for duplicate in [42, 100, 9, 87, 5, 16] {
            assert!(validate_request_identity(
                duplicate,
                pending.into_iter().chain(prefilling).chain(active)
            )
            .is_err());
        }
        // Slot permutation and removal have no bearing on identity.
        assert!(validate_request_identity(16, [5, 16]).is_err());
        assert!(validate_request_identity(5, [16]).is_ok());
        assert!(validate_request_identity(999, []).is_ok());
    }
}
